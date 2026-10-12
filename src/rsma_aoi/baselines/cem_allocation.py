"""Online one-step CEM allocation policies for controlled reference studies.

The policies see only the structured transmitter state (AoI and ``h_hat``).
They optimise the exact finite-blocklength one-step expected-AoI reduction and
emit the same action dictionary as PPO.  These are bounded-compute numerical
references, not long-horizon or globally optimal policies.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from rsma_aoi.baselines.common import make_action, make_direct_action
from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer
from rsma_aoi.optimizers import CEMAllocator, make_problem
from rsma_aoi.utils.config import EnvConfig


MODES = ("rsma_raw", "sdma", "rsma_safeguarded")


def _sdma_boundary(power_fractions: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Embed an SDMA private-power allocation in the exact RSMA boundary."""
    private = np.where(active, np.clip(power_fractions[1:], 0.0, None), 0.0)
    if private.sum() <= 1e-12:
        private = active.astype(np.float64)
    private /= private.sum()
    return np.concatenate(([0.0], private))


class CEMAllocationPolicy:
    """All-user online CEM policy with an optional exact SDMA safeguard.

    ``rsma_raw`` and ``sdma`` each use one CEM solve. ``rsma_safeguarded``
    independently solves both problems and selects the larger estimated-channel
    objective, so it uses twice the candidate-evaluation budget.
    """

    def __init__(
        self,
        cfg: EnvConfig,
        mode: str,
        *,
        iterations: int = 5,
        population: int = 192,
        restarts: int = 1,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if cfg.multiple_access != "rsma" or cfg.scheduling_enabled:
            raise ValueError("CEM allocation reference requires all-user RSMA environment")
        self.cfg = replace(cfg, allocator_mode="none")
        self.mode = mode
        self.name = f"cem_{mode}"
        self.rsma_phy = RSMAPhysicalLayer(self.cfg)
        self.sdma_phy = RSMAPhysicalLayer(replace(self.cfg, multiple_access="sdma"))
        self.allocator = CEMAllocator(
            iterations=iterations, population=population, restarts=restarts,
            label=f"cem({restarts}x{iterations}x{population})",
        )
        self.evaluations_per_solve = restarts * iterations * population
        self._base_seed = 0
        self.calls = 0
        self.candidate_evaluations = 0
        self.boundary_selections = 0
        self.selection_history: list[str] = []
        self.last_selection: dict[str, float | str] = {}

    def reset(self, seed: int | None = None) -> None:
        self._base_seed = int(seed or 0)
        self.calls = 0
        self.candidate_evaluations = 0
        self.boundary_selections = 0
        self.selection_history = []
        self.last_selection = {}

    @property
    def boundary_selection_rate(self) -> float:
        return self.boundary_selections / max(self.calls, 1)

    def _problem(self, phy: RSMAPhysicalLayer, state: dict[str, Any]):
        h_hat = np.asarray(state["h_hat"], dtype=np.complex128)
        active = np.ones(self.cfg.num_users, dtype=bool)
        aoi_limit = float(self.cfg.aoi_max) if self.cfg.aoi_cap_enabled else np.inf
        problem, _ = make_problem(
            phy, np.asarray(state["aoi"], dtype=np.float64), h_hat, h_hat,
            np.asarray(state["weights"], dtype=np.float64), aoi_limit, active,
            csit_mode="estimated",
        )
        return active, problem

    def _one(self, state: dict[str, Any]) -> dict[str, np.ndarray]:
        # Episode-randomized SNR changes the physical power budget.  Synchronise
        # both numerical models with the transmitter state before designing RZF
        # beams or scoring allocations.
        total_power = float(state.get("total_power", self.cfg.total_power))
        self.rsma_phy.total_power = total_power
        self.sdma_phy.total_power = total_power
        active, rsma_problem = self._problem(self.rsma_phy, state)
        _, sdma_problem = self._problem(self.sdma_phy, state)
        slot_seed = self._base_seed * 1_000_003 + int(state["step"])
        rsma = sdma = None
        if self.mode in ("rsma_raw", "rsma_safeguarded"):
            rsma = self.allocator.solve(rsma_problem, np.random.default_rng(slot_seed))
            self.candidate_evaluations += rsma.evaluations
        if self.mode in ("sdma", "rsma_safeguarded"):
            sdma = self.allocator.solve(sdma_problem, np.random.default_rng(slot_seed))
            self.candidate_evaluations += sdma.evaluations

        self.calls += 1
        if self.mode == "rsma_raw":
            assert rsma is not None
            chosen = rsma
            selection = "rsma"
            action = make_action(
                active, chosen.power_fractions, chosen.rate_fractions, chosen.rho_r
            )
        elif self.mode == "sdma":
            assert sdma is not None
            selection = "sdma"
            action = make_action(
                active, _sdma_boundary(sdma.power_fractions, active),
                active.astype(np.float64) / active.sum(), 0.0,
            )
        else:
            assert rsma is not None and sdma is not None
            if sdma.gain > rsma.gain:
                self.boundary_selections += 1
                selection = "sdma"
                action = make_action(
                    active, _sdma_boundary(sdma.power_fractions, active),
                    active.astype(np.float64) / active.sum(), 0.0,
                )
            else:
                selection = "rsma"
                action = make_action(
                    active, rsma.power_fractions, rsma.rate_fractions, rsma.rho_r
                )
        if self.cfg.rate_split_mode == "per_user_fraction":
            if selection == "sdma":
                action["rate_fractions"] = np.zeros(self.cfg.num_users, dtype=np.float32)
            action = make_direct_action(
                active, action["power_fractions"], action["rate_fractions"]
            )
        self.last_selection = {
            "selection": selection,
            "rsma_estimated_gain": float(rsma.gain) if rsma is not None else np.nan,
            "sdma_estimated_gain": float(sdma.gain) if sdma is not None else np.nan,
        }
        self.selection_history.append(selection)
        return action

    def act(
        self, obs: np.ndarray, states: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, np.ndarray]]:
        if states is None or len(states) != len(obs):
            raise ValueError("CEM policy requires one structured transmitter state per observation")
        return [self._one(state) for state in states]
