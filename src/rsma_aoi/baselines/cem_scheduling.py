"""Finite-budget joint subset/allocation CEM references.

Unlike :class:`CEMAllocationPolicy`, this policy enumerates every subset of a
fixed size before selecting an action.  Selection uses only current AoI and
``h_hat``.  It is intended for small diagnostic systems, not scalable online
deployment.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from typing import Any

import numpy as np

from rsma_aoi.baselines.common import make_action
from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer
from rsma_aoi.optimizers import CEMAllocator, make_problem
from rsma_aoi.utils.config import EnvConfig


def _sdma_power(power_fractions: np.ndarray, active: np.ndarray) -> np.ndarray:
    private = np.where(active, np.clip(power_fractions[1:], 0.0, None), 0.0)
    if private.sum() <= 1e-12:
        private = active.astype(np.float64)
    private /= private.sum()
    return np.concatenate(([0.0], private))


class ScheduledCEMAllocationPolicy:
    """Enumerate fixed-size user subsets and optimise allocation for each."""

    def __init__(
        self,
        cfg: EnvConfig,
        mode: str,
        *,
        schedule_size: int,
        iterations: int = 5,
        population: int = 192,
        restarts: int = 1,
    ) -> None:
        if mode not in ("rsma_raw", "sdma", "rsma_safeguarded"):
            raise ValueError("invalid scheduled CEM mode")
        if cfg.multiple_access != "rsma":
            raise ValueError("scheduled CEM expects an RSMA base config")
        if not 1 <= schedule_size <= cfg.num_users:
            raise ValueError("schedule_size must lie in [1, num_users]")
        self.cfg = replace(cfg, allocator_mode="none", scheduling_enabled=True)
        self.mode = mode
        self.schedule_size = int(schedule_size)
        self.name = f"scheduled{schedule_size}_{mode}"
        self.rsma_phy = RSMAPhysicalLayer(self.cfg)
        self.sdma_phy = RSMAPhysicalLayer(replace(self.cfg, multiple_access="sdma"))
        self.allocator = CEMAllocator(
            iterations=iterations, population=population, restarts=restarts,
            label=f"scheduled_cem({restarts}x{iterations}x{population})",
        )
        self._subsets = tuple(combinations(range(cfg.num_users), schedule_size))
        self._base_seed = 0
        self.calls = 0
        self.candidate_evaluations = 0
        self.boundary_selections = 0
        self.subset_history: list[tuple[int, ...]] = []

    def reset(self, seed: int | None = None) -> None:
        self._base_seed = int(seed or 0)
        self.calls = 0
        self.candidate_evaluations = 0
        self.boundary_selections = 0
        self.subset_history = []

    def _problem(self, phy: RSMAPhysicalLayer, state: dict[str, Any], active: np.ndarray):
        h_hat = np.asarray(state["h_hat"], dtype=np.complex128)
        aoi_limit = float(self.cfg.aoi_max) if self.cfg.aoi_cap_enabled else np.inf
        problem, _ = make_problem(
            phy, np.asarray(state["aoi"], dtype=np.float64), h_hat, h_hat,
            np.asarray(state["weights"], dtype=np.float64), aoi_limit, active,
            csit_mode="estimated",
        )
        return problem

    def _one(self, state: dict[str, Any]) -> dict[str, np.ndarray]:
        total_power = float(state.get("total_power", self.cfg.total_power))
        self.rsma_phy.total_power = total_power
        self.sdma_phy.total_power = total_power
        slot_seed = self._base_seed * 1_000_003 + int(state["step"]) * 101
        candidates: list[tuple[float, str, tuple[int, ...], np.ndarray, Any]] = []

        for subset_index, subset in enumerate(self._subsets):
            active = np.zeros(self.cfg.num_users, dtype=bool)
            active[list(subset)] = True
            seed = slot_seed + subset_index * 7_919
            if self.mode in ("rsma_raw", "rsma_safeguarded"):
                problem = self._problem(self.rsma_phy, state, active)
                result = self.allocator.solve(problem, np.random.default_rng(seed))
                estimated = float(problem.evaluate(result.latent[None, :])[0])
                candidates.append((estimated, "rsma", subset, active, result))
                self.candidate_evaluations += result.evaluations
            if self.mode in ("sdma", "rsma_safeguarded"):
                problem = self._problem(self.sdma_phy, state, active)
                result = self.allocator.solve(problem, np.random.default_rng(seed))
                estimated = float(problem.evaluate(result.latent[None, :])[0])
                candidates.append((estimated, "sdma", subset, active, result))
                self.candidate_evaluations += result.evaluations

        _, selection, subset, active, chosen = max(candidates, key=lambda x: x[0])
        self.calls += 1
        self.subset_history.append(subset)
        if selection == "sdma":
            self.boundary_selections += 1
            action = make_action(
                active, _sdma_power(chosen.power_fractions, active),
                active.astype(np.float64) / active.sum(), 0.0,
            )
        else:
            action = make_action(
                active, chosen.power_fractions, chosen.rate_fractions, chosen.rho_r
            )
        return action

    def act(
        self, observations: np.ndarray, states: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, np.ndarray]]:
        if states is None or len(states) != len(observations):
            raise ValueError("scheduled CEM requires structured transmitter states")
        return [self._one(state) for state in states]

