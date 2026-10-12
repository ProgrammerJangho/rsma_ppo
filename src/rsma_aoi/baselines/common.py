"""Shared helpers for the non-learning baselines.

Every baseline emits exactly the same feasible action dict as the PPO policy,
so all algorithms are evaluated through the identical PHY / AoI pipeline.  The
only differences between OMA / SDMA / NOMA and RSMA are *environment* settings
(``multiple_access``), never a different physical model implemented twice.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class Policy(Protocol):
    """Interface shared by PPO and every baseline."""

    name: str

    def act(
        self, obs: np.ndarray, states: list[dict[str, Any]] | None = None
    ) -> list[dict[str, np.ndarray]]:
        """Map a batch of observations (and structured states) to actions."""
        ...


def make_action(
    active: np.ndarray,
    power_fractions: np.ndarray,
    rate_fractions: np.ndarray,
    common_payload_scale: float,
    power_scale: float | None = None,
) -> dict[str, np.ndarray]:
    """Assemble one environment action dict (no validation beyond dtypes)."""
    action = {
        "schedule": np.asarray(active, dtype=np.int8),
        "power_fractions": np.asarray(power_fractions, dtype=np.float64),
        "rate_fractions": np.asarray(rate_fractions, dtype=np.float64),
        "common_payload_scale": np.asarray([common_payload_scale], dtype=np.float64),
    }
    if power_scale is not None:
        action["power_scale"] = np.asarray([power_scale], dtype=np.float64)
    return action


def make_direct_action(
    active: np.ndarray,
    power_fractions: np.ndarray,
    common_payload_fractions: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assemble the canonical gate/rho-free ``alpha, q_k, phi_k`` action."""
    active = np.asarray(active, dtype=bool)
    power = np.asarray(power_fractions, dtype=np.float64)
    alpha = float(np.clip(power[0], 0.0, 1.0))
    private = np.where(active, np.clip(power[1:], 0.0, None), 0.0)
    if private.sum() <= 1e-12:
        private = active.astype(np.float64)
    q = private / max(private.sum(), 1e-12)
    return {
        "schedule": active.astype(np.int8),
        "common_power_fraction": np.asarray([alpha], dtype=np.float64),
        "private_power_fractions": q,
        "common_payload_fractions": np.asarray(common_payload_fractions, dtype=np.float64),
    }


def repair_schedule(active: np.ndarray, priority: np.ndarray) -> np.ndarray:
    """Guarantee at least one active user (highest-priority user wins)."""
    active = np.asarray(active, dtype=bool)
    if not active.any():
        active = np.zeros_like(active)
        active[int(np.argmax(priority))] = True
    return active


def equal_split(active: np.ndarray, common_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Equal power/rate allocation over the active users.

    Returns:
        ``(power_fractions (K+1,), rate_fractions (K,))`` with the common stream
        receiving ``common_fraction`` of the budget and the remainder shared
        equally by the active private streams.
    """
    active = np.asarray(active, dtype=bool)
    k = active.shape[0]
    num_active = max(int(active.sum()), 1)
    pf = np.zeros(k + 1, dtype=np.float64)
    pf[0] = float(common_fraction)
    pf[1:][active] = (1.0 - float(common_fraction)) / num_active
    rf = np.zeros(k, dtype=np.float64)
    rf[active] = 1.0 / num_active
    return pf, rf


def top_m_mask(priority: np.ndarray, num_scheduled: int) -> np.ndarray:
    """Boolean mask selecting the ``num_scheduled`` highest-priority users."""
    priority = np.asarray(priority, dtype=np.float64)
    k = priority.shape[0]
    m = int(np.clip(num_scheduled, 1, k))
    order = np.argsort(-priority, kind="stable")
    mask = np.zeros(k, dtype=bool)
    mask[order[:m]] = True
    return mask


def default_num_scheduled(num_users: int, num_antennas: int) -> int:
    """Default schedule size: the spatial multiplexing limit ``min(K, Nt)``."""
    return int(max(1, min(num_users, num_antennas)))


class EqualAllocationPolicy:
    """Keep another policy's *schedule* but replace its allocation with an equal split.

    Used by the RSMA-contribution ablation to separate the MAC gain (which users
    are served) from the PHY gain (how the power and payload are split).
    """

    def __init__(self, base: Any, common_fraction: float = 0.0,
                 common_payload_scale: float = 0.0) -> None:
        self.base = base
        self.common_fraction = float(common_fraction)
        self.common_payload_scale = float(common_payload_scale)
        self.name = f"{getattr(base, 'name', 'policy')}+equal_alloc"

    def reset(self, seed: int | None = None) -> None:
        if hasattr(self.base, "reset"):
            self.base.reset(seed=seed)

    def act(
        self, obs: np.ndarray, states: list[dict[str, Any]] | None = None
    ) -> list[dict[str, np.ndarray]]:
        actions = self.base.act(obs, states)
        out = []
        for action in actions:
            active = np.asarray(action["schedule"], dtype=bool)
            pf, rf = equal_split(active, self.common_fraction)
            new = make_action(active, pf, rf, self.common_payload_scale)
            if "power_scale" in action:
                new["power_scale"] = action["power_scale"]
            out.append(new)
        return out
