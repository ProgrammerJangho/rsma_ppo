"""NOMA (single-beam SC-SIC) heuristic baseline.

``multiple_access='noma'`` superposes all scheduled users on one beam and
applies successive interference cancellation in ascending effective-gain order.
The classic power heuristic gives *more* power to the weaker users, which this
policy implements as ``P_k propto (w_k Delta_k) / g_k``.

ASSUMPTION: the beam is the same "common direction" heuristic used by RSMA, so
the only difference from RSMA is the absence of a rate-split layer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rsma_aoi.baselines.common import default_num_scheduled, make_action, top_m_mask

#: Environment overrides that turn any config into its NOMA counterpart.
ENV_OVERRIDES: dict[str, Any] = {"multiple_access": "noma"}

_EPS = 1e-12


class NOMAPolicy:
    """Max-AoI-first scheduling with inverse-gain power allocation."""

    name = "noma"

    def __init__(
        self,
        num_users: int,
        num_antennas: int,
        num_scheduled: int | None = None,
        power_scale_action: bool = False,
    ) -> None:
        self.num_users = num_users
        self.num_scheduled = (
            default_num_scheduled(num_users, num_antennas)
            if num_scheduled is None
            else int(num_scheduled)
        )
        self.power_scale_action = power_scale_action

    def reset(self, seed: int | None = None) -> None:
        return None

    def act(
        self, obs: np.ndarray, states: list[dict[str, Any]] | None = None
    ) -> list[dict[str, np.ndarray]]:
        if states is None:
            raise ValueError("NOMAPolicy requires the structured environment state")
        actions = []
        for state in states:
            priority = np.asarray(state["weights"]) * np.asarray(state["aoi"])
            active = top_m_mask(priority, self.num_scheduled)
            gains = np.sum(np.abs(np.asarray(state["h_hat"])) ** 2, axis=1)
            weights = np.where(active, priority / np.clip(gains, _EPS, None), 0.0)
            total = weights.sum()
            share = (
                weights / total
                if total > _EPS
                else active.astype(np.float64) / max(int(active.sum()), 1)
            )
            k = self.num_users
            pf = np.zeros(k + 1, dtype=np.float64)
            pf[1:] = share
            rf = np.zeros(k, dtype=np.float64)
            rf[active] = 1.0 / max(int(active.sum()), 1)
            actions.append(
                make_action(active, pf, rf, 0.0, 1.0 if self.power_scale_action else None)
            )
        return actions
