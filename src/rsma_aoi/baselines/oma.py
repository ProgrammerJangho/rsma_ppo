"""OMA (orthogonal / time-division) baseline.

The *physics* difference lives in the environment: ``multiple_access='oma'``
makes the PHY split the ``n`` channel uses among the scheduled users, each
transmitting alone with MRT at full instantaneous power.  This policy only
decides *who* is scheduled and how the time is shared.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rsma_aoi.baselines.common import default_num_scheduled, make_action, top_m_mask

#: Environment overrides that turn any config into its OMA counterpart.
ENV_OVERRIDES: dict[str, Any] = {"multiple_access": "oma"}


class OMAPolicy:
    """Max-AoI-first user selection with proportional or equal time sharing.

    Args:
        time_share: ``'equal'`` splits the slot equally between the scheduled
            users; ``'aoi'`` gives time proportional to ``w_k * Delta_k``.
    """

    name = "oma"

    def __init__(
        self,
        num_users: int,
        num_antennas: int,
        num_scheduled: int | None = None,
        time_share: str = "equal",
        power_scale_action: bool = False,
    ) -> None:
        if time_share not in ("equal", "aoi"):
            raise ValueError("time_share must be 'equal' or 'aoi'")
        self.num_users = num_users
        self.num_scheduled = (
            default_num_scheduled(num_users, num_antennas)
            if num_scheduled is None
            else int(num_scheduled)
        )
        self.time_share = time_share
        self.power_scale_action = power_scale_action

    def reset(self, seed: int | None = None) -> None:
        return None

    def act(
        self, obs: np.ndarray, states: list[dict[str, Any]] | None = None
    ) -> list[dict[str, np.ndarray]]:
        if states is None:
            raise ValueError("OMAPolicy requires the structured environment state")
        actions = []
        for state in states:
            priority = np.asarray(state["weights"]) * np.asarray(state["aoi"])
            active = top_m_mask(priority, self.num_scheduled)
            k = self.num_users
            share = np.zeros(k, dtype=np.float64)
            if self.time_share == "equal":
                share[active] = 1.0 / max(int(active.sum()), 1)
            else:
                weights = np.where(active, priority, 0.0)
                total = weights.sum()
                share = weights / total if total > 0 else active / max(int(active.sum()), 1)
            pf = np.zeros(k + 1, dtype=np.float64)
            pf[1:] = share  # re-interpreted as time fractions by the OMA PHY
            rf = np.zeros(k, dtype=np.float64)
            rf[active] = 1.0 / max(int(active.sum()), 1)
            actions.append(
                make_action(active, pf, rf, 0.0, 1.0 if self.power_scale_action else None)
            )
        return actions
