"""SDMA / no-common-stream ablation.

``multiple_access='sdma'`` removes the common stream from the PHY entirely: the
whole power budget goes to the private beams, every user's payload is private
and there is no SIC.  Comparing it against RSMA isolates the *rate-splitting*
gain from the scheduling gain.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rsma_aoi.baselines.common import default_num_scheduled, equal_split, make_action, top_m_mask

#: Environment overrides that turn any config into its SDMA counterpart.
ENV_OVERRIDES: dict[str, Any] = {"multiple_access": "sdma"}


class SDMAPolicy:
    """Max-AoI-first scheduling with equal private power (no common stream)."""

    name = "sdma"

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
            raise ValueError("SDMAPolicy requires the structured environment state")
        actions = []
        for state in states:
            priority = np.asarray(state["weights"]) * np.asarray(state["aoi"])
            active = top_m_mask(priority, self.num_scheduled)
            pf, rf = equal_split(active, common_fraction=0.0)
            actions.append(
                make_action(active, pf, rf, 0.0, 1.0 if self.power_scale_action else None)
            )
        return actions
