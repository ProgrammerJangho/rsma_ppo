"""Max-AoI-first scheduling with an equal feasible allocation.

This is the strongest simple MAC heuristic for AoI: schedule the ``m`` users
with the largest ``w_k * Delta_k`` and split the remaining budget equally.  It
isolates the *MAC* effect, so comparing it against PPO separates scheduling
gains from PHY (rate-splitting) gains.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rsma_aoi.baselines.common import default_num_scheduled, equal_split, make_action, top_m_mask


class MaxAoIFirstPolicy:
    """Greedy weighted-AoI scheduler."""

    name = "max_aoi_first"

    def __init__(
        self,
        num_users: int,
        num_antennas: int,
        num_scheduled: int | None = None,
        common_fraction: float = 0.0,
        common_payload_scale: float = 0.0,
        power_scale_action: bool = False,
        use_channel_gain: bool = False,
    ) -> None:
        self.num_users = num_users
        self.num_scheduled = (
            default_num_scheduled(num_users, num_antennas)
            if num_scheduled is None
            else int(num_scheduled)
        )
        self.common_fraction = float(common_fraction)
        self.common_payload_scale = float(common_payload_scale)
        self.power_scale_action = power_scale_action
        self.use_channel_gain = use_channel_gain

    def reset(self, seed: int | None = None) -> None:
        return None

    def _priority(self, state: dict[str, Any]) -> np.ndarray:
        priority = np.asarray(state["weights"], dtype=np.float64) * np.asarray(
            state["aoi"], dtype=np.float64
        )
        if self.use_channel_gain:
            gains = np.sum(np.abs(np.asarray(state["h_hat"])) ** 2, axis=1)
            priority = priority * gains
        return priority

    def act(
        self, obs: np.ndarray, states: list[dict[str, Any]] | None = None
    ) -> list[dict[str, np.ndarray]]:
        if states is None:
            raise ValueError("MaxAoIFirstPolicy requires the structured environment state")
        actions = []
        for state in states:
            active = top_m_mask(self._priority(state), self.num_scheduled)
            pf, rf = equal_split(active, self.common_fraction)
            actions.append(
                make_action(
                    active, pf, rf, self.common_payload_scale,
                    1.0 if self.power_scale_action else None,
                )
            )
        return actions
