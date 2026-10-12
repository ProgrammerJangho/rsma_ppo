"""Equal-power RSMA: all users active, fixed common fraction, equal splits."""

from __future__ import annotations

from typing import Any

import numpy as np

from rsma_aoi.baselines.common import equal_split, make_action


class EqualPowerPolicy:
    """Schedule everyone and share the power budget equally.

    Args:
        num_users: K.
        common_fraction: share of ``P_max`` given to the common stream.
        common_payload_scale: fixed ``rho_r``.
        power_scale_action: emit the optional ``rho_P`` component.
    """

    name = "equal_power"

    def __init__(
        self,
        num_users: int,
        common_fraction: float = 0.2,
        common_payload_scale: float = 0.5,
        power_scale_action: bool = False,
    ) -> None:
        self.num_users = num_users
        self.common_fraction = float(common_fraction)
        self.common_payload_scale = float(common_payload_scale)
        self.power_scale_action = power_scale_action

    def reset(self, seed: int | None = None) -> None:
        return None

    def act(
        self, obs: np.ndarray, states: list[dict[str, Any]] | None = None
    ) -> list[dict[str, np.ndarray]]:
        active = np.ones(self.num_users, dtype=bool)
        pf, rf = equal_split(active, self.common_fraction)
        action = make_action(
            active, pf, rf, self.common_payload_scale,
            1.0 if self.power_scale_action else None,
        )
        return [dict(action) for _ in range(obs.shape[0])]
