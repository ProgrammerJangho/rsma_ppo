"""Low-complexity AoI baselines used by the thesis evaluation."""
from __future__ import annotations

from typing import Any

import numpy as np

from rsma_aoi.baselines.common import make_direct_action, top_m_mask


class MaxAoISingleUserPolicy:
    """Serve only the user with largest weighted AoI using the full private budget."""

    name = "max_aoi_single_user"

    def __init__(self, num_users: int) -> None:
        self.num_users = int(num_users)

    def reset(self, seed: int | None = None) -> None:
        return None

    def act(self, obs: np.ndarray, states: list[dict[str, Any]] | None = None):
        if states is None:
            raise ValueError("MaxAoISingleUserPolicy requires structured state")
        actions = []
        for state in states:
            priority = np.asarray(state["weights"]) * np.asarray(state["aoi"])
            active = top_m_mask(priority, 1)
            q = active.astype(np.float64)
            actions.append(make_direct_action(active, np.r_[0.0, q], np.zeros(self.num_users)))
        return actions


class OldestCommonFixedAlphaPolicy:
    """Put the stalest user's update fully in common, with fixed common power.

    All users are active.  The stalest user has no private payload and receives
    no private power; the remaining users split the residual private power.
    """

    name = "oldest_common_fixed_alpha"

    def __init__(self, num_users: int, alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        self.num_users = int(num_users)
        self.alpha = float(alpha)

    def reset(self, seed: int | None = None) -> None:
        return None

    def act(self, obs: np.ndarray, states: list[dict[str, Any]] | None = None):
        if states is None:
            raise ValueError("OldestCommonFixedAlphaPolicy requires structured state")
        actions = []
        for state in states:
            priority = np.asarray(state["weights"]) * np.asarray(state["aoi"])
            oldest = int(np.argmax(priority))
            active = np.ones(self.num_users, dtype=bool)
            q = np.ones(self.num_users, dtype=np.float64)
            q[oldest] = 0.0
            if q.sum() == 0:
                q[oldest] = 1.0
            q /= q.sum()
            phi = np.zeros(self.num_users, dtype=np.float64)
            phi[oldest] = 1.0
            actions.append(make_direct_action(active, np.r_[self.alpha, q], phi))
        return actions

