"""Random scheduling with a random *feasible* resource allocation."""

from __future__ import annotations

from typing import Any

import numpy as np

from rsma_aoi.baselines.common import make_action, repair_schedule


class RandomPolicy:
    """Uniformly random schedule + Dirichlet(1) power/rate splits.

    The allocation is drawn directly on the simplex, so it is feasible by
    construction (the point of the comparison is the *decision quality*, not
    constraint handling).
    """

    name = "random"

    def __init__(
        self,
        num_users: int,
        seed: int = 0,
        schedule_prob: float = 0.5,
        power_scale_action: bool = False,
    ) -> None:
        self.num_users = num_users
        self.rng = np.random.default_rng(seed)
        self.schedule_prob = float(schedule_prob)
        self.power_scale_action = power_scale_action

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(
        self, obs: np.ndarray, states: list[dict[str, Any]] | None = None
    ) -> list[dict[str, np.ndarray]]:
        n = obs.shape[0]
        actions = []
        for _ in range(n):
            k = self.num_users
            active = repair_schedule(
                self.rng.random(k) < self.schedule_prob, self.rng.random(k)
            )
            pf = np.zeros(k + 1)
            weights = self.rng.dirichlet(np.ones(int(active.sum()) + 1))
            pf[0] = weights[0]
            pf[1:][active] = weights[1:]
            rf = np.zeros(k)
            rf[active] = self.rng.dirichlet(np.ones(int(active.sum())))
            actions.append(
                make_action(
                    active, pf, rf, float(self.rng.random()),
                    float(self.rng.random()) if self.power_scale_action else None,
                )
            )
        return actions
