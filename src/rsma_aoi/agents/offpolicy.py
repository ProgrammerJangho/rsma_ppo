"""Factory for the off-policy baselines.

STATUS
------
* ``td3``: **implemented** in :mod:`rsma_aoi.agents.td3` - continuous resource
  allocation only (``scheduling_enabled: false``), sharing the PPO policy's
  feasible-action transform.
* ``sac``: not implemented; the same interface applies when it is added.

Design constraints any off-policy agent here must respect:

1. **Action parameterisation.**  The raw actor output must be mapped through
   :func:`rsma_aoi.agents.distributions.latent_to_feasible_action` (masked
   softmax for the two simplexes, sigmoid for the scalars) rather than clipped,
   so that the constraints hold exactly and the comparison against PPO is not
   confounded by different action handling.  A discrete scheduling head needs a
   separate treatment (e.g. Gumbel-softmax or a discrete critic) and is out of
   scope for the current TD3 baseline.
2. **Fair comparison.**  PPO is on-policy; matching only the number of episodes
   would be unfair.  Compare at equal *environment interaction* budgets and
   report wall-clock and inference latency alongside (see the README).
"""

from __future__ import annotations

from typing import Any, Protocol

from rsma_aoi.utils.config import EnvConfig

SUPPORTED_ALGORITHMS = ("td3", "sac")
IMPLEMENTED_ALGORITHMS = ("td3",)


class OffPolicyTrainer(Protocol):
    """Minimal contract an off-policy trainer must satisfy."""

    def train(self) -> dict[str, float]:
        """Run the full interaction budget and return the final metrics."""
        ...

    def save_checkpoint(self, path: str) -> Any:
        """Persist weights + normaliser state (same layout as PPO)."""
        ...

    def make_policy(self, deterministic: bool = True) -> Any:
        """Return an object exposing ``act(obs, states) -> list[action dict]``."""
        ...


def build_offpolicy_agent(
    algorithm: str, env_cfg: EnvConfig, agent_cfg: Any = None, **kwargs: Any
) -> OffPolicyTrainer:
    """Build an off-policy trainer.

    Args:
        algorithm: one of :data:`SUPPORTED_ALGORITHMS`.
        env_cfg: environment configuration (must have ``scheduling_enabled: false``
            for the TD3 baseline).
        agent_cfg: algorithm config; defaults to :class:`TD3Config` for ``td3``.
        **kwargs: forwarded to the trainer (``run_dir``, ``seed``, ...).

    Raises:
        ValueError: unknown algorithm.
        NotImplementedError: recognised but not implemented yet (``sac``).
    """
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"algorithm must be one of {SUPPORTED_ALGORITHMS}, got {algorithm!r}")
    if algorithm == "td3":
        from rsma_aoi.agents.td3 import TD3Trainer
        from rsma_aoi.utils.config import TD3Config

        return TD3Trainer(env_cfg, agent_cfg or TD3Config(), **kwargs)
    raise NotImplementedError(
        f"the {algorithm.upper()} baseline is not implemented yet; "
        "see the module docstring for the required action parameterisation and "
        "the equal-interaction-budget comparison protocol"
    )
