"""Non-learning baseline policies sharing the environment's PHY model.

Each entry of :data:`BASELINE_REGISTRY` couples a policy factory with the
*environment overrides* that realise its multiple-access scheme, so OMA / SDMA /
NOMA differ in physics, not only in name.
"""

from __future__ import annotations

from typing import Any, Callable

from rsma_aoi.baselines.common import EqualAllocationPolicy, Policy, make_action
from rsma_aoi.baselines.cem_allocation import CEMAllocationPolicy
from rsma_aoi.baselines.equal_power import EqualPowerPolicy
from rsma_aoi.baselines.max_aoi_first import MaxAoIFirstPolicy
from rsma_aoi.baselines.noma import NOMAPolicy
from rsma_aoi.baselines.oma import OMAPolicy
from rsma_aoi.baselines.random_policy import RandomPolicy
from rsma_aoi.baselines.sdma import SDMAPolicy
from rsma_aoi.baselines.thesis_rules import MaxAoISingleUserPolicy, OldestCommonFixedAlphaPolicy
from rsma_aoi.utils.config import EnvConfig

#: Fixed common-stream share used by the RSMA heuristic baseline.
DEFAULT_COMMON_FRACTION = 0.25
#: Fixed rho_r used by the RSMA heuristic baseline.
DEFAULT_COMMON_PAYLOAD_SCALE = 0.3

__all__ = [
    "Policy", "make_action", "EqualAllocationPolicy", "RandomPolicy", "EqualPowerPolicy", "MaxAoIFirstPolicy",
    "OMAPolicy", "SDMAPolicy", "NOMAPolicy", "BASELINE_REGISTRY", "build_baseline",
    "DEFAULT_COMMON_FRACTION", "DEFAULT_COMMON_PAYLOAD_SCALE", "CEMAllocationPolicy",
    "MaxAoISingleUserPolicy", "OldestCommonFixedAlphaPolicy",
]


def _random(cfg: EnvConfig, seed: int) -> RandomPolicy:
    return RandomPolicy(cfg.num_users, seed=seed, power_scale_action=cfg.power_scale_action)


def _equal(cfg: EnvConfig, seed: int) -> EqualPowerPolicy:
    return EqualPowerPolicy(cfg.num_users, power_scale_action=cfg.power_scale_action)


def _max_aoi(cfg: EnvConfig, seed: int) -> MaxAoIFirstPolicy:
    """Pure MAC heuristic: no common stream, so it isolates the scheduling gain.

    In an RSMA environment this coincides *exactly* with :func:`_sdma` (zero
    common power and zero common payload reduce 1-layer RSMA to SDMA), which is
    the intended meaning of the no-common-stream ablation.
    """
    return MaxAoIFirstPolicy(
        cfg.num_users, cfg.num_antennas, power_scale_action=cfg.power_scale_action
    )


def _max_aoi_rsma(cfg: EnvConfig, seed: int) -> MaxAoIFirstPolicy:
    """Max-AoI-first scheduling that actually uses the common stream.

    Serves ``min(K, Nt) + 1`` users (one more than the spatial multiplexing
    limit) and puts a fixed share of the power/payload on the common stream, so
    it is a genuine RSMA heuristic rather than an SDMA policy in disguise.
    """
    return MaxAoIFirstPolicy(
        cfg.num_users,
        cfg.num_antennas,
        num_scheduled=min(cfg.num_users, min(cfg.num_users, cfg.num_antennas) + 1),
        common_fraction=DEFAULT_COMMON_FRACTION,
        common_payload_scale=DEFAULT_COMMON_PAYLOAD_SCALE,
        power_scale_action=cfg.power_scale_action,
    )


def _oma(cfg: EnvConfig, seed: int) -> OMAPolicy:
    return OMAPolicy(cfg.num_users, cfg.num_antennas, power_scale_action=cfg.power_scale_action)


def _sdma(cfg: EnvConfig, seed: int) -> SDMAPolicy:
    return SDMAPolicy(cfg.num_users, cfg.num_antennas, power_scale_action=cfg.power_scale_action)


def _noma(cfg: EnvConfig, seed: int) -> NOMAPolicy:
    return NOMAPolicy(cfg.num_users, cfg.num_antennas, power_scale_action=cfg.power_scale_action)


#: name -> (factory, environment overrides)
BASELINE_REGISTRY: dict[str, tuple[Callable[[EnvConfig, int], Any], dict[str, Any]]] = {
    "random": (_random, {}),
    "equal_power": (_equal, {}),
    "max_aoi_first": (_max_aoi, {}),
    "max_aoi_rsma": (_max_aoi_rsma, {}),
    "oma": (_oma, {"multiple_access": "oma"}),
    "sdma": (_sdma, {"multiple_access": "sdma"}),
    "noma": (_noma, {"multiple_access": "noma"}),
}


def build_baseline(name: str, cfg: EnvConfig, seed: int = 0) -> tuple[Any, dict[str, Any]]:
    """Instantiate a baseline and return ``(policy, env_overrides)``."""
    if name not in BASELINE_REGISTRY:
        raise KeyError(f"unknown baseline {name!r}; available: {sorted(BASELINE_REGISTRY)}")
    factory, overrides = BASELINE_REGISTRY[name]
    return factory(cfg, seed), dict(overrides)
