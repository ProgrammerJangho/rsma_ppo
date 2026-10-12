"""Tests for the optional FBL-decodability-normalised common payload action."""

from __future__ import annotations

import numpy as np
import pytest

from rsma_aoi.envs.aoi_env import AoIRSMAEnv
from rsma_aoi.utils.config import ConfigError, EnvConfig


def _action(rho: float) -> dict[str, np.ndarray]:
    return {
        "schedule": np.ones(2, dtype=np.int8),
        "power_fractions": np.array([0.5, 0.25, 0.25]),
        "rate_fractions": np.array([0.5, 0.5]),
        "common_payload_scale": np.array([rho]),
    }


def test_direct_mapping_preserves_the_policy_scalar():
    env = AoIRSMAEnv(EnvConfig(
        num_users=2, num_antennas=2, scheduling_enabled=False,
        common_payload_mapping="direct",
    ))
    env.reset(seed=7)
    parsed, violation = env._parse_action(_action(0.73))
    assert violation == pytest.approx(0.0)
    assert parsed.common_payload_scale == pytest.approx(0.73)


def test_fbl_normalized_mapping_meets_nominal_worst_user_target():
    target = 0.1
    env = AoIRSMAEnv(EnvConfig(
        num_users=2, num_antennas=2, scheduling_enabled=False,
        common_payload_mapping="fbl_normalized", common_target_bler=target,
        snr_db=10.0, blocklength=128, payload_bits=256.0,
    ))
    env.reset(seed=11)
    parsed, violation = env._parse_action(_action(1.0))
    assert violation == pytest.approx(0.0)
    assert 0.0 <= parsed.common_payload_scale <= 1.0

    beams = env.phy.rsma_beams(env._h_hat, parsed.active, use_common=True)
    out = env.phy.evaluate_allocations(
        env._h_hat,
        parsed.active,
        parsed.power_fractions[None, :],
        parsed.rate_fractions[None, :],
        np.asarray([parsed.common_payload_scale]),
        power_scale=np.asarray([parsed.power_scale]),
        use_common=True,
        beams=beams,
    )
    assert float(np.max(out["common_bler"][0, parsed.active])) <= target + 1e-9


def test_fbl_normalized_mapping_scales_payload_monotonically():
    env = AoIRSMAEnv(EnvConfig(
        num_users=2, num_antennas=2, scheduling_enabled=False,
        common_payload_mapping="fbl_normalized", common_target_bler=0.1,
    ))
    env.reset(seed=19)
    low, _ = env._parse_action(_action(0.25))
    high, _ = env._parse_action(_action(0.75))
    assert low.common_payload_scale <= high.common_payload_scale
    assert high.common_payload_scale == pytest.approx(3.0 * low.common_payload_scale)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"common_payload_mapping": "fbl_normalized", "link_model": "shannon"},
        {"common_payload_mapping": "fbl_normalized", "rate_split_mode": "per_user_fraction"},
        {"common_payload_mapping": "fbl_normalized", "multiple_access": "sdma"},
        {"common_target_bler": 0.0},
        {"common_target_bler": 1.0},
    ],
)
def test_invalid_fbl_normalized_configuration_is_rejected(kwargs):
    with pytest.raises(ConfigError):
        EnvConfig(**kwargs)

