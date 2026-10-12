"""Sweep/ablation runner helpers: variant parsing and the fairness guard."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_sweep import (  # noqa: E402
    PROTECTED_ENV_FIELDS, check_env_overrides, split_variant,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_split_variant_forms():
    assert split_variant(None) == ({}, {})
    assert split_variant({}) == ({}, {})
    assert split_variant({"entropy_coef": 0.02}) == ({"entropy_coef": 0.02}, {})
    assert split_variant({"env": {"link_model": "shannon"}}) == ({}, {"link_model": "shannon"})
    assert split_variant({"ppo": {"gamma": 0.95}, "env": {"private_beam": "mrt"}}) == (
        {"gamma": 0.95}, {"private_beam": "mrt"}
    )


def test_split_variant_rejects_mixed_forms():
    with pytest.raises(ValueError):
        split_variant({"ppo": {"gamma": 0.95}, "entropy_coef": 0.01})


def test_fairness_guard_blocks_budget_changing_overrides():
    check_env_overrides({"common_beam": "mrt_sum", "link_model": "shannon"}, False)
    for field in ("blocklength", "payload_bits", "snr_db", "num_users", "episode_length"):
        with pytest.raises(SystemExit, match="comparison budget"):
            check_env_overrides({field: 1}, False)
        check_env_overrides({field: 1}, True)  # explicit opt-out is allowed


def test_protected_fields_cover_the_resource_budget():
    for field in ("blocklength", "payload_bits", "snr_db", "p_max", "noise_power",
                  "episode_length", "num_users", "num_antennas"):
        assert field in PROTECTED_ENV_FIELDS


@pytest.mark.parametrize("name", ["sweep_m2.yaml", "ablations_m2.yaml"])
def test_shipped_sweep_files_are_valid(name):
    """Every shipped variant must parse and pass the fairness guard."""
    from rsma_aoi.utils.config import (
        EnvConfig, PPOConfig, apply_overrides_from_strings, load_run_configs,
    )

    spec = yaml.safe_load((CONFIG_DIR / name).read_text())["sweep"]
    env_cfg, ppo_cfg = load_run_configs(spec["base_env"], spec["base_agent"])
    assert spec["seeds"] and spec["total_timesteps"] > 0

    for variant, raw in spec["variants"].items():
        ppo_over, env_over = split_variant(raw)
        check_env_overrides(env_over, False)
        # the overrides must actually apply and revalidate
        if ppo_over:
            new = apply_overrides_from_strings(
                ppo_cfg, [f"{k}={v}" for k, v in ppo_over.items()]
            )
            assert isinstance(new, PPOConfig)
        if env_over:
            new_env = apply_overrides_from_strings(
                env_cfg, [f"{k}={v}" for k, v in env_over.items()]
            )
            assert isinstance(new_env, EnvConfig)
            for field in PROTECTED_ENV_FIELDS:
                assert getattr(new_env, field) == getattr(env_cfg, field), (
                    f"ablation {variant!r} changed protected field {field!r}"
                )
