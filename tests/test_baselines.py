"""Baselines: feasibility, scheduling behaviour and genuine PHY differences."""

from __future__ import annotations

import numpy as np
import pytest

from rsma_aoi.baselines import BASELINE_REGISTRY, build_baseline
from rsma_aoi.baselines.common import default_num_scheduled, equal_split, top_m_mask
from rsma_aoi.envs.aoi_env import AoIRSMAEnv
from rsma_aoi.evaluation.evaluator import apply_overrides, evaluate_policy
from rsma_aoi.utils.config import EnvConfig


@pytest.fixture
def cfg() -> EnvConfig:
    return EnvConfig(num_users=4, num_antennas=2, episode_length=20, payload_bits=192.0)


@pytest.mark.parametrize("name", sorted(BASELINE_REGISTRY))
def test_baselines_emit_feasible_actions(name, cfg):
    policy, overrides = build_baseline(name, cfg, seed=0)
    env_cfg = apply_overrides(cfg, overrides)
    env = AoIRSMAEnv(env_cfg)
    obs, _ = env.reset(seed=0)
    for _ in range(env_cfg.episode_length):
        action = policy.act(obs[None, :], [env.get_state()])[0]
        pf, rf = action["power_fractions"], action["rate_fractions"]
        active = action["schedule"].astype(bool)
        assert active.any()
        assert np.all(pf >= 0.0) and pf.sum() <= 1.0 + 1e-9
        np.testing.assert_allclose(pf[1:][~active], 0.0, atol=1e-12)
        np.testing.assert_allclose(rf[~active], 0.0, atol=1e-12)
        obs, reward, _, truncated, info = env.step(action)
        assert np.isfinite(reward)
        assert info["total_power"] <= env_cfg.total_power + 1e-9
        if truncated:
            obs, _ = env.reset()
    env.close()


def test_max_aoi_first_schedules_the_oldest_users(cfg):
    policy, _ = build_baseline("max_aoi_first", cfg, seed=0)
    state = {
        "aoi": np.array([1.0, 9.0, 3.0, 7.0]),
        "weights": np.ones(4),
        "h_hat": np.ones((4, 2), dtype=complex),
        "payload": np.full(4, 192.0),
        "num_users": 4, "num_antennas": 2, "step": 0,
    }
    action = policy.act(np.zeros((1, 3)), [state])[0]
    # min(K, Nt) = 2 users -> the two oldest are indices 1 and 3
    np.testing.assert_array_equal(action["schedule"], np.array([0, 1, 0, 1], dtype=np.int8))


def test_weights_shift_the_priority(cfg):
    policy, _ = build_baseline("max_aoi_first", cfg, seed=0)
    state = {
        "aoi": np.array([2.0, 9.0, 3.0, 1.0]),
        "weights": np.array([10.0, 1.0, 1.0, 1.0]),
        "h_hat": np.ones((4, 2), dtype=complex),
        "payload": np.full(4, 192.0), "num_users": 4, "num_antennas": 2, "step": 0,
    }
    action = policy.act(np.zeros((1, 3)), [state])[0]
    assert action["schedule"][0] == 1  # w*Delta = 20 dominates


def test_equal_power_schedules_everyone(cfg):
    policy, _ = build_baseline("equal_power", cfg, seed=0)
    action = policy.act(np.zeros((1, 3)), None)[0]
    np.testing.assert_array_equal(action["schedule"], np.ones(4, dtype=np.int8))
    assert action["power_fractions"].sum() == pytest.approx(1.0)


def test_random_policy_is_seed_reproducible(cfg):
    a, _ = build_baseline("random", cfg, seed=5)
    b, _ = build_baseline("random", cfg, seed=5)
    c, _ = build_baseline("random", cfg, seed=6)
    obs = np.zeros((1, 3))
    act_a = a.act(obs, None)[0]
    act_b = b.act(obs, None)[0]
    act_c = c.act(obs, None)[0]
    np.testing.assert_allclose(act_a["power_fractions"], act_b["power_fractions"])
    assert not np.allclose(act_a["power_fractions"], act_c["power_fractions"])


def test_oma_sdma_and_rsma_differ_physically(cfg):
    """Same policy family, same seeds - the PHY must produce different SINRs."""
    results = {}
    for name in ("sdma", "oma", "noma"):
        policy, overrides = build_baseline(name, cfg, seed=0)
        env_cfg = apply_overrides(cfg, overrides)
        env = AoIRSMAEnv(env_cfg)
        obs, _ = env.reset(seed=11)
        action = policy.act(obs[None, :], [env.get_state()])[0]
        _, _, _, _, info = env.step(action)
        results[name] = info
        env.close()

    assert not np.allclose(results["sdma"]["private_sinr"], results["oma"]["private_sinr"])
    assert not np.allclose(results["sdma"]["private_sinr"], results["noma"]["private_sinr"])
    # OMA divides the blocklength, SDMA does not
    assert results["oma"]["common_power"] == 0.0
    assert results["sdma"]["common_power"] == 0.0


def test_sdma_override_removes_the_common_stream(cfg):
    _, overrides = build_baseline("sdma", cfg, seed=0)
    env_cfg = apply_overrides(cfg, overrides)
    assert env_cfg.multiple_access == "sdma"
    env = AoIRSMAEnv(env_cfg)
    env.reset(seed=0)
    k = cfg.num_users
    action = {
        "schedule": np.ones(k, dtype=np.int8),
        "power_fractions": np.full(k + 1, 1.0 / (k + 1)),
        "rate_fractions": np.full(k, 1.0 / k),
        "common_payload_scale": np.array([1.0]),
    }
    _, _, _, _, info = env.step(action)
    assert info["common_power"] == 0.0
    assert info["common_payload"] == 0.0
    np.testing.assert_allclose(info["common_bler"], 0.0)
    env.close()


def test_all_baselines_are_evaluated_on_identical_channels(cfg):
    """Every algorithm must see the same channel realisations for a given seed."""
    traces = {}
    for name in sorted(BASELINE_REGISTRY):
        policy, overrides = build_baseline(name, cfg, seed=0)
        env_cfg = apply_overrides(cfg, overrides)
        env = AoIRSMAEnv(env_cfg)
        obs, _ = env.reset(seed=321)
        trace = []
        for _ in range(10):
            action = policy.act(obs[None, :], [env.get_state()])[0]
            obs, _, _, _, _ = env.step(action)
            trace.append(env.get_state()["h_hat"].copy())
        traces[name] = np.stack(trace)
        env.close()
    reference = traces["random"]
    for name, trace in traces.items():
        np.testing.assert_allclose(trace, reference, err_msg=f"{name} saw different channels")


def test_evaluate_policy_returns_one_row_per_episode(cfg):
    policy, _ = build_baseline("max_aoi_first", cfg, seed=0)
    df = evaluate_policy(policy, cfg, episodes=3, eval_seed=7, algorithm="max_aoi_first")
    assert len(df) == 3
    assert set(df["seed"]) == {7, 8, 9}
    assert (df["algorithm"] == "max_aoi_first").all()
    assert (df["steps"] == cfg.episode_length).all()
    assert np.all(np.isfinite(df["weighted_avg_aoi"]))
    assert (df["violation_rate"] == 0.0).all()
    assert {"aoi_user_0", "aoi_user_3"} <= set(df.columns)


def test_unknown_baseline_raises(cfg):
    with pytest.raises(KeyError):
        build_baseline("does_not_exist", cfg)


def test_unknown_override_raises(cfg):
    with pytest.raises(KeyError):
        apply_overrides(cfg, {"not_a_field": 1})


# --- helpers ---------------------------------------------------------------
def test_top_m_mask_and_equal_split():
    mask = top_m_mask(np.array([3.0, 1.0, 5.0]), 2)
    np.testing.assert_array_equal(mask, [True, False, True])
    assert top_m_mask(np.array([1.0, 2.0]), 99).all()
    pf, rf = equal_split(np.array([True, False, True]), common_fraction=0.4)
    assert pf.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(pf, [0.4, 0.3, 0.0, 0.3])
    np.testing.assert_allclose(rf, [0.5, 0.0, 0.5])


def test_default_num_scheduled():
    assert default_num_scheduled(4, 2) == 2
    assert default_num_scheduled(2, 4) == 2
    assert default_num_scheduled(1, 1) == 1


def test_max_aoi_rsma_actually_uses_the_common_stream(cfg):
    """The RSMA heuristic must differ from the SDMA-equivalent MAC heuristic."""
    policy, overrides = build_baseline("max_aoi_rsma", cfg, seed=0)
    assert overrides == {}
    env = AoIRSMAEnv(cfg)
    obs, _ = env.reset(seed=0)
    action = policy.act(obs[None, :], [env.get_state()])[0]
    assert action["power_fractions"][0] > 0.0, "the common stream must get power"
    assert float(action["common_payload_scale"][0]) > 0.0
    assert action["schedule"].sum() > min(cfg.num_users, cfg.num_antennas)
    _, _, _, _, info = env.step(action)
    assert info["common_power"] > 0.0
    assert info["common_payload"] > 0.0
    assert np.any(info["common_sinr"] > 0.0)
    env.close()


def test_max_aoi_first_coincides_with_sdma_by_construction(cfg):
    """Zero common power/payload reduces 1-layer RSMA to SDMA exactly."""
    rsma_policy, _ = build_baseline("max_aoi_first", cfg, seed=0)
    sdma_policy, overrides = build_baseline("sdma", cfg, seed=0)
    env_rsma = AoIRSMAEnv(cfg)
    env_sdma = AoIRSMAEnv(apply_overrides(cfg, overrides))
    obs_r, _ = env_rsma.reset(seed=5)
    obs_s, _ = env_sdma.reset(seed=5)
    for _ in range(10):
        a_r = rsma_policy.act(obs_r[None, :], [env_rsma.get_state()])[0]
        a_s = sdma_policy.act(obs_s[None, :], [env_sdma.get_state()])[0]
        obs_r, _, _, _, info_r = env_rsma.step(a_r)
        obs_s, _, _, _, info_s = env_sdma.step(a_s)
        np.testing.assert_allclose(info_r["private_sinr"], info_s["private_sinr"], rtol=1e-9)
        np.testing.assert_allclose(info_r["aoi"], info_s["aoi"])
    env_rsma.close()
    env_sdma.close()
