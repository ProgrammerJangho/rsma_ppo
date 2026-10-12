"""Gymnasium compliance, determinism, spaces and NaN-freedom of the environment."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rsma_aoi.envs.aoi_env import AoIRSMAEnv
from rsma_aoi.utils.config import EnvConfig

CONFIGS = {
    "m0_siso_fixed": dict(num_users=2, num_antennas=1, link_model="fixed", episode_length=20),
    "m1_rsma": dict(num_users=2, num_antennas=2, csit_error_std=0.5, alpha=0.8,
                    scheduling_enabled=False, episode_length=20),
    "m2_hybrid": dict(num_users=4, num_antennas=2, csit_error_std=0.3, alpha=0.8,
                      payload_bits=192.0, episode_length=20),
    "gain_features": dict(num_users=3, num_antennas=2, obs_channel_mode="gain_features",
                          episode_length=20),
    "beam_sinr_features": dict(
        num_users=3, num_antennas=2, obs_channel_mode="beam_sinr_features",
        obs_include_snr=True, episode_length=20,
    ),
    "no_channel_obs": dict(num_users=2, num_antennas=2, obs_channel_mode="none",
                           obs_include_prev_action=False, episode_length=20),
    "power_scale": dict(num_users=2, num_antennas=2, power_scale_action=True,
                        episode_length=20),
    "shannon": dict(num_users=2, num_antennas=2, link_model="shannon", episode_length=20),
}


@pytest.fixture(params=sorted(CONFIGS), ids=sorted(CONFIGS))
def env_cfg(request) -> EnvConfig:
    return EnvConfig(**CONFIGS[request.param])


def test_gymnasium_env_checker_passes(env_cfg):
    env = AoIRSMAEnv(env_cfg)
    check_env(env, skip_render_check=True)
    env.close()


def test_observation_matches_the_declared_space(env_cfg):
    env = AoIRSMAEnv(env_cfg)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs), "reset observation outside the space"
    assert isinstance(info, dict)
    for _ in range(30):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs)) and np.isfinite(reward)
        if truncated:
            obs, _ = env.reset()
    env.close()


def test_observation_layout_widths_sum_to_the_space(env_cfg):
    env = AoIRSMAEnv(env_cfg)
    total = sum(width for _, width in env.observation_layout)
    assert total == env.observation_space.shape[0]
    env.close()


def test_episode_snr_sampling_is_observed_seeded_and_updates_phy_power():
    cfg = EnvConfig(
        num_users=2, num_antennas=2, snr_db=10.0,
        snr_db_values=[5.0, 10.0, 15.0], obs_include_snr=True,
    )
    env = AoIRSMAEnv(cfg)
    observed = []
    for seed in range(30):
        obs, info = env.reset(seed=seed)
        observed.append(info["snr_db"])
        assert obs[-1] == pytest.approx(info["snr_db"] / 20.0)
        assert env.total_power == pytest.approx(10.0 ** (info["snr_db"] / 10.0))
        assert env.phy.total_power == pytest.approx(env.total_power)
        state = env.get_state()
        assert state["snr_db"] == info["snr_db"]
        assert state["total_power"] == pytest.approx(env.total_power)
        obs_again, info_again = env.reset(seed=seed)
        assert info_again["snr_db"] == info["snr_db"]
        np.testing.assert_array_equal(obs_again, obs)
    assert set(observed) == {5.0, 10.0, 15.0}
    env.close()


def test_episode_blocklength_sampling_is_observed_seeded_and_updates_phy():
    cfg = EnvConfig(
        num_users=3, num_antennas=2, blocklength=128,
        blocklength_values=[64, 128, 256], obs_include_blocklength=True,
    )
    env = AoIRSMAEnv(cfg)
    observed = []
    for seed in range(30):
        obs, info = env.reset(seed=seed)
        observed.append(info["blocklength"])
        assert obs[-1] == pytest.approx(info["blocklength"] / 128.0)
        assert env.phy.blocklength == info["blocklength"]
        assert env.get_state()["blocklength"] == info["blocklength"]
        obs_again, info_again = env.reset(seed=seed)
        assert info_again["blocklength"] == info["blocklength"]
        np.testing.assert_array_equal(obs_again, obs)
    assert set(observed) == {64, 128, 256}
    env.close()


def test_beam_sinr_features_are_causal_and_finite():
    cfg = EnvConfig(
        num_users=3, num_antennas=2, obs_channel_mode="beam_sinr_features",
        obs_include_prev_action=True, episode_length=4,
    )
    env = AoIRSMAEnv(cfg)
    obs, _ = env.reset(seed=123)
    assert env.observation_space.contains(obs)
    # Altering the hidden true channel alone must not change transmitter-side
    # features; they are functions of h_hat and known previous allocation.
    before = env._build_observation()
    env._h_true = 100.0 * env._h_true
    after = env._build_observation()
    np.testing.assert_allclose(after, before)
    env.close()


def test_same_seed_and_actions_give_the_same_trajectory(env_cfg):
    env = AoIRSMAEnv(env_cfg)
    env.action_space.seed(0)  # the sampled actions must not vary between runs
    actions = [env.action_space.sample() for _ in range(15)]

    def rollout(seed: int):
        obs, _ = env.reset(seed=seed)
        observations, rewards, ages = [obs.copy()], [], []
        channels = [env.get_state()["h_hat"].copy()]
        for action in actions:
            obs, reward, _, _, info = env.step(action)
            observations.append(obs.copy())
            rewards.append(reward)
            ages.append(info["aoi"].copy())
            channels.append(env.get_state()["h_hat"].copy())
        return np.stack(observations), np.asarray(rewards), np.stack(ages), np.stack(channels)

    a = rollout(2024)
    b = rollout(2024)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y)

    # A different seed must drive a different *trajectory*.  The comparison is
    # made on the channel rather than the observation because some observation
    # modes (obs_channel_mode='none') deliberately exclude the channel, and the
    # FBL success probabilities are near-deterministic, so two seeds can produce
    # identical AoI/ACK traces without the environment being non-reproducible.
    c = rollout(999)
    assert not np.allclose(a[3], c[3]), "different seeds produced the same channel"
    env.close()


def test_channel_sequence_is_action_independent(env_cfg):
    """Different policies must see identical channels for the same seed.

    The environment draws a fixed number of random variates per slot, which is
    what makes the baseline/PPO comparison use the same channel realisations.
    """
    env = AoIRSMAEnv(env_cfg)

    def channels(seed: int, sampler):
        env.reset(seed=seed)
        trace = [env.get_state()["h_hat"].copy()]
        for _ in range(12):
            env.step(sampler())
            trace.append(env.get_state()["h_hat"].copy())
        return np.stack(trace)

    rng = np.random.default_rng(0)
    k = env_cfg.num_users

    def greedy():
        action = env.action_space.sample()
        action["schedule"] = np.ones(k, dtype=np.int8)
        return action

    def sparse():
        action = env.action_space.sample()
        mask = np.zeros(k, dtype=np.int8)
        mask[rng.integers(0, k)] = 1
        action["schedule"] = mask
        return action

    np.testing.assert_allclose(channels(77, greedy), channels(77, sparse))
    env.close()


def test_info_contains_the_documented_keys(env_cfg):
    env = AoIRSMAEnv(env_cfg)
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    required = {
        "weighted_aoi", "mean_aoi", "max_aoi", "tail_excess", "total_power",
        "common_power", "private_power", "common_sinr", "private_sinr",
        "common_bler", "private_bler", "success_probabilities", "ack",
        "scheduled_users", "constraint_violation",
    }
    assert required <= set(info)
    for key, value in info.items():
        if isinstance(value, np.ndarray):
            assert np.all(np.isfinite(value)), f"non-finite values in info[{key!r}]"
    assert np.all((info["success_probabilities"] >= 0) & (info["success_probabilities"] <= 1))
    assert np.all((info["common_bler"] >= 0) & (info["common_bler"] <= 1))
    assert np.all((info["private_bler"] >= 0) & (info["private_bler"] <= 1))
    env.close()


def test_termination_and_truncation_are_distinguished(env_cfg):
    env = AoIRSMAEnv(env_cfg)
    env.reset(seed=0)
    for step in range(env_cfg.episode_length):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert terminated is False
        assert truncated == (step == env_cfg.episode_length - 1)
    env.close()


def test_extreme_channel_conditions_stay_finite():
    for snr_db in (-30.0, 0.0, 60.0):
        cfg = EnvConfig(num_users=3, num_antennas=2, snr_db=snr_db, csit_error_std=0.9,
                        payload_bits=512.0, blocklength=8, episode_length=10)
        env = AoIRSMAEnv(cfg)
        env.reset(seed=0)
        for _ in range(10):
            obs, reward, _, _, info = env.step(env.action_space.sample())
            assert np.all(np.isfinite(obs)) and np.isfinite(reward)
            assert np.all(np.isfinite(info["private_sinr"]))
            assert np.all(np.isfinite(info["success_probabilities"]))
        env.close()


def test_malformed_actions_raise_clear_errors():
    env = AoIRSMAEnv(EnvConfig(num_users=2, num_antennas=2))
    env.reset(seed=0)
    with pytest.raises(TypeError):
        env.step(np.zeros(3))
    with pytest.raises(ValueError):
        env.step({"schedule": np.ones(2, dtype=np.int8)})
    env.close()


def test_vector_env_matches_single_env_rollouts():
    from rsma_aoi.utils.vec_env import SyncVecEnv

    cfg = EnvConfig(num_users=2, num_antennas=2, episode_length=5)
    vec = SyncVecEnv([lambda: AoIRSMAEnv(cfg) for _ in range(3)])
    obs, _ = vec.reset([1, 2, 3])
    assert obs.shape == (3, vec.obs_dim)
    single = AoIRSMAEnv(cfg)
    single_obs, _ = single.reset(seed=2)
    np.testing.assert_allclose(obs[1], single_obs)

    action = vec.action_space.sample()
    for _ in range(5):
        obs, rewards, term, trunc, infos = vec.step([action] * 3)
        assert obs.shape == (3, vec.obs_dim) and rewards.shape == (3,)
        assert term.shape == (3,) and trunc.shape == (3,)
    assert np.all(trunc) and not np.any(term)
    assert "final_observation" in infos[0] and "episode" in infos[0]
    assert len(vec.get_states()) == 3
    vec.close()
    single.close()
