"""AoI recursion: reset on success, increment on failure, cap at Delta_max."""

from __future__ import annotations

import numpy as np
import pytest

from rsma_aoi.baselines.common import make_action
from rsma_aoi.envs.aoi_env import AoIRSMAEnv
from rsma_aoi.utils.config import EnvConfig


def make_env(**kwargs) -> AoIRSMAEnv:
    cfg = EnvConfig(
        **{
            "num_users": 2, "num_antennas": 2, "link_model": "fixed",
            "aoi_max": 5, "episode_length": 50, "blocklength": 100,
            "payload_bits": 100.0, **kwargs,
        }
    )
    return AoIRSMAEnv(cfg)


def all_active_action(num_users: int = 2, common_fraction: float = 0.2):
    pf = np.zeros(num_users + 1)
    pf[0] = common_fraction
    pf[1:] = (1.0 - common_fraction) / num_users
    rf = np.full(num_users, 1.0 / num_users)
    return make_action(np.ones(num_users, dtype=bool), pf, rf, 0.5)


def test_success_resets_the_age_to_one():
    env = make_env(fixed_success_prob=1.0)
    env.reset(seed=0)
    for _ in range(5):
        _, _, _, _, info = env.step(all_active_action())
    np.testing.assert_allclose(info["aoi"], [1.0, 1.0])
    np.testing.assert_allclose(info["ack"], [1.0, 1.0])


def test_failure_increments_the_age_by_one():
    env = make_env(fixed_success_prob=0.0)
    env.reset(seed=0)
    ages = []
    for _ in range(3):
        _, _, _, _, info = env.step(all_active_action())
        ages.append(info["aoi"].copy())
    np.testing.assert_allclose(ages[0], [2.0, 2.0])
    np.testing.assert_allclose(ages[1], [3.0, 3.0])
    np.testing.assert_allclose(ages[2], [4.0, 4.0])


def test_age_is_capped_at_aoi_max():
    env = make_env(fixed_success_prob=0.0, aoi_max=5)
    env.reset(seed=0)
    for _ in range(30):
        _, _, _, _, info = env.step(all_active_action())
    np.testing.assert_allclose(info["aoi"], [5.0, 5.0])


def test_unscheduled_users_age():
    env = make_env(fixed_success_prob=1.0, aoi_max=64)
    env.reset(seed=0)
    pf = np.array([0.2, 0.8, 0.0])
    action = make_action(np.array([True, False]), pf, np.array([1.0, 0.0]), 0.5)
    for step in range(1, 5):
        _, _, _, _, info = env.step(action)
        assert info["aoi"][0] == pytest.approx(1.0)
        assert info["aoi"][1] == pytest.approx(1.0 + step)


def test_age_stays_within_bounds_under_random_actions():
    env = make_env(link_model="fbl", aoi_max=10)
    env.reset(seed=3)
    for _ in range(200):
        _, _, _, _, info = env.step(env.action_space.sample())
        assert np.all(info["aoi"] >= 1.0)
        assert np.all(info["aoi"] <= 10.0)


def test_expected_mode_gives_the_expected_transition():
    """With p = 0.8 and Delta = 1: E[Delta'] = 0.8 * 1 + 0.2 * 2 = 1.2."""
    env = make_env(fixed_success_prob=0.8, ack_mode="expected", aoi_max=64)
    env.reset(seed=0)
    _, _, _, _, info = env.step(all_active_action())
    np.testing.assert_allclose(info["aoi"], [1.2, 1.2], rtol=1e-9)
    _, _, _, _, info = env.step(all_active_action())
    np.testing.assert_allclose(info["aoi"], [0.8 + 0.2 * 2.2] * 2, rtol=1e-9)


def test_reward_equals_the_negative_weighted_age():
    env = make_env(fixed_success_prob=0.0, aoi_max=64, aoi_weights=[1.0, 2.0])
    env.reset(seed=0)
    _, reward, _, _, info = env.step(all_active_action())
    expected = -(1.0 * info["aoi"][0] + 2.0 * info["aoi"][1]) * env.reward_scale
    assert reward == pytest.approx(expected)
    assert info["weighted_aoi"] == pytest.approx(1.0 * info["aoi"][0] + 2.0 * info["aoi"][1])


def test_expected_aoi_reward_is_conditional_mean_without_changing_transition():
    env = make_env(
        fixed_success_prob=0.8, ack_mode="sampled", reward_mode="expected_aoi",
        aoi_max=64, reward_scale=1.0,
    )
    env.reset(seed=0)
    _, reward, _, _, info = env.step(all_active_action())
    # Starting from Delta=1, each user's conditional next age is 1.2.
    assert reward == pytest.approx(-2.4)
    assert info["reward_aoi"] == pytest.approx(-2.4)
    assert info["reward_expected_aoi"] == 1.0
    # The MDP state itself still follows the sampled ACK transition.
    assert set(np.asarray(info["aoi"]).tolist()).issubset({1.0, 2.0})


def test_tail_and_power_penalties_enter_the_reward():
    env = make_env(fixed_success_prob=0.0, aoi_max=64, lambda_tail=1.0,
                   aoi_tail_threshold=1, lambda_power=2.0)
    env.reset(seed=0)
    _, reward, _, _, info = env.step(all_active_action())
    total = (
        info["reward_aoi"] + info["reward_power"] + info["reward_tail"]
        + info["reward_violation"]
    )
    assert reward == pytest.approx(total)
    assert info["reward_tail"] < 0.0
    assert info["reward_power"] < 0.0


def test_random_aoi_initialisation_stays_in_range():
    env = make_env(aoi_init="random", aoi_max=8)
    for seed in range(20):
        env.reset(seed=seed)
        assert np.all(env.aoi >= 1.0) and np.all(env.aoi <= 8.0)


def test_episode_truncates_at_the_horizon_without_terminating():
    env = make_env(episode_length=7)
    env.reset(seed=0)
    for step in range(7):
        _, _, terminated, truncated, info = env.step(all_active_action())
        assert terminated is False
        assert truncated is (step == 6)
    assert "episode" in info
    assert info["episode"]["l"] == 7
