"""Feasibility of the constraint-preserving action parameterisation.

The policy's transform must produce, for every latent sample:
    * non-negative power fractions summing to 1,
    * exactly zero private power / rate for inactive users,
    * a rate simplex over the active users,
and the environment must therefore report zero constraint violation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rsma_aoi.agents.networks import HybridActorCritic
from rsma_aoi.agents.ppo import latent_to_env_actions
from rsma_aoi.envs.aoi_env import AoIRSMAEnv
from rsma_aoi.envs.rsma_phy import SlotAction, action_violation
from rsma_aoi.utils.config import EnvConfig


@pytest.fixture
def env_cfg() -> EnvConfig:
    return EnvConfig(num_users=4, num_antennas=2, episode_length=20)


def build_model(cfg: EnvConfig, obs_dim: int) -> HybridActorCritic:
    torch.manual_seed(0)
    return HybridActorCritic(
        obs_dim=obs_dim, num_users=cfg.num_users, hidden_sizes=[32, 32],
        scheduling_enabled=cfg.scheduling_enabled,
        power_scale_action=cfg.power_scale_action,
    )


def test_policy_actions_are_feasible_by_construction(env_cfg):
    env = AoIRSMAEnv(env_cfg)
    obs, _ = env.reset(seed=0)
    model = build_model(env_cfg, obs.shape[0])
    gen = torch.Generator().manual_seed(7)

    for _ in range(50):
        obs_t = torch.as_tensor(obs[None, :], dtype=torch.float32)
        dist = model.distribution(obs_t)
        latent = dist.sample(gen)
        action = latent_to_env_actions(dist, latent)[0]

        active = action["schedule"].astype(bool)
        pf = action["power_fractions"]
        rf = action["rate_fractions"]

        assert active.any(), "the all-zero schedule must be repaired"
        assert np.all(pf >= 0.0) and np.all(rf >= 0.0)
        assert pf.sum() == pytest.approx(1.0, abs=1e-6)
        assert rf[active].sum() == pytest.approx(1.0, abs=1e-6)
        np.testing.assert_array_equal(pf[1:][~active], 0.0)
        np.testing.assert_array_equal(rf[~active], 0.0)
        assert 0.0 <= float(action["common_payload_scale"][0]) <= 1.0

        slot = SlotAction(active, pf, rf, float(action["common_payload_scale"][0]))
        assert action_violation(slot) == pytest.approx(0.0)

        obs, _, _, truncated, info = env.step(action)
        assert info["constraint_violation"] == pytest.approx(0.0)
        assert info["total_power"] <= env_cfg.total_power + 1e-9
        if truncated:
            obs, _ = env.reset()
    env.close()


def test_all_zero_schedule_is_repaired_to_the_most_probable_user():
    from rsma_aoi.agents.distributions import BernoulliSchedule

    logits = torch.tensor([[-5.0, -1.0, -3.0], [2.0, -2.0, 1.0]])
    dist = BernoulliSchedule(logits)
    repaired = dist.repair(torch.zeros(2, 3))
    assert repaired[0].tolist() == [False, True, False]  # argmax logit of row 0
    assert repaired[1].tolist() == [True, False, False]
    # a non-empty mask is left untouched
    raw = torch.tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    np.testing.assert_array_equal(dist.repair(raw).numpy(), raw.numpy().astype(bool))


def test_scheduling_disabled_activates_everyone(env_cfg):
    cfg = EnvConfig(**{**env_cfg.to_dict(), "scheduling_enabled": False})
    env = AoIRSMAEnv(cfg)
    obs, _ = env.reset(seed=0)
    model = build_model(cfg, obs.shape[0])
    gen = torch.Generator().manual_seed(1)
    obs_t = torch.as_tensor(obs[None, :], dtype=torch.float32)
    dist = model.distribution(obs_t)
    latent = dist.sample(gen)
    action = latent_to_env_actions(dist, latent)[0]
    np.testing.assert_array_equal(action["schedule"], np.ones(cfg.num_users, dtype=np.int8))
    assert np.all(action["power_fractions"] > 0.0)
    env.close()


def test_power_scale_action_is_emitted_when_enabled(env_cfg):
    cfg = EnvConfig(**{**env_cfg.to_dict(), "power_scale_action": True})
    env = AoIRSMAEnv(cfg)
    obs, _ = env.reset(seed=0)
    assert "power_scale" in env.action_space.spaces
    model = build_model(cfg, obs.shape[0])
    gen = torch.Generator().manual_seed(2)
    obs_t = torch.as_tensor(obs[None, :], dtype=torch.float32)
    dist = model.distribution(obs_t)
    latent = dist.sample(gen)
    action = latent_to_env_actions(dist, latent)[0]
    assert 0.0 < float(action["power_scale"][0]) < 1.0
    _, _, _, _, info = env.step(action)
    assert info["total_power"] <= cfg.total_power + 1e-9
    env.close()


def test_environment_reports_violations_for_infeasible_actions(env_cfg):
    env = AoIRSMAEnv(env_cfg)
    env.reset(seed=0)
    k = env_cfg.num_users
    bad = {
        "schedule": np.ones(k, dtype=np.int8),
        "power_fractions": np.full(k + 1, 0.9),        # sums to 3.6 > 1
        "rate_fractions": np.full(k, 0.1),             # sums to 0.4 != 1
        "common_payload_scale": np.array([0.5]),
    }
    _, _, _, _, info = env.step(bad)
    assert info["constraint_violation"] > 0.0
    # the transmission is still physically valid after sanitisation
    assert info["total_power"] <= env_cfg.total_power + 1e-9
    assert info["reward_violation"] < 0.0
    env.close()


def test_inactive_user_power_is_zeroed_by_the_environment(env_cfg):
    env = AoIRSMAEnv(env_cfg)
    env.reset(seed=0)
    k = env_cfg.num_users
    schedule = np.zeros(k, dtype=np.int8)
    schedule[0] = 1
    leaky = {
        "schedule": schedule,
        "power_fractions": np.full(k + 1, 1.0 / (k + 1)),  # leaks to inactive users
        "rate_fractions": np.full(k, 1.0 / k),
        "common_payload_scale": np.array([0.5]),
    }
    _, _, _, _, info = env.step(leaky)
    assert info["constraint_violation"] > 0.0
    np.testing.assert_allclose(info["private_power"][1:], 0.0)
    assert info["num_scheduled"] == 1
    env.close()


def test_random_in_space_actions_never_break_the_environment(env_cfg):
    env = AoIRSMAEnv(env_cfg)
    env.reset(seed=0)
    for _ in range(100):
        _, reward, _, truncated, info = env.step(env.action_space.sample())
        assert np.isfinite(reward)
        assert info["total_power"] <= env_cfg.total_power + 1e-9
        assert np.all(np.isfinite(info["success_probabilities"]))
        if truncated:
            env.reset()
    env.close()
