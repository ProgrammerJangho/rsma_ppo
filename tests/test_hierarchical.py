"""Hierarchical action space: PPO chooses the schedule, an optimiser allocates."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rsma_aoi.agents.ppo import PPOTrainer
from rsma_aoi.envs.aoi_env import AoIRSMAEnv
from rsma_aoi.utils.config import ConfigError, EnvConfig, PPOConfig

BASE = dict(num_users=4, num_antennas=2, snr_db=10.0, csit_error_std=0.3,
            blocklength=64, payload_bits=96.0, episode_length=12, aoi_max=32)


def action_for(env: AoIRSMAEnv, schedule: np.ndarray) -> dict:
    action = env.action_space.sample()
    action["schedule"] = schedule.astype(np.int8)
    return action


@pytest.mark.parametrize("mode", ["equal", "equal_rsma", "grid", "cem_fast", "cached"])
def test_allocator_modes_produce_feasible_transmissions(mode):
    env = AoIRSMAEnv(EnvConfig(**BASE, allocator_mode=mode))
    env.reset(seed=0)
    assert env.is_hierarchical
    schedule = np.array([1, 1, 0, 0])
    for _ in range(env.cfg.episode_length):
        _, reward, _, truncated, info = env.step(action_for(env, schedule))
        assert np.isfinite(reward)
        assert info["constraint_violation"] == pytest.approx(0.0, abs=1e-5)
        assert info["total_power"] <= env.cfg.total_power + 1e-9
        np.testing.assert_allclose(info["private_power"][2:], 0.0, atol=1e-12)
        if truncated:
            env.reset()
    assert info["allocator_calls"] > 0
    env.close()


def test_flat_mode_is_unchanged():
    """allocator_mode='none' must reproduce the original behaviour exactly."""
    schedule = np.array([1, 1, 0, 0])
    traces = []
    for mode in ("none", "none"):
        env = AoIRSMAEnv(EnvConfig(**BASE, allocator_mode=mode))
        env.reset(seed=3)
        env.action_space.seed(1)
        trace = [env.step(action_for(env, schedule))[4]["weighted_aoi"] for _ in range(10)]
        traces.append(trace)
        assert not env.is_hierarchical
        env.close()
    np.testing.assert_allclose(traces[0], traces[1])


def test_allocator_ignores_the_agents_continuous_action():
    """Two very different continuous actions must give the same transmission."""
    env = AoIRSMAEnv(EnvConfig(**BASE, allocator_mode="cem_fast"))
    schedule = np.array([1, 1, 0, 0], dtype=np.int8)
    results = []
    for fill in (0.05, 0.9):
        env.reset(seed=7)
        action = {
            "schedule": schedule,
            "power_fractions": np.full(5, fill),
            "rate_fractions": np.full(4, fill),
            "common_payload_scale": np.array([fill]),
        }
        _, _, _, _, info = env.step(action)
        results.append((info["common_power"], info["private_power"].copy(),
                        info["common_payload"]))
    assert results[0][0] == pytest.approx(results[1][0])
    np.testing.assert_allclose(results[0][1], results[1][1])
    assert results[0][2] == pytest.approx(results[1][2])
    env.close()


def test_common_gate_can_disable_the_common_stream():
    env = AoIRSMAEnv(EnvConfig(**BASE, allocator_mode="cem_fast",
                               allocator_common_gate=True))
    schedule = np.array([1, 1, 0, 0], dtype=np.int8)
    outputs = {}
    for label, rho in (("off", 0.1), ("on", 0.9)):
        env.reset(seed=5)
        action = {
            "schedule": schedule, "power_fractions": np.full(5, 0.2),
            "rate_fractions": np.full(4, 0.25),
            "common_payload_scale": np.array([rho]),
        }
        _, _, _, _, info = env.step(action)
        outputs[label] = info
    assert outputs["off"]["common_power"] == pytest.approx(0.0)
    assert outputs["off"]["common_payload"] == pytest.approx(0.0)
    assert outputs["on"]["common_power"] >= 0.0
    env.close()


def test_allocator_calls_and_evaluations_are_accounted():
    """Compute cost must be reported, not just environment interactions."""
    steps = 6
    counts = {}
    for mode in ("equal", "cem_fast"):
        env = AoIRSMAEnv(EnvConfig(**BASE, allocator_mode=mode))
        env.reset(seed=0)
        for _ in range(steps):
            _, _, _, _, info = env.step(action_for(env, np.array([1, 1, 0, 0])))
        counts[mode] = (info["allocator_calls"], info["allocator_evaluations"])
        env.close()
    assert counts["equal"][0] == steps
    assert counts["cem_fast"][0] == steps
    assert counts["cem_fast"][1] > counts["equal"][1] * 10


def test_cached_mode_reduces_optimizer_evaluations():
    common = dict(BASE, episode_length=40)
    evaluations = {}
    for mode in ("cem_fast", "cached"):
        env = AoIRSMAEnv(EnvConfig(**common, allocator_mode=mode))
        env.reset(seed=0)
        for _ in range(40):
            _, _, _, _, info = env.step(action_for(env, np.array([1, 1, 0, 0])))
        evaluations[mode] = info["allocator_evaluations"]
        env.close()
    assert evaluations["cached"] < evaluations["cem_fast"]


def test_hierarchical_env_is_seed_reproducible():
    schedule = np.array([1, 0, 1, 0])

    def rollout(seed: int):
        env = AoIRSMAEnv(EnvConfig(**BASE, allocator_mode="cem_fast"))
        env.reset(seed=seed)
        out = [env.step(action_for(env, schedule))[4]["weighted_aoi"] for _ in range(8)]
        env.close()
        return out

    np.testing.assert_allclose(rollout(11), rollout(11))
    assert not np.allclose(rollout(11), rollout(12))


def test_ppo_trains_on_the_hierarchical_env(tmp_path):
    env_cfg = EnvConfig(**BASE, allocator_mode="cem_fast")
    ppo_cfg = PPOConfig(total_timesteps=128, num_envs=2, rollout_steps=32,
                        minibatch_size=32, update_epochs=1, hidden_sizes=[32, 32],
                        scheduling_head="topm_categorical", checkpoint_interval=0)
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / "hier", seed=0)
    trainer.collect_rollout()

    # The PPO ratio must still be exact before any gradient step: the allocator
    # is part of the environment, so it does not enter the log-probability.
    trainer.buffer.compute_returns(ppo_cfg.gamma, ppo_cfg.gae_lambda)
    max_error = 0.0
    for batch in trainer.buffer.iter_minibatches(32, np.random.default_rng(0)):
        with torch.no_grad():
            recomputed = trainer.model.distribution(batch.obs).log_prob(batch.latent)
        max_error = max(max_error, float((torch.exp(recomputed - batch.log_probs) - 1).abs().max()))
    assert max_error < 1e-3

    metrics = trainer.update()
    assert np.isfinite(metrics["train/policy_loss"])
    trainer.close()


def test_noma_with_an_allocator_is_rejected():
    with pytest.raises(ConfigError, match="NOMA"):
        EnvConfig(**BASE, allocator_mode="cem_fast", multiple_access="noma")


def test_unknown_allocator_mode_is_rejected():
    with pytest.raises(ConfigError):
        EnvConfig(**BASE, allocator_mode="magic")
