"""Integration checks for direct power + common-payload PPO control."""
from pathlib import Path

import numpy as np
import pytest
import torch

from rsma_aoi.agents.ppo import PPOTrainer
from rsma_aoi.envs.aoi_env import AoIRSMAEnv
from rsma_aoi.utils.config import EnvConfig, PPOConfig

ROOT = Path(__file__).resolve().parents[1]


def test_common_rate_example_reaches_phy_without_allocator_override():
    cfg = EnvConfig(num_users=2, num_antennas=2, payload_bits=120, blocklength=128,
                    scheduling_enabled=False, allocator_mode="none")
    env = AoIRSMAEnv(cfg)
    try:
        env.reset(seed=10)
        _, _, _, _, info = env.step({
            "schedule": np.ones(2), "power_fractions": np.array([0.2, 0.5, 0.3]),
            "rate_fractions": np.array([0.75, 0.25]), "common_payload_scale": np.array([0.5]),
        })
        np.testing.assert_allclose(info["private_payload"], [60, 100])
        assert info["common_payload"] == pytest.approx(80)
        np.testing.assert_allclose((120 - info["private_payload"]) / 128, [0.46875, 0.15625])
        np.testing.assert_allclose(info["private_power"], cfg.total_power * np.array([0.5, 0.3]))
        assert info["common_power"] == pytest.approx(0.2 * cfg.total_power)
        assert info["allocator_calls"] == 0
    finally:
        env.close()


def test_both_allocation_heads_learn_while_schedule_head_stays_unused(tmp_path):
    env_cfg = EnvConfig.from_yaml(ROOT / "configs/thesis_allocation.yaml")
    agent_cfg = PPOConfig(total_timesteps=128, num_envs=2, rollout_steps=32,
                          minibatch_size=32, update_epochs=2, hidden_sizes=[32, 32],
                          checkpoint_interval=0, normalize_reward=True)
    trainer = PPOTrainer(env_cfg, agent_cfg, run_dir=tmp_path, seed=12)
    try:
        names = ["power_head.weight", "rate_head.weight", "rho_r_head.weight", "schedule_head.weight"]
        before = {n: p.detach().clone() for n, p in trainer.model.named_parameters() if n in names}
        trainer.collect_rollout()
        assert np.all(trainer.buffer.active_mask)
        metrics = trainer.update()
        assert all(np.isfinite(v) for v in metrics.values())
        after = dict(trainer.model.named_parameters())
        for n in names[:3]:
            assert not torch.equal(before[n], after[n]), f"allocation head did not update: {n}"
        torch.testing.assert_close(before[names[3]], after[names[3]], rtol=0, atol=0)
    finally:
        trainer.close()
