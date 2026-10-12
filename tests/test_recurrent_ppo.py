"""GRU-PPO sequence handling, warm-start compatibility, and checkpoint smoke tests."""

from __future__ import annotations

import numpy as np
import torch

from rsma_aoi.agents.networks import HybridActorCritic, RecurrentHybridActorCritic
from rsma_aoi.agents.ppo import PPOPolicy, PPOTrainer
from rsma_aoi.evaluation.evaluator import evaluate_policy
from rsma_aoi.utils.config import EnvConfig, PPOConfig


def _model(cls):
    kwargs = dict(
        obs_dim=11, num_users=3, hidden_sizes=[32, 32],
        scheduling_enabled=False, rate_split_mode="per_user_fraction",
    )
    if cls is RecurrentHybridActorCritic:
        kwargs["recurrent_hidden_size"] = 32
    return cls(**kwargs)


def test_zero_residual_recurrent_model_preserves_feedforward_policy():
    torch.manual_seed(4)
    feedforward = _model(HybridActorCritic)
    recurrent = _model(RecurrentHybridActorCritic)
    recurrent.load_state_dict(feedforward.state_dict(), strict=False)
    obs = torch.randn(7, 11)
    hidden = recurrent.initial_state(7)
    with torch.no_grad():
        ff = feedforward.distribution(obs).mode()
        gru, _, actor_hidden, _ = recurrent.recurrent_step(obs, *hidden)
        rec = gru.mode()
    torch.testing.assert_close(ff.power_latent, rec.power_latent)
    torch.testing.assert_close(ff.rate_latent, rec.rate_latent)
    assert torch.count_nonzero(actor_hidden).item() > 0


def test_recurrent_training_and_checkpoint_are_finite_and_repeatable(tmp_path):
    env = EnvConfig(
        num_users=3, num_antennas=2, episode_length=16, aoi_max=16,
        blocklength=64, payload_bits=96.0,
    )
    ppo = PPOConfig(
        total_timesteps=128, num_envs=2, rollout_steps=32, minibatch_size=32,
        update_epochs=1, hidden_sizes=[32, 32], recurrent_enabled=True,
        recurrent_hidden_size=32, recurrent_seq_len=8, checkpoint_interval=0,
        device="cpu",
    )
    trainer = PPOTrainer(env, ppo, tmp_path / "run", seed=7)
    metrics = trainer.train()
    assert all(np.isfinite(value) for key, value in metrics.items() if key.startswith("train/"))
    checkpoint = trainer.save_checkpoint(tmp_path / "run" / "checkpoint.pt")
    policy, loaded_env, loaded_ppo = PPOPolicy.from_checkpoint(checkpoint)
    assert loaded_ppo.recurrent_enabled
    first = evaluate_policy(policy, loaded_env, episodes=2, eval_seed=81)
    second = evaluate_policy(policy, loaded_env, episodes=2, eval_seed=81)
    np.testing.assert_allclose(first["weighted_avg_aoi"], second["weighted_avg_aoi"])
    trainer.close()


def test_episode_start_resets_hidden_state():
    model = _model(RecurrentHybridActorCritic)
    obs = torch.randn(2, 11)
    actor = torch.randn(2, 32)
    critic = torch.randn(2, 32)
    starts = torch.tensor([True, False])
    with torch.no_grad():
        _, values, next_actor, next_critic = model.recurrent_step(
            obs, actor, critic, episode_start=starts
        )
        _, reset_values, reset_actor, reset_critic = model.recurrent_step(
            obs[:1], torch.zeros_like(actor[:1]), torch.zeros_like(critic[:1])
        )
    torch.testing.assert_close(values[:1], reset_values)
    torch.testing.assert_close(next_actor[:1], reset_actor)
    torch.testing.assert_close(next_critic[:1], reset_critic)
