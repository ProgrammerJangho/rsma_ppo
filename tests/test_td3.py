"""TD3 baseline: replay buffer, algorithm components, feasibility, checkpoints."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rsma_aoi.agents.replay_buffer import ReplayBuffer
from rsma_aoi.agents.td3 import TD3Policy, TD3Trainer, latent_to_env_actions, split_latent
from rsma_aoi.evaluation.evaluator import evaluate_policy
from rsma_aoi.utils.config import EnvConfig, TD3Config


def tiny_configs(**env_kwargs) -> tuple[EnvConfig, TD3Config]:
    env_cfg = EnvConfig(
        **{
            "num_users": 3, "num_antennas": 2, "episode_length": 16, "aoi_max": 16,
            "blocklength": 64, "payload_bits": 96.0, "scheduling_enabled": False,
            **env_kwargs,
        }
    )
    td3_cfg = TD3Config(
        total_timesteps=256, num_envs=2, buffer_size=2048, batch_size=32,
        learning_starts=32, hidden_sizes=[32, 32], log_interval=0,
        checkpoint_interval=0, device="cpu",
    )
    return env_cfg, td3_cfg


# --- replay buffer ---------------------------------------------------------
def test_replay_buffer_stores_and_samples():
    buffer = ReplayBuffer(capacity=100, obs_dim=4, action_dim=3)
    rng = np.random.default_rng(0)
    assert len(buffer) == 0
    with pytest.raises(ValueError):
        buffer.sample(4, rng)

    buffer.add_batch(
        obs=np.ones((5, 4), dtype=np.float32),
        action=np.zeros((5, 3), dtype=np.float32),
        reward=np.arange(5, dtype=np.float32),
        next_obs=2 * np.ones((5, 4), dtype=np.float32),
        terminated=np.array([0, 0, 1, 0, 0], dtype=bool),
    )
    assert len(buffer) == 5
    batch = buffer.sample(8, rng)
    assert batch.obs.shape == (8, 4) and batch.action.shape == (8, 3)
    assert batch.reward.shape == (8,) and batch.not_done.shape == (8,)
    assert set(np.unique(buffer.not_done[:5])) == {0.0, 1.0}


def test_replay_buffer_wraps_around_and_keeps_capacity():
    buffer = ReplayBuffer(capacity=10, obs_dim=2, action_dim=2)
    for step in range(4):
        buffer.add_batch(
            obs=np.full((4, 2), step, dtype=np.float32),
            action=np.zeros((4, 2), dtype=np.float32),
            reward=np.zeros(4, dtype=np.float32),
            next_obs=np.zeros((4, 2), dtype=np.float32),
            terminated=np.zeros(4, dtype=bool),
        )
    assert len(buffer) == 10
    assert buffer.obs.shape[0] == 10
    with pytest.raises(ValueError):
        buffer.add_batch(
            np.zeros((11, 2), np.float32), np.zeros((11, 2), np.float32),
            np.zeros(11, np.float32), np.zeros((11, 2), np.float32), np.zeros(11, bool),
        )
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0, obs_dim=2, action_dim=2)


def test_replay_buffer_truncation_is_bootstrapped():
    """Time-limit truncation must be stored as not-done (terminated=False)."""
    buffer = ReplayBuffer(capacity=4, obs_dim=2, action_dim=2)
    buffer.add_batch(
        np.zeros((2, 2), np.float32), np.zeros((2, 2), np.float32),
        np.zeros(2, np.float32), np.zeros((2, 2), np.float32),
        terminated=np.array([False, True]),
    )
    np.testing.assert_allclose(buffer.not_done[:2], [1.0, 0.0])


# --- latent handling -------------------------------------------------------
def test_split_latent_layout():
    latent = torch.arange(2 * (2 * 3 + 2), dtype=torch.float32).reshape(2, -1)
    power, rate, rho_r, rho_p = split_latent(latent, num_users=3, power_scale_action=False)
    assert power.shape == (2, 4) and rate.shape == (2, 3) and rho_r.shape == (2, 1)
    assert rho_p is None
    latent2 = torch.zeros(2, 2 * 3 + 3)
    _, _, _, rho_p2 = split_latent(latent2, 3, True)
    assert rho_p2.shape == (2, 1)


def test_td3_actions_are_feasible_and_all_active():
    latent = 3.0 * torch.tanh(torch.randn(16, 8, generator=torch.Generator().manual_seed(0)))
    actions = latent_to_env_actions(latent, num_users=3, power_scale_action=False)
    assert len(actions) == 16
    for action in actions:
        pf, rf = action["power_fractions"], action["rate_fractions"]
        np.testing.assert_array_equal(action["schedule"], np.ones(3, dtype=np.int8))
        assert np.all(pf >= 0.0) and pf.sum() == pytest.approx(1.0, abs=1e-6)
        assert rf.sum() == pytest.approx(1.0, abs=1e-6)
        assert 0.0 < float(action["common_payload_scale"][0]) < 1.0


def test_td3_uses_the_same_transform_as_the_ppo_policy():
    """The two agents must differ in the algorithm, never in action handling."""
    from rsma_aoi.agents.distributions import latent_to_feasible_action

    k = 3
    latent = torch.randn(5, 2 * k + 2, generator=torch.Generator().manual_seed(1))
    power, rate, rho_r, _ = split_latent(latent, k, False)
    active = torch.ones(5, k, dtype=torch.bool)
    reference = latent_to_feasible_action(power, rate, rho_r, active, "squashed_gaussian")
    actions = latent_to_env_actions(latent, k, False)
    for i, action in enumerate(actions):
        np.testing.assert_allclose(
            action["power_fractions"], reference["power_fractions"][i].numpy(), atol=1e-7
        )
        np.testing.assert_allclose(
            action["rate_fractions"], reference["rate_fractions"][i].numpy(), atol=1e-7
        )


# --- trainer ---------------------------------------------------------------
def test_td3_requires_a_continuous_only_environment(tmp_path):
    env_cfg, td3_cfg = tiny_configs(scheduling_enabled=True)
    with pytest.raises(ValueError, match="continuous allocation only"):
        TD3Trainer(env_cfg, td3_cfg, run_dir=tmp_path / "bad", seed=0)


def test_td3_update_produces_finite_losses(tmp_path):
    env_cfg, td3_cfg = tiny_configs()
    trainer = TD3Trainer(env_cfg, td3_cfg, run_dir=tmp_path / "run", seed=0)
    for _ in range(40):
        trainer.collect_step()
    assert len(trainer.buffer) == 80

    seen_actor = False
    for _ in range(6):
        stats = trainer.update()
        assert np.isfinite(stats["train/critic_loss"])
        assert np.isfinite(stats["train/q1_mean"])
        if "train/actor_loss" in stats:
            seen_actor = True
            assert np.isfinite(stats["train/actor_loss"])
    assert seen_actor, "the delayed actor update never fired"
    trainer.close()


def test_td3_delayed_actor_update_respects_policy_delay(tmp_path):
    env_cfg, td3_cfg = tiny_configs()
    td3_cfg.policy_delay = 3
    trainer = TD3Trainer(env_cfg, td3_cfg, run_dir=tmp_path / "run", seed=0)
    for _ in range(40):
        trainer.collect_step()
    fired = [("train/actor_loss" in trainer.update()) for _ in range(9)]
    assert fired == [False, False, True] * 3
    trainer.close()


def test_td3_target_networks_track_the_online_networks(tmp_path):
    env_cfg, td3_cfg = tiny_configs()
    td3_cfg.policy_delay = 1
    td3_cfg.tau = 0.5
    trainer = TD3Trainer(env_cfg, td3_cfg, run_dir=tmp_path / "run", seed=0)
    for _ in range(40):
        trainer.collect_step()
    before = [p.detach().clone() for p in trainer.actor_target.parameters()]
    trainer.update()
    after = list(trainer.actor_target.parameters())
    assert any(not torch.allclose(a, b) for a, b in zip(before, after))
    # targets must never accumulate gradients
    assert all(not p.requires_grad for p in trainer.actor_target.parameters())
    assert all(not p.requires_grad for p in trainer.critic_target.parameters())
    trainer.close()


def test_td3_twin_critics_are_distinct(tmp_path):
    env_cfg, td3_cfg = tiny_configs()
    trainer = TD3Trainer(env_cfg, td3_cfg, run_dir=tmp_path / "run", seed=0)
    obs = torch.randn(8, trainer.obs_dim)
    action = torch.randn(8, trainer.action_dim)
    q1, q2 = trainer.critic(obs, action)
    assert q1.shape == (8,) and q2.shape == (8,)
    assert not torch.allclose(q1, q2), "the twin critics must be independently initialised"
    torch.testing.assert_close(trainer.critic.q1(obs, action), q1)
    trainer.close()


def test_td3_warmup_uses_uniform_random_latents(tmp_path):
    env_cfg, td3_cfg = tiny_configs()
    td3_cfg.learning_starts = 10_000
    trainer = TD3Trainer(env_cfg, td3_cfg, run_dir=tmp_path / "run", seed=0)
    latents = np.stack(
        [trainer._select_latent(np.zeros((4, trainer.obs_dim), np.float32)).numpy()
         for _ in range(50)]
    )
    assert np.abs(latents).max() <= td3_cfg.latent_scale + 1e-6
    assert latents.std() > 0.5 * td3_cfg.latent_scale, "warm-up should explore widely"
    trainer.close()


def test_td3_actor_output_is_bounded(tmp_path):
    env_cfg, td3_cfg = tiny_configs()
    trainer = TD3Trainer(env_cfg, td3_cfg, run_dir=tmp_path / "run", seed=0)
    out = trainer.actor(torch.randn(32, trainer.obs_dim) * 50.0)
    assert torch.all(out.abs() <= td3_cfg.latent_scale + 1e-6)
    trainer.close()


def test_td3_training_loop_and_checkpoint_roundtrip(tmp_path):
    env_cfg, td3_cfg = tiny_configs()
    trainer = TD3Trainer(env_cfg, td3_cfg, run_dir=tmp_path / "run", seed=1)
    metrics = trainer.train()
    assert np.isfinite(metrics["train/critic_loss"])
    assert trainer.global_step == td3_cfg.total_timesteps
    assert (tmp_path / "run" / "config_snapshot.yaml").exists()
    assert (tmp_path / "run" / "run_meta.json").exists()

    ckpt = trainer.save_checkpoint(tmp_path / "run" / "checkpoint.pt")
    policy, env_loaded, td3_loaded = TD3Policy.from_checkpoint(ckpt)
    assert env_loaded.num_users == env_cfg.num_users
    assert td3_loaded.hidden_sizes == td3_cfg.hidden_sizes

    obs = np.zeros((1, trainer.obs_dim), dtype=np.float32)
    np.testing.assert_allclose(
        policy.act(obs)[0]["power_fractions"],
        trainer.make_policy(deterministic=True).act(obs)[0]["power_fractions"],
        atol=1e-6,
    )

    fresh = TD3Trainer(env_cfg, td3_cfg, run_dir=tmp_path / "b", seed=1)
    fresh.load_checkpoint(ckpt)
    assert fresh.global_step == trainer.global_step
    fresh.close()
    trainer.close()


def test_td3_deterministic_evaluation_is_repeatable_and_feasible(tmp_path):
    env_cfg, td3_cfg = tiny_configs()
    trainer = TD3Trainer(env_cfg, td3_cfg, run_dir=tmp_path / "run", seed=2)
    for _ in range(40):
        trainer.collect_step()
    trainer.update()
    policy = trainer.make_policy(deterministic=True)

    a = evaluate_policy(policy, env_cfg, episodes=2, eval_seed=7, algorithm="td3")
    b = evaluate_policy(policy, env_cfg, episodes=2, eval_seed=7, algorithm="td3")
    np.testing.assert_allclose(a["weighted_avg_aoi"], b["weighted_avg_aoi"])
    assert (a["violation_rate"] == 0.0).all()
    assert (a["avg_power"] <= env_cfg.total_power + 1e-9).all()
    assert (a["avg_num_scheduled"] == env_cfg.num_users).all()
    trainer.close()


def test_td3_stochastic_policy_differs_from_deterministic(tmp_path):
    env_cfg, td3_cfg = tiny_configs()
    trainer = TD3Trainer(env_cfg, td3_cfg, run_dir=tmp_path / "run", seed=3)
    obs = np.zeros((4, trainer.obs_dim), dtype=np.float32)
    det = np.stack([a["power_fractions"] for a in trainer.make_policy(True).act(obs)])
    sto = np.stack([a["power_fractions"] for a in trainer.make_policy(False).act(obs)])
    np.testing.assert_allclose(det, det[0][None, :].repeat(4, 0), atol=1e-6)
    assert not np.allclose(det, sto)
    trainer.close()


def test_td3_is_seed_reproducible(tmp_path):
    def run(seed: int) -> np.ndarray:
        env_cfg, td3_cfg = tiny_configs()
        trainer = TD3Trainer(env_cfg, td3_cfg, run_dir=tmp_path / f"s{seed}", seed=seed)
        for _ in range(40):
            trainer.collect_step()
        trainer.update()
        flat = torch.cat([p.detach().flatten() for p in trainer.actor.parameters()])
        trainer.close()
        return flat.numpy()

    np.testing.assert_allclose(run(5), run(5), rtol=1e-6)
    assert not np.allclose(run(5), run(6))


def test_td3_stores_terminal_observation_not_the_reset_one(tmp_path):
    """On truncation the buffer must hold the final obs so bootstrapping is right."""
    env_cfg, td3_cfg = tiny_configs(episode_length=4)
    trainer = TD3Trainer(env_cfg, td3_cfg, run_dir=tmp_path / "run", seed=4)
    for _ in range(4):
        trainer.collect_step()
    # after 4 steps both envs truncated; the stored next_obs must differ from the
    # post-reset observation that the trainer carries forward
    stored = trainer.buffer.next_obs[trainer.buffer.size - 2 : trainer.buffer.size]
    current = trainer._normalize_obs(trainer._obs, update=False)
    assert not np.allclose(stored, current)
    np.testing.assert_allclose(trainer.buffer.not_done[: trainer.buffer.size], 1.0)
    trainer.close()
