"""Tiny end-to-end PPO runs: finite losses, checkpointing, deterministic eval."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rsma_aoi.agents.ppo import PPOPolicy, PPOTrainer
from rsma_aoi.evaluation.evaluator import evaluate_policy
from rsma_aoi.utils.config import EnvConfig, PPOConfig


def tiny_configs(**env_kwargs) -> tuple[EnvConfig, PPOConfig]:
    env_cfg = EnvConfig(
        **{
            "num_users": 3, "num_antennas": 2, "episode_length": 16,
            "aoi_max": 16, "blocklength": 64, "payload_bits": 96.0, **env_kwargs,
        }
    )
    ppo_cfg = PPOConfig(
        total_timesteps=256, num_envs=2, rollout_steps=32, minibatch_size=32,
        update_epochs=2, hidden_sizes=[32, 32], checkpoint_interval=0, device="cpu",
    )
    return env_cfg, ppo_cfg


def test_two_ppo_updates_produce_finite_losses(tmp_path):
    env_cfg, ppo_cfg = tiny_configs()
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / "run", seed=0)
    assert trainer.num_updates == 4

    for _ in range(2):
        rollout_metrics = trainer.collect_rollout()
        update_metrics = trainer.update()
        for key, value in {**rollout_metrics, **update_metrics}.items():
            assert np.isfinite(value), f"non-finite metric {key}={value}"
        assert update_metrics["train/entropy"] > 0.0
        assert 0.0 <= update_metrics["train/clip_fraction"] <= 1.0
        assert update_metrics["train/approx_kl"] >= -1e-6
    assert trainer.global_step == 2 * ppo_cfg.batch_size
    trainer.close()


def test_parameters_actually_change():
    env_cfg, ppo_cfg = tiny_configs()
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp, seed=0)
        before = [p.detach().clone() for p in trainer.model.parameters()]
        trainer.collect_rollout()
        trainer.update()
        after = list(trainer.model.parameters())
        assert any(not torch.allclose(a, b) for a, b in zip(before, after))
        trainer.close()


def test_full_training_loop_and_checkpoint_roundtrip(tmp_path):
    env_cfg, ppo_cfg = tiny_configs()
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / "run", seed=1)
    metrics = trainer.train()
    assert np.isfinite(metrics["train/policy_loss"])
    assert (tmp_path / "run" / "config_snapshot.yaml").exists()
    assert (tmp_path / "run" / "scalars.csv").exists()
    assert (tmp_path / "run" / "run_meta.json").exists()

    ckpt = trainer.save_checkpoint(tmp_path / "run" / "checkpoint.pt")
    assert ckpt.exists()

    policy, loaded_env_cfg, loaded_ppo_cfg = PPOPolicy.from_checkpoint(ckpt)
    assert loaded_env_cfg.num_users == env_cfg.num_users
    assert loaded_ppo_cfg.hidden_sizes == ppo_cfg.hidden_sizes

    obs = np.zeros((1, trainer.obs_dim), dtype=np.float32)
    reloaded = policy.act(obs)[0]
    original = trainer.make_policy(deterministic=True).act(obs)[0]
    np.testing.assert_allclose(reloaded["power_fractions"], original["power_fractions"], atol=1e-6)
    trainer.close()


def test_load_checkpoint_restores_the_trainer_state(tmp_path):
    env_cfg, ppo_cfg = tiny_configs()
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / "a", seed=2)
    trainer.collect_rollout()
    trainer.update()
    ckpt = trainer.save_checkpoint(tmp_path / "a" / "checkpoint.pt")
    step = trainer.global_step
    trainer.close()

    fresh = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / "b", seed=2)
    fresh.load_checkpoint(ckpt)
    assert fresh.global_step == step
    fresh.close()


def test_deterministic_evaluation_is_repeatable(tmp_path):
    env_cfg, ppo_cfg = tiny_configs()
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / "run", seed=3)
    trainer.collect_rollout()
    trainer.update()
    policy = trainer.make_policy(deterministic=True)

    a = evaluate_policy(policy, env_cfg, episodes=2, eval_seed=42, algorithm="ppo")
    b = evaluate_policy(policy, env_cfg, episodes=2, eval_seed=42, algorithm="ppo")
    np.testing.assert_allclose(a["weighted_avg_aoi"], b["weighted_avg_aoi"])
    assert np.all(np.isfinite(a["weighted_avg_aoi"]))
    assert (a["violation_rate"] == 0.0).all(), "the policy must never violate the constraints"
    assert (a["avg_power"] <= env_cfg.total_power + 1e-9).all()
    assert (a["inference_latency_ms"] > 0.0).all()
    trainer.close()


def test_stochastic_and_deterministic_policies_differ(tmp_path):
    env_cfg, ppo_cfg = tiny_configs()
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / "run", seed=4)
    obs = np.zeros((4, trainer.obs_dim), dtype=np.float32)
    det = trainer.make_policy(deterministic=True).act(obs)
    sto = trainer.make_policy(deterministic=False).act(obs)
    det_powers = np.stack([a["power_fractions"] for a in det])
    sto_powers = np.stack([a["power_fractions"] for a in sto])
    np.testing.assert_allclose(det_powers, det_powers[0][None, :].repeat(4, 0), atol=1e-6)
    assert not np.allclose(sto_powers, det_powers)
    trainer.close()


@pytest.mark.parametrize("scalar_dist", ["beta", "squashed_gaussian"])
def test_both_scalar_distributions_train(tmp_path, scalar_dist):
    env_cfg, ppo_cfg = tiny_configs()
    ppo_cfg.scalar_dist = scalar_dist
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / scalar_dist, seed=5)
    trainer.collect_rollout()
    metrics = trainer.update()
    assert np.isfinite(metrics["train/policy_loss"])
    assert np.isfinite(metrics["train/entropy"])
    trainer.close()


def test_continuous_only_mode_trains(tmp_path):
    env_cfg, ppo_cfg = tiny_configs(scheduling_enabled=False)
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / "cont", seed=6)
    trainer.collect_rollout()
    metrics = trainer.update()
    assert np.isfinite(metrics["train/policy_loss"])
    assert np.all(trainer.buffer.active_mask), "every user must be active without scheduling"
    trainer.close()


def test_reward_and_observation_normalisation_paths(tmp_path):
    env_cfg, ppo_cfg = tiny_configs()
    ppo_cfg.normalize_reward = True
    ppo_cfg.normalize_observation = True
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / "norm", seed=7)
    trainer.collect_rollout()
    metrics = trainer.update()
    assert np.isfinite(metrics["train/value_loss"])
    assert trainer.obs_rms is not None and trainer.obs_rms.count > 1
    assert np.all(np.isfinite(trainer.buffer.rewards))
    trainer.close()


def test_target_kl_early_stopping_can_trigger(tmp_path):
    env_cfg, ppo_cfg = tiny_configs()
    ppo_cfg.target_kl = 1e-9   # forces an early stop after the first epoch
    ppo_cfg.update_epochs = 5
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / "kl", seed=8)
    trainer.collect_rollout()
    metrics = trainer.update()
    assert metrics["train/early_stop"] == 1.0
    assert metrics["train/epochs_completed"] < ppo_cfg.update_epochs
    trainer.close()


def test_ppo_training_is_seed_reproducible(tmp_path):
    def run(seed: int) -> np.ndarray:
        env_cfg, ppo_cfg = tiny_configs()
        trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / f"s{seed}", seed=seed)
        trainer.collect_rollout()
        trainer.update()
        flat = torch.cat([p.detach().flatten() for p in trainer.model.parameters()])
        trainer.close()
        return flat.numpy()

    np.testing.assert_allclose(run(11), run(11), rtol=1e-6)
    assert not np.allclose(run(11), run(12))


def test_stored_log_probs_are_exactly_reproducible_from_the_buffer(tmp_path):
    """The PPO ratio must be 1 before any gradient step.

    This is the end-to-end version of the correctness requirement: the latents
    and masks that go through the numpy buffer must re-score to the same
    log-probability under the unchanged policy, otherwise the importance ratio
    is meaningless.
    """
    env_cfg, ppo_cfg = tiny_configs()
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / "run", seed=9)
    trainer.collect_rollout()
    trainer.buffer.compute_returns(ppo_cfg.gamma, ppo_cfg.gae_lambda)

    max_error = 0.0
    for batch in trainer.buffer.iter_minibatches(ppo_cfg.minibatch_size, np.random.default_rng(0)):
        with torch.no_grad():
            recomputed = trainer.model.distribution(batch.obs).log_prob(batch.latent)
        ratio = torch.exp(recomputed - batch.log_probs)
        max_error = max(max_error, float((ratio - 1.0).abs().max()))
    assert max_error < 1e-3, f"PPO ratio drifted before any update (max |ratio-1| = {max_error})"
    trainer.close()


def test_replayed_latents_reproduce_the_environment_actions(tmp_path):
    """Transforming a stored latent must give back the executed action."""
    from rsma_aoi.agents.ppo import latent_to_env_actions

    env_cfg, ppo_cfg = tiny_configs()
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / "run", seed=10)
    obs = trainer._normalize_obs(trainer._obs, update=False)
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    dist = trainer.model.distribution(obs_t)
    latent = dist.sample(trainer.torch_gen)
    first = latent_to_env_actions(dist, latent)
    second = latent_to_env_actions(trainer.model.distribution(obs_t), latent)
    for a, b in zip(first, second):
        np.testing.assert_allclose(a["power_fractions"], b["power_fractions"], atol=1e-7)
        np.testing.assert_allclose(a["rate_fractions"], b["rate_fractions"], atol=1e-7)
        np.testing.assert_array_equal(a["schedule"], b["schedule"])
    trainer.close()


def test_requesting_cuda_without_cuda_fails_with_a_clear_message():
    from rsma_aoi.agents.ppo import resolve_device

    assert resolve_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            resolve_device("cuda")


def test_subset_categorical_head_trains_and_never_needs_a_repair(tmp_path):
    """The categorical head must produce feasible schedules with no repair rule."""
    env_cfg, ppo_cfg = tiny_configs()
    ppo_cfg.scheduling_head = "subset_categorical"
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / "cat", seed=0)
    assert trainer.model.schedule_dim == 2**env_cfg.num_users - 1   # logit width
    assert trainer.model.schedule_action_dim == 1                    # stored index
    assert trainer.buffer.schedule_dim == 1

    trainer.collect_rollout()
    metrics = trainer.update()
    assert np.isfinite(metrics["train/policy_loss"])
    assert np.isfinite(metrics["train/entropy"])
    # every stored mask is non-empty without any repair being applied
    assert trainer.buffer.active_mask.any(axis=-1).all()

    ckpt = trainer.save_checkpoint(tmp_path / "cat" / "checkpoint.pt")
    policy, _, loaded_cfg = PPOPolicy.from_checkpoint(ckpt)
    assert loaded_cfg.scheduling_head == "subset_categorical"
    action = policy.act(np.zeros((1, trainer.obs_dim), dtype=np.float32))[0]
    assert action["schedule"].sum() >= 1
    trainer.close()


def test_categorical_head_ratio_is_one_before_any_update(tmp_path):
    env_cfg, ppo_cfg = tiny_configs()
    ppo_cfg.scheduling_head = "subset_categorical"
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / "cat", seed=1)
    trainer.collect_rollout()
    trainer.buffer.compute_returns(ppo_cfg.gamma, ppo_cfg.gae_lambda)
    max_error = 0.0
    for batch in trainer.buffer.iter_minibatches(ppo_cfg.minibatch_size, np.random.default_rng(0)):
        with torch.no_grad():
            recomputed = trainer.model.distribution(batch.obs).log_prob(batch.latent)
        max_error = max(max_error, float((torch.exp(recomputed - batch.log_probs) - 1).abs().max()))
    assert max_error < 1e-3
    trainer.close()


def test_conditional_allocation_makes_the_heads_schedule_dependent(tmp_path):
    """Without conditioning, the allocation logits are identical for every subset."""
    env_cfg, ppo_cfg = tiny_configs()
    obs = torch.zeros(4, 1)

    for conditional in (False, True):
        cfg = PPOConfig(**{**ppo_cfg.to_dict(), "conditional_allocation": conditional})
        trainer = PPOTrainer(env_cfg, cfg, run_dir=tmp_path / f"c{conditional}", seed=0)
        obs = torch.zeros(4, trainer.obs_dim)
        dist = trainer.model.distribution(obs)
        k = env_cfg.num_users
        mask_a = torch.zeros(4, k, dtype=torch.bool)
        mask_a[:, 0] = True
        mask_b = torch.ones(4, k, dtype=torch.bool)

        power_a = dist._continuous(mask_a)["power_mean"]
        power_b = dist._continuous(mask_b)["power_mean"]
        if conditional:
            assert not torch.allclose(power_a, power_b), (
                "conditioned heads must respond to the schedule"
            )
        else:
            torch.testing.assert_close(power_a, power_b)
        trainer.close()


def test_conditional_allocation_preserves_exact_rescoring(tmp_path):
    """The autoregressive factorisation must keep the PPO ratio at 1."""
    env_cfg, ppo_cfg = tiny_configs()
    ppo_cfg.conditional_allocation = True
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / "cond", seed=2)
    trainer.collect_rollout()
    trainer.buffer.compute_returns(ppo_cfg.gamma, ppo_cfg.gae_lambda)
    max_error = 0.0
    for batch in trainer.buffer.iter_minibatches(ppo_cfg.minibatch_size, np.random.default_rng(0)):
        with torch.no_grad():
            recomputed = trainer.model.distribution(batch.obs).log_prob(batch.latent)
        max_error = max(max_error, float((torch.exp(recomputed - batch.log_probs) - 1).abs().max()))
    assert max_error < 1e-3, f"ratio drifted with conditional allocation: {max_error}"

    metrics = trainer.update()
    assert np.isfinite(metrics["train/policy_loss"])
    trainer.close()


def test_conditional_allocation_actions_stay_feasible(tmp_path):
    env_cfg, ppo_cfg = tiny_configs()
    ppo_cfg.conditional_allocation = True
    ppo_cfg.scheduling_head = "subset_categorical"
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / "cc", seed=3)
    trainer.collect_rollout()
    obs = np.zeros((4, trainer.obs_dim), dtype=np.float32)
    for action in trainer.make_policy(deterministic=False).act(obs):
        pf, rf = action["power_fractions"], action["rate_fractions"]
        active = action["schedule"].astype(bool)
        assert active.any()
        assert pf.sum() == pytest.approx(1.0, abs=1e-6)
        np.testing.assert_array_equal(pf[1:][~active], 0.0)
        np.testing.assert_array_equal(rf[~active], 0.0)
    ckpt = trainer.save_checkpoint(tmp_path / "cc" / "checkpoint.pt")
    policy, _, loaded = PPOPolicy.from_checkpoint(ckpt)
    assert loaded.conditional_allocation is True
    assert policy.act(obs)[0]["power_fractions"].sum() == pytest.approx(1.0, abs=1e-6)
    trainer.close()
