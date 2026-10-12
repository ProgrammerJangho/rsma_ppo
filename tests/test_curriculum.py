"""Curriculum scheduling heads inside PPO: feasibility, log-prob, freezing."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rsma_aoi.agents.distributions import CountThenSubset, TopMCategorical, enumerate_sized_subsets
from rsma_aoi.agents.ppo import PPOTrainer, resolve_schedule_spec
from rsma_aoi.utils.config import EnvConfig, PPOConfig

HEADS = ["bernoulli", "subset_categorical", "topm_categorical", "count_then_subset",
         "fixed_top_m"]


def configs(head: str, **ppo_kwargs):
    env_cfg = EnvConfig(num_users=4, num_antennas=2, episode_length=16, aoi_max=16,
                        blocklength=64, payload_bits=96.0)
    ppo_cfg = PPOConfig(
        total_timesteps=128, num_envs=2, rollout_steps=16, minibatch_size=16,
        update_epochs=1, hidden_sizes=[32, 32], scheduling_head=head,
        conditional_allocation=True, checkpoint_interval=0, **ppo_kwargs,
    )
    return env_cfg, ppo_cfg


def test_schedule_spec_defaults_to_the_multiplexing_limit():
    env_cfg = EnvConfig(num_users=4, num_antennas=2)
    size, sizes = resolve_schedule_spec(env_cfg, PPOConfig())
    assert size == 2 and sizes == (1, 2)
    size, sizes = resolve_schedule_spec(
        env_cfg, PPOConfig(schedule_size=3, schedule_sizes=[2, 3])
    )
    assert size == 3 and sizes == (2, 3)


@pytest.mark.parametrize("head", HEADS)
def test_every_head_trains_and_produces_feasible_actions(tmp_path, head):
    env_cfg, ppo_cfg = configs(head)
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / head, seed=0)
    trainer.collect_rollout()
    metrics = trainer.update()
    assert np.isfinite(metrics["train/policy_loss"])
    assert np.isfinite(metrics["train/entropy"])
    masks = trainer.buffer.active_mask
    assert masks.any(axis=-1).all(), "every stored schedule must be non-empty"
    if head == "topm_categorical":
        assert (masks.sum(axis=-1) == 2).all()
    if head == "count_then_subset":
        assert set(np.unique(masks.sum(axis=-1))) <= {1, 2}
    if head == "fixed_top_m":
        assert (masks.sum(axis=-1) == 2).all()
    trainer.close()


@pytest.mark.parametrize("head", HEADS)
def test_ppo_ratio_is_one_before_any_update(tmp_path, head):
    """Every curriculum head must survive the buffer round-trip exactly."""
    env_cfg, ppo_cfg = configs(head)
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / head, seed=1)
    trainer.collect_rollout()
    trainer.buffer.compute_returns(ppo_cfg.gamma, ppo_cfg.gae_lambda)
    worst = 0.0
    for batch in trainer.buffer.iter_minibatches(ppo_cfg.minibatch_size, np.random.default_rng(0)):
        with torch.no_grad():
            recomputed = trainer._distribution(batch.obs, batch.latent.active_mask).log_prob(
                batch.latent
            )
        worst = max(worst, float((torch.exp(recomputed - batch.log_probs) - 1).abs().max()))
    assert worst < 1e-3, f"{head}: ratio drifted by {worst}"
    trainer.close()


def test_count_then_subset_log_prob_factorises_exactly():
    """log p(A) = log p(m) + log p(A|m), and the total mass is 1."""
    gen = torch.Generator().manual_seed(0)
    table = enumerate_sized_subsets(4, (1, 2))
    n = table.shape[0]
    count_logits = torch.randn(1, 2, generator=gen).repeat(n, 1)
    subset_logits = torch.randn(1, n, generator=gen).repeat(n, 1)
    dist = CountThenSubset(count_logits, subset_logits, 4, (1, 2))
    idx = torch.arange(n, dtype=torch.float32).unsqueeze(-1)
    log_p = dist.log_prob(idx)
    assert float(log_p.exp().sum()) == pytest.approx(1.0, abs=1e-5)

    # marginal over each size must equal p(m)
    count_probs = torch.softmax(count_logits[0], dim=-1)
    for block, size in enumerate((1, 2)):
        rows = dist.size_of_index == block
        assert float(log_p[rows].exp().sum()) == pytest.approx(float(count_probs[block]), abs=1e-5)
    assert float(dist.entropy()[0]) == pytest.approx(float(-(log_p.exp() * log_p).sum()), abs=1e-4)


def test_topm_categorical_only_emits_size_m_subsets():
    gen = torch.Generator().manual_seed(1)
    dist = TopMCategorical(torch.randn(64, 6, generator=gen), 4, 2)
    masks = dist.decode(dist.sample(torch.Generator().manual_seed(2)))
    assert (masks.sum(-1) == 2).all()
    assert float(dist.entropy().mean()) > 0.0


def test_count_prior_biases_the_initial_count_distribution(tmp_path):
    env_cfg, ppo_cfg = configs("count_then_subset", count_prior=[0.05, 0.95])
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / "prior", seed=0)
    obs = torch.zeros(256, trainer.obs_dim)
    with torch.no_grad():
        dist = trainer.model.distribution(obs)
        counts = dist.schedule.decode(dist.sample(trainer.torch_gen).schedule_raw).sum(-1)
    assert float((counts == 2).float().mean()) > 0.8, "the count prior was not applied"
    trainer.close()


def test_fixed_top_m_uses_the_max_aoi_schedule(tmp_path):
    env_cfg, ppo_cfg = configs("fixed_top_m")
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / "fixed", seed=0)
    trainer.collect_rollout()
    masks = trainer.buffer.active_mask
    assert (masks.sum(axis=-1) == 2).all()
    # the scheduling head contributes no entropy for this stage
    with torch.no_grad():
        obs = torch.zeros(4, trainer.obs_dim)
        forced = torch.tensor([[True, True, False, False]]).repeat(4, 1)
        dist = trainer.model.distribution(obs, forced_mask=forced)
        ent = dist.head_entropies(forced)
    assert "schedule" not in ent
    trainer.close()


def test_freeze_windows_zero_the_right_gradients(tmp_path):
    env_cfg, ppo_cfg = configs("topm_categorical", freeze_scheduling_steps=10_000)
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / "freeze", seed=0)
    assert trainer.global_step < ppo_cfg.freeze_scheduling_steps
    frozen = trainer._frozen_parameters()
    assert any(p is trainer.model.schedule_head.weight for p in frozen)

    before = trainer.model.schedule_head.weight.detach().clone()
    power_before = trainer.model.power_head.weight.detach().clone()
    trainer.collect_rollout()
    trainer.update()
    torch.testing.assert_close(trainer.model.schedule_head.weight, before)
    assert not torch.allclose(trainer.model.power_head.weight, power_before)

    trainer.global_step = 20_000  # past the freeze window
    assert trainer._frozen_parameters() == []
    trainer.close()


def test_allocation_freeze_window(tmp_path):
    env_cfg, ppo_cfg = configs("topm_categorical", freeze_allocation_steps=10_000)
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / "freeze2", seed=0)
    power_before = trainer.model.power_head.weight.detach().clone()
    sched_before = trainer.model.schedule_head.weight.detach().clone()
    trainer.collect_rollout()
    trainer.update()
    torch.testing.assert_close(trainer.model.power_head.weight, power_before)
    assert not torch.allclose(trainer.model.schedule_head.weight, sched_before)
    trainer.close()


def test_separate_actor_and_critic_learning_rates(tmp_path):
    env_cfg, ppo_cfg = configs("topm_categorical", actor_lr=1e-5, critic_lr=7e-4,
                               anneal_lr=False)
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / "lr", seed=0)
    groups = {g["name"]: g["lr"] for g in trainer.optimizer.param_groups}
    assert groups == {"actor": pytest.approx(1e-5), "critic": pytest.approx(7e-4)}
    trainer.collect_rollout()
    metrics = trainer.update()
    assert metrics["train/actor_lr"] == pytest.approx(1e-5)
    assert metrics["train/critic_lr"] == pytest.approx(7e-4)
    trainer.close()


def test_reference_kl_penalty_is_applied_and_annealed(tmp_path):
    import torch as _torch

    from rsma_aoi.agents.imitation import save_imitation_checkpoint

    env_cfg, base_cfg = configs("topm_categorical")
    donor_trainer = PPOTrainer(env_cfg, base_cfg, run_dir=tmp_path / "donor", seed=0)
    path = save_imitation_checkpoint(
        tmp_path / "bc.pt", donor_trainer.model, env_cfg, base_cfg, {}
    )
    donor_trainer.close()

    env_cfg, ppo_cfg = configs(
        "topm_categorical", pretrained_actor=str(path), reference_kl_coef=0.5,
        reference_kl_anneal_steps=1000,
    )
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / "kl", seed=0)
    assert trainer.reference_model is not None
    assert all(not p.requires_grad for p in trainer.reference_model.parameters())
    assert trainer._reference_kl_coef() == pytest.approx(0.5)

    trainer.collect_rollout()
    metrics = trainer.update()
    assert "train/reference_kl" in metrics and np.isfinite(metrics["train/reference_kl"])

    trainer.global_step = 500
    assert trainer._reference_kl_coef() == pytest.approx(0.25, abs=1e-6)
    trainer.global_step = 2000
    assert trainer._reference_kl_coef() == 0.0
    del _torch
    trainer.close()


def test_collect_rollout_logs_the_collapse_diagnostics(tmp_path):
    env_cfg, ppo_cfg = configs("bernoulli")
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / "diag", seed=0)
    metrics = trainer.collect_rollout()
    for key in ("env/entropy_schedule", "env/entropy_power", "env/raw_all_zero_rate",
                "env/p_count_1", "env/p_count_2"):
        assert key in metrics and np.isfinite(metrics[key])
    total = sum(metrics[f"env/p_count_{c}"] for c in range(1, env_cfg.num_users + 1))
    assert total == pytest.approx(1.0, abs=1e-6)
    trainer.close()


def test_probe_metrics_reach_the_scalar_log(tmp_path):
    """Probe diagnostics must be logged, not computed after the log call."""
    import pandas as pd

    from rsma_aoi.agents.imitation import OracleDataset, OracleProbe

    env_cfg, ppo_cfg = configs("topm_categorical")
    ppo_cfg.total_timesteps = 2 * ppo_cfg.num_envs * ppo_cfg.rollout_steps
    ppo_cfg.probe_interval = 1
    trainer = PPOTrainer(env_cfg, ppo_cfg, run_dir=tmp_path / "probe", seed=0)

    n = 16
    rng = np.random.default_rng(0)
    aoi = rng.integers(1, 9, size=(n, 4)).astype(np.float32)
    mask = np.zeros((n, 4), dtype=bool)
    np.put_along_axis(mask, np.argsort(-aoi, axis=1)[:, :2], True, axis=1)
    obs = rng.standard_normal((n, trainer.obs_dim)).astype(np.float32)
    h = (rng.standard_normal((n, 4, 2)) + 1j * rng.standard_normal((n, 4, 2))) / np.sqrt(2)
    dataset = OracleDataset(
        observation=torch.as_tensor(obs), aoi=torch.as_tensor(aoi),
        oracle_mask=torch.as_tensor(mask),
        oracle_power=torch.as_tensor(np.full((n, 5), 0.2, dtype=np.float32)),
        oracle_rate=torch.as_tensor(np.full((n, 4), 0.25, dtype=np.float32)),
        oracle_rho_r=torch.as_tensor(np.full(n, 0.5, dtype=np.float32)),
        oracle_expected_aoi=torch.as_tensor(np.full(n, 8.0, dtype=np.float32)),
        objective_gap=torch.as_tensor(np.zeros(n, dtype=np.float32)),
        h_hat=h, h_true=h, trajectory=np.array(["t"] * n), source=np.array(["toy"] * n),
        config_hash="toy",
    )
    trainer.probe = OracleProbe(dataset, env_cfg, size=n)
    trainer.train()
    trainer.close()

    df = pd.read_csv(tmp_path / "probe" / "scalars.csv")
    for key in ("probe/exact_subset_accuracy", "probe/mean_active", "probe/count_accuracy"):
        assert key in df.columns, f"{key} never reached scalars.csv"
        assert df[key].notna().any()
