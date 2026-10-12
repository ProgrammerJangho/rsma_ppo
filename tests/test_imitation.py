"""Behaviour cloning, curriculum heads and PPO warm-start."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rsma_aoi.agents.imitation import (
    ImitationLossWeights, OracleDataset, OracleProbe, beta_nll, evaluate_imitation,
    fixed_top_m_masks, imitation_loss, load_pretrained_actor, match_subset_index,
    save_imitation_checkpoint, simplex_cross_entropy,
)
from rsma_aoi.agents.networks import HybridActorCritic
from rsma_aoi.agents.ppo import PPOTrainer, max_aoi_masks, resolve_schedule_spec
from rsma_aoi.utils.config import EnvConfig, PPOConfig

K, NT, OBS = 4, 2, 12


def make_model(head: str = "count_then_subset", conditional: bool = True) -> HybridActorCritic:
    torch.manual_seed(0)
    return HybridActorCritic(
        obs_dim=OBS, num_users=K, hidden_sizes=[32, 32], scheduling_head=head,
        conditional_allocation=conditional, schedule_size=2, schedule_sizes=(1, 2),
    )


def toy_dataset(n: int = 64, seed: int = 0) -> OracleDataset:
    """Synthetic oracle labels: always schedule the two oldest users."""
    rng = np.random.default_rng(seed)
    aoi = rng.integers(1, 10, size=(n, K)).astype(np.float32)
    order = np.argsort(-aoi, axis=1)
    mask = np.zeros((n, K), dtype=bool)
    np.put_along_axis(mask, order[:, :2], True, axis=1)
    power = np.zeros((n, K + 1), dtype=np.float32)
    power[:, 0] = 0.4
    power[:, 1:][mask] = 0.3
    rate = np.zeros((n, K), dtype=np.float32)
    rate[mask] = 0.5
    obs = np.concatenate([aoi / 10.0, rng.standard_normal((n, OBS - K))], axis=1).astype(np.float32)
    h = (rng.standard_normal((n, K, NT)) + 1j * rng.standard_normal((n, K, NT))) / np.sqrt(2)
    return OracleDataset(
        observation=torch.as_tensor(obs),
        aoi=torch.as_tensor(aoi),
        oracle_mask=torch.as_tensor(mask),
        oracle_power=torch.as_tensor(power),
        oracle_rate=torch.as_tensor(rate),
        oracle_rho_r=torch.as_tensor(rng.uniform(0.2, 0.8, n).astype(np.float32)),
        oracle_expected_aoi=torch.as_tensor(rng.uniform(5, 10, n).astype(np.float32)),
        objective_gap=torch.as_tensor(rng.uniform(0, 1, n).astype(np.float32)),
        h_hat=h, h_true=h,
        trajectory=np.array([f"t{i // 8}" for i in range(n)]),
        source=np.array(["toy"] * n),
        config_hash="toyhash",
    )


# --- loss components -------------------------------------------------------
def test_simplex_cross_entropy_is_minimised_at_the_target():
    target = torch.tensor([[0.0, 0.5, 0.5]])
    mask = torch.tensor([[False, True, True]])
    perfect = torch.tensor([[0.0, 1.0, 1.0]])       # softmax -> [0, .5, .5]
    skewed = torch.tensor([[0.0, 5.0, -5.0]])
    assert float(simplex_cross_entropy(target, perfect, mask)) < float(
        simplex_cross_entropy(target, skewed, mask)
    )
    # the masked coordinate cannot influence the loss
    a = simplex_cross_entropy(target, perfect, mask)
    b = simplex_cross_entropy(target, perfect + torch.tensor([[99.0, 0.0, 0.0]]), mask)
    assert float(a) == pytest.approx(float(b), abs=1e-6)


def test_beta_nll_is_finite_and_lowest_near_the_value():
    raw = torch.tensor([[2.0, 2.0]])
    near = beta_nll(raw, torch.tensor([0.5]))
    far = beta_nll(raw, torch.tensor([0.99]))
    assert torch.isfinite(near) and torch.isfinite(far)
    assert float(near) < float(far)


def test_match_subset_index_recovers_exact_and_nearest_subsets():
    from rsma_aoi.agents.distributions import enumerate_sized_subsets

    table = enumerate_sized_subsets(4, (2,))
    exact = torch.tensor([[True, True, False, False], [False, False, True, True]])
    idx = match_subset_index(table, exact)
    np.testing.assert_array_equal(table[idx].numpy(), exact.numpy())
    # a size-3 subset is not representable; the closest size-2 subset is chosen
    odd = torch.tensor([[True, True, True, False]])
    chosen = table[match_subset_index(table, odd)]
    assert int((chosen & odd).sum()) == 2


@pytest.mark.parametrize(
    "head", ["bernoulli", "subset_categorical", "topm_categorical", "count_then_subset"]
)
def test_imitation_loss_is_finite_for_every_head(head):
    model = make_model(head)
    data = toy_dataset(32)
    loss, components = imitation_loss(
        model, data.observation, data.oracle_mask, data.oracle_power,
        data.oracle_rate, data.oracle_rho_r, ImitationLossWeights(),
    )
    assert torch.isfinite(loss)
    assert all(np.isfinite(v) for v in components.values())
    if head == "count_then_subset":
        assert "loss/count" in components
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_behaviour_cloning_reduces_the_loss_on_a_toy_dataset():
    model = make_model("topm_categorical")
    data = toy_dataset(128)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(60):
        loss, _ = imitation_loss(
            model, data.observation, data.oracle_mask, data.oracle_power,
            data.oracle_rate, data.oracle_rho_r, ImitationLossWeights(),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())
    assert losses[-1] < losses[0] * 0.9, f"BC did not learn: {losses[0]:.3f} -> {losses[-1]:.3f}"

    # and it should now mostly reproduce the "two oldest users" rule
    from rsma_aoi.agents.imitation import predict_actions

    mask = predict_actions(model, data.observation)["active_mask"]
    accuracy = float((mask == data.oracle_mask).all(dim=-1).float().mean())
    assert accuracy > 0.5, f"subset accuracy only {accuracy:.2f}"


def test_allocation_loss_ignores_inactive_dimensions():
    model = make_model("topm_categorical")
    data = toy_dataset(16)
    base, _ = imitation_loss(
        model, data.observation, data.oracle_mask, data.oracle_power,
        data.oracle_rate, data.oracle_rho_r, ImitationLossWeights(),
    )
    # perturbing the target on inactive users must not change the loss
    power = data.oracle_power.clone()
    power[:, 1:][~data.oracle_mask] = 7.0
    perturbed, _ = imitation_loss(
        model, data.observation, data.oracle_mask, power,
        data.oracle_rate, data.oracle_rho_r, ImitationLossWeights(),
    )
    assert base.detach().item() == pytest.approx(perturbed.detach().item(), abs=1e-5)


def test_allocation_only_gate_labels_train_the_exact_common_off_decision():
    torch.manual_seed(4)
    model = HybridActorCritic(
        obs_dim=OBS, num_users=K, hidden_sizes=[32, 32],
        scheduling_enabled=False, common_gate_enabled=True,
        allocation_control="joint",
    )
    data = toy_dataset(32)
    gate = torch.zeros(32, dtype=torch.bool)
    gate[::2] = True
    power = data.oracle_power.clone()
    power[~gate, 0] = 0.0
    private = power[~gate, 1:]
    power[~gate, 1:] = private / private.sum(dim=-1, keepdim=True)
    rho = data.oracle_rho_r.clone()
    rho[~gate] = 0.0
    loss, components = imitation_loss(
        model, data.observation, torch.ones_like(data.oracle_mask), power,
        data.oracle_rate, rho, ImitationLossWeights(), gate,
    )
    assert torch.isfinite(loss)
    assert "loss/gate" in components
    loss.backward()
    assert model.common_gate_head is not None
    assert model.common_gate_head.weight.grad is not None
    assert torch.isfinite(model.common_gate_head.weight.grad).all()
    # Scheduling is outside this policy class and must not affect the loss.
    assert components["loss/schedule"] == 0.0


def test_evaluate_imitation_reports_supervised_and_physical_metrics():
    cfg = EnvConfig(num_users=K, num_antennas=NT, blocklength=64, payload_bits=96.0)
    model = make_model("topm_categorical")
    metrics = evaluate_imitation(model, toy_dataset(24), cfg, max_phy_states=8)
    for key in ("val/exact_subset_accuracy", "val/scheduling_f1", "val/count_accuracy",
                "val/power_l1", "val/rate_l1", "val/rho_mae",
                "val/predicted_expected_aoi", "val/objective_gap"):
        assert key in metrics and np.isfinite(metrics[key])
    assert 0.0 <= metrics["val/exact_subset_accuracy"] <= 1.0


def test_fixed_top_m_masks_match_the_trainer_rule():
    aoi = torch.tensor([[1.0, 9.0, 3.0, 7.0]])
    mask = fixed_top_m_masks(aoi, np.ones(4), 2)
    np.testing.assert_array_equal(mask.numpy(), [[False, True, False, True]])
    states = [{"aoi": np.array([1.0, 9.0, 3.0, 7.0]), "weights": np.ones(4)}]
    np.testing.assert_array_equal(max_aoi_masks(states, 2), mask.numpy())


# --- checkpoint / warm start ----------------------------------------------
def test_pretrained_actor_roundtrip_and_critic_reset(tmp_path):
    cfg = EnvConfig(num_users=K, num_antennas=NT, episode_length=8, blocklength=64,
                    payload_bits=96.0)
    ppo_cfg = PPOConfig(num_envs=2, rollout_steps=8, minibatch_size=8, update_epochs=1,
                        hidden_sizes=[32, 32], scheduling_head="count_then_subset",
                        conditional_allocation=True, checkpoint_interval=0)
    probe_trainer = PPOTrainer(cfg, ppo_cfg, run_dir=tmp_path / "probe", seed=0)
    obs_dim = probe_trainer.obs_dim
    probe_trainer.close()

    size, sizes = resolve_schedule_spec(cfg, ppo_cfg)
    donor = HybridActorCritic(
        obs_dim=obs_dim, num_users=K, hidden_sizes=[32, 32],
        scheduling_head="count_then_subset", conditional_allocation=True,
        schedule_size=size, schedule_sizes=sizes,
    )
    with torch.no_grad():
        for p in donor.parameters():
            p.add_(0.37)
    path = save_imitation_checkpoint(
        tmp_path / "bc.pt", donor, cfg, ppo_cfg, {"val/exact_subset_accuracy": 0.5}
    )

    warm_cfg = PPOConfig(**{**ppo_cfg.to_dict(), "pretrained_actor": str(path),
                            "warm_start_log_std": -2.0, "reset_critic": True})
    trainer = PPOTrainer(cfg, warm_cfg, run_dir=tmp_path / "warm", seed=0)
    torch.testing.assert_close(
        trainer.model.schedule_head.weight, donor.schedule_head.weight
    )
    torch.testing.assert_close(trainer.model.power_head.weight, donor.power_head.weight)
    torch.testing.assert_close(
        trainer.model.power_log_std,
        torch.full_like(trainer.model.power_log_std, -2.0),
    )
    torch.testing.assert_close(
        trainer.model.rate_log_std,
        torch.full_like(trainer.model.rate_log_std, -2.0),
    )
    # the critic must NOT have been copied (it was reset)
    assert not torch.allclose(trainer.model.value_head.weight, donor.value_head.weight)
    assert trainer.warm_start_meta["metrics"]["val/exact_subset_accuracy"] == 0.5
    trainer.close()


def test_freeze_actor_steps_excludes_only_critic(tmp_path):
    cfg = EnvConfig(num_users=K, num_antennas=NT, episode_length=8, blocklength=64,
                    payload_bits=96.0)
    ppo_cfg = PPOConfig(
        num_envs=2, rollout_steps=8, minibatch_size=8, update_epochs=1,
        hidden_sizes=[32, 32], scheduling_head="count_then_subset",
        freeze_actor_steps=16, checkpoint_interval=0,
    )
    trainer = PPOTrainer(cfg, ppo_cfg, run_dir=tmp_path / "freeze", seed=0)
    frozen_ids = {id(p) for p in trainer._frozen_parameters()}
    for name, param in trainer.model.named_parameters():
        if name.startswith(("critic_trunk", "value_head")):
            assert id(param) not in frozen_ids
        else:
            assert id(param) in frozen_ids
    trainer.global_step = 16
    assert trainer._frozen_parameters() == []
    trainer.close()


def test_warm_start_restores_the_observation_normaliser(tmp_path):
    """BC trains under its own obs scale; PPO must continue from it, not from scratch."""
    from rsma_aoi.utils.running_stats import RunningMeanStd

    cfg = EnvConfig(num_users=K, num_antennas=NT, episode_length=8, blocklength=64,
                    payload_bits=96.0)
    ppo_cfg = PPOConfig(num_envs=2, rollout_steps=8, minibatch_size=8, update_epochs=1,
                        hidden_sizes=[32, 32], scheduling_head="topm_categorical",
                        conditional_allocation=True, checkpoint_interval=0)
    probe = PPOTrainer(cfg, ppo_cfg, run_dir=tmp_path / "probe", seed=0)
    obs_dim = probe.obs_dim
    probe.close()

    size, sizes = resolve_schedule_spec(cfg, ppo_cfg)
    donor = HybridActorCritic(
        obs_dim=obs_dim, num_users=K, hidden_sizes=[32, 32],
        scheduling_head="topm_categorical", conditional_allocation=True,
        schedule_size=size, schedule_sizes=sizes,
    )
    rms = RunningMeanStd(shape=(obs_dim,))
    rms.update(np.random.default_rng(0).normal(3.0, 2.0, size=(500, obs_dim)))
    path = save_imitation_checkpoint(
        tmp_path / "bc.pt", donor, cfg, ppo_cfg, {}, obs_rms=rms
    )

    warm_cfg = PPOConfig(**{
        **ppo_cfg.to_dict(), "pretrained_actor": str(path),
        "freeze_observation_stats": True,
    })
    trainer = PPOTrainer(cfg, warm_cfg, run_dir=tmp_path / "warm", seed=0)
    assert trainer.obs_rms is not None
    np.testing.assert_allclose(trainer.obs_rms.mean, rms.mean, rtol=1e-6)
    np.testing.assert_allclose(trainer.obs_rms.var, rms.var, rtol=1e-6)
    count_before = trainer.obs_rms.count
    mean_before = trainer.obs_rms.mean.copy()
    # Training still runs end to end while the warm actor's input scale stays fixed.
    trainer.collect_rollout()
    assert trainer.obs_rms.count == count_before
    np.testing.assert_array_equal(trainer.obs_rms.mean, mean_before)
    assert np.isfinite(trainer.update()["train/policy_loss"])
    trainer.close()


def test_load_pretrained_actor_reports_mismatches(tmp_path):
    cfg = EnvConfig(num_users=K, num_antennas=NT)
    ppo_cfg = PPOConfig(num_envs=2, rollout_steps=8, minibatch_size=8)
    donor = make_model("topm_categorical")
    path = save_imitation_checkpoint(tmp_path / "bc.pt", donor, cfg, ppo_cfg, {})
    target = make_model("topm_categorical")
    info = load_pretrained_actor(target, path, strict=True)
    torch.testing.assert_close(target.power_head.weight, donor.power_head.weight)
    assert info["missing"] == [] and info["unexpected"] == []


def test_oracle_probe_reports_the_count_distribution():
    cfg = EnvConfig(num_users=K, num_antennas=NT, blocklength=64, payload_bits=96.0)
    probe = OracleProbe(toy_dataset(32), cfg, size=16)
    metrics = probe.evaluate(make_model("topm_categorical"))
    assert probe.size == 16
    assert metrics["probe/mean_active"] == pytest.approx(2.0)
    assert sum(metrics[f"probe/p_count_{c}"] for c in range(K + 1)) == pytest.approx(1.0)
    assert 0.0 <= metrics["probe/exact_subset_accuracy"] <= 1.0
