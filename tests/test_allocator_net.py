"""Objective-aware learned allocator: feasibility, masking, finite losses."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rsma_aoi.agents.allocator_net import (
    LOSS_PRESETS, AllocatorNet, allocator_losses, evaluate_allocator, precompute_batch,
    hybrid_actor_objective_losses, sdma_actor_objective_losses, simplex_cross_entropy,
)
from rsma_aoi.agents.networks import HybridActorCritic
from rsma_aoi.utils.config import EnvConfig

K, NT, N = 4, 2, 24


def toy_batch(seed: int = 0, obs_dim: int = 9):
    cfg = EnvConfig(num_users=K, num_antennas=NT, snr_db=10.0, blocklength=64,
                    payload_bits=96.0, csit_error_std=0.3)
    rng = np.random.default_rng(seed)
    h_true = (rng.standard_normal((N, K, NT)) + 1j * rng.standard_normal((N, K, NT))) / np.sqrt(2)
    h_hat = h_true + 0.2 * (rng.standard_normal((N, K, NT))
                            + 1j * rng.standard_normal((N, K, NT))) / np.sqrt(2)
    aoi = rng.integers(1, 20, size=(N, K)).astype(np.float64)
    active = np.zeros((N, K), dtype=bool)
    for i in range(N):
        active[i, rng.choice(K, size=2, replace=False)] = True
    power = np.zeros((N, K + 1))
    rate = np.zeros((N, K))
    for i in range(N):
        w = rng.dirichlet(np.ones(3))
        power[i, 0] = w[0]
        power[i, 1:][active[i]] = w[1:]
        rate[i, active[i]] = rng.dirichlet(np.ones(2))
    rho = rng.random(N)
    obs = rng.standard_normal((N, obs_dim))
    batch = precompute_batch(cfg, obs, aoi, h_true, h_hat, active, power, rate, rho)
    global batch_h_true, batch_h_hat, batch_aoi
    batch_h_true, batch_h_hat, batch_aoi = h_true, h_hat, aoi
    return cfg, batch


batch_h_true = None
batch_h_hat = None
batch_aoi = None


def test_network_output_is_always_feasible():
    cfg, batch = toy_batch()
    model = AllocatorNet(batch.features.shape[1], K, [32, 32])
    power, rate, rho = model(batch.features, batch.active)
    assert power.shape == (N, K + 1) and rate.shape == (N, K) and rho.shape == (N,)
    np.testing.assert_allclose(power.sum(-1).detach().numpy(), 1.0, atol=1e-6)
    np.testing.assert_allclose(rate.sum(-1).detach().numpy(), 1.0, atol=1e-6)
    assert torch.all(power >= 0.0) and torch.all(rate >= 0.0)
    assert torch.all((rho > 0.0) & (rho < 1.0))
    inactive = ~batch.active
    np.testing.assert_array_equal(power[:, 1:][inactive].detach().numpy(), 0.0)
    np.testing.assert_array_equal(rate[inactive].detach().numpy(), 0.0)


def test_precomputed_batch_preserves_per_state_snr_power():
    cfg, base = toy_batch(seed=19)
    powers = np.geomspace(10.0 ** 0.5, 10.0 ** 1.5, N)
    varied = precompute_batch(
        cfg, base.features.numpy(), batch_aoi, batch_h_true, batch_h_hat,
        base.active.numpy(), base.reference_power.numpy(), base.reference_rate.numpy(),
        base.reference_rho.numpy(), total_power=powers,
    )
    np.testing.assert_allclose(varied.total_power.numpy(), powers, rtol=1e-6)
    model = AllocatorNet(varied.features.shape[1], K, [32, 32])
    loss, metrics = allocator_losses(model, varied, cfg, LOSS_PRESETS["objective_only"])
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())


@pytest.mark.parametrize("preset", sorted(LOSS_PRESETS))
def test_every_loss_preset_is_finite_and_differentiable(preset):
    cfg, batch = toy_batch()
    model = AllocatorNet(batch.features.shape[1], K, [32, 32])
    loss, metrics = allocator_losses(model, batch, cfg, LOSS_PRESETS[preset])
    assert torch.isfinite(loss)
    for key, value in metrics.items():
        assert torch.isfinite(value), f"{preset}: {key} is not finite"
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0.0


def test_allocation_loss_ignores_inactive_dimensions():
    """Reference mass on an inactive user must not affect the loss."""
    predicted = torch.tensor([[0.5, 0.5, 0.0]])
    target = torch.tensor([[0.5, 0.5, 0.0]])
    mask = torch.tensor([[True, True, False]])
    base = simplex_cross_entropy(predicted, target, mask)
    polluted = simplex_cross_entropy(predicted, torch.tensor([[0.5, 0.5, 9.9]]), mask)
    assert base.item() == pytest.approx(polluted.item())


def test_objective_loss_prefers_the_better_allocation():
    """The objective term must reward a genuinely higher success probability."""
    cfg, batch = toy_batch()
    model = AllocatorNet(batch.features.shape[1], K, [32, 32])
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    weights = LOSS_PRESETS["objective_only"]
    first = allocator_losses(model, batch, cfg, weights)[1]["gain"].item()
    for _ in range(40):
        loss, _ = allocator_losses(model, batch, cfg, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    last = allocator_losses(model, batch, cfg, weights)[1]["gain"].item()
    assert last > first, "objective-aware training did not improve the objective"


def test_evaluation_reports_regret_and_zero_feasibility_error():
    cfg, batch = toy_batch()
    model = AllocatorNet(batch.features.shape[1], K, [32, 32])
    metrics = evaluate_allocator(model, batch, cfg)
    for key in ("expected_aoi_regret", "expected_aoi", "reference_expected_aoi",
                "power_l1", "rate_l1", "rho_mae", "threshold_crossing_rate"):
        assert np.isfinite(metrics[key])
    assert metrics["feasibility_error"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["inactive_leak"] == pytest.approx(0.0, abs=1e-12)
    assert 0.0 <= metrics["threshold_crossing_rate"] <= 1.0


def test_hybrid_ppo_actor_objective_is_finite_and_trains_gate_and_allocation():
    cfg, batch = toy_batch()
    model = HybridActorCritic(
        obs_dim=batch.features.shape[1], num_users=K, hidden_sizes=[32, 32],
        scheduling_enabled=False, allocation_control="joint",
        common_gate_enabled=True,
    )
    loss, metrics = hybrid_actor_objective_losses(
        model, batch, cfg, LOSS_PRESETS["objective_only"]
    )
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    loss.backward()
    for module in (model.power_head, model.rate_head, model.rho_r_head, model.common_gate_head):
        assert module is not None
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and sum(float(g.abs().sum()) for g in grads) > 0.0


@pytest.mark.parametrize("mode", ["power_only", "split_only", "joint"])
def test_hybrid_objective_supports_rsma_action_ablation_modes(mode):
    cfg, batch = toy_batch(seed=7)
    model = HybridActorCritic(
        obs_dim=batch.features.shape[1], num_users=K, hidden_sizes=[32, 32],
        scheduling_enabled=False, allocation_control=mode,
        common_gate_enabled=False,
    )
    loss, metrics = hybrid_actor_objective_losses(
        model, batch, cfg, LOSS_PRESETS["objective_only"]
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["expected_aoi"])
    loss.backward()
    learned_head = model.power_head if mode != "split_only" else model.rate_head
    grads = [p.grad for p in learned_head.parameters() if p.grad is not None]
    assert grads and sum(float(g.abs().sum()) for g in grads) > 0.0


def test_sdma_objective_has_exact_common_off_boundary_and_power_gradients():
    cfg, batch = toy_batch(seed=11)
    model = HybridActorCritic(
        obs_dim=batch.features.shape[1], num_users=K, hidden_sizes=[32, 32],
        scheduling_enabled=False, allocation_control="sdma_power",
        common_gate_enabled=False,
    )
    loss, metrics = sdma_actor_objective_losses(
        model, batch, cfg, LOSS_PRESETS["objective_only"]
    )
    assert torch.isfinite(loss)
    assert metrics["common_power"].item() == pytest.approx(0.0, abs=0.0)
    assert metrics["rho"].item() == pytest.approx(0.0, abs=0.0)
    assert metrics["gate_probability"].item() == pytest.approx(0.0, abs=0.0)
    loss.backward()
    power_grads = [p.grad for p in model.power_head.parameters() if p.grad is not None]
    assert power_grads and sum(float(g.abs().sum()) for g in power_grads) > 0.0
    for head in (model.rate_head, model.rho_r_head):
        assert all(p.grad is None for p in head.parameters())


def test_sdma_objective_rejects_non_sdma_actor_and_reference_action_loss():
    cfg, batch = toy_batch(seed=12)
    rsma = HybridActorCritic(
        obs_dim=batch.features.shape[1], num_users=K, hidden_sizes=[32, 32],
        scheduling_enabled=False, allocation_control="joint",
        common_gate_enabled=True,
    )
    with pytest.raises(ValueError, match="allocation_control"):
        sdma_actor_objective_losses(rsma, batch, cfg, LOSS_PRESETS["objective_only"])

    sdma = HybridActorCritic(
        obs_dim=batch.features.shape[1], num_users=K, hidden_sizes=[32, 32],
        scheduling_enabled=False, allocation_control="sdma_power",
        common_gate_enabled=False,
    )
    with pytest.raises(ValueError, match="objective_only"):
        sdma_actor_objective_losses(sdma, batch, cfg, LOSS_PRESETS["action_only"])


def test_reference_gain_matches_the_numpy_phy():
    """precompute_batch must score the reference action with the real PHY."""
    from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer

    cfg, batch = toy_batch(seed=1)
    phy = RSMAPhysicalLayer(cfg)
    for i in range(N):
        mask = batch.active[i].numpy()
        out = phy.evaluate_allocations(
            batch_h_true[i], mask,
            batch.reference_power[i].numpy()[None, :].astype(np.float64),
            batch.reference_rate[i].numpy()[None, :].astype(np.float64),
            np.asarray([batch.reference_rho[i].item()], dtype=np.float64),
            h_hat=batch_h_hat[i],
        )
        expected = float(out["success_prob"][0] @ batch.urgency[i].numpy())
        assert batch.reference_gain[i].item() == pytest.approx(expected, rel=1e-4, abs=1e-5)
    assert phy.num_users == K


def test_precompute_batch_keeps_posterior_samples_inside_each_state():
    cfg, batch = toy_batch(seed=4)
    posterior = np.stack([batch_h_true, batch_h_true], axis=1)
    expanded = precompute_batch(
        cfg, batch.features.numpy(), batch_aoi,
        posterior, batch_h_hat, batch.active.numpy(),
        batch.reference_power.numpy(), batch.reference_rate.numpy(),
        batch.reference_rho.numpy(),
    )
    assert len(expanded) == N
    assert expanded.gain_priv.shape[:2] == (N, 2)
    np.testing.assert_allclose(
        expanded.gain_priv[:, 0].numpy(), expanded.gain_priv[:, 1].numpy()
    )
    np.testing.assert_allclose(expanded.features.numpy(), batch.features.numpy())
    model = HybridActorCritic(
        obs_dim=batch.features.shape[1], num_users=K, hidden_sizes=[32, 32],
        scheduling_enabled=False, allocation_control="joint",
        common_gate_enabled=True,
    )
    base_loss, base_metrics = hybrid_actor_objective_losses(
        model, batch, cfg, LOSS_PRESETS["objective_only"]
    )
    repeated_loss, repeated_metrics = hybrid_actor_objective_losses(
        model, expanded, cfg, LOSS_PRESETS["objective_only"]
    )
    assert repeated_loss.item() == pytest.approx(base_loss.item(), abs=1e-6)
    assert repeated_metrics["gain"].item() == pytest.approx(
        base_metrics["gain"].item(), abs=1e-6
    )
