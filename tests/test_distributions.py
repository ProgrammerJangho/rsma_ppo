"""Action distributions: shapes, log-probability correctness, PPO ratio sanity."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.stats import beta as scipy_beta

from rsma_aoi.agents.distributions import (
    BernoulliSchedule, BetaDistribution, HybridPolicyDistribution, LatentAction,
    MaskedLogisticNormal, SquashedGaussianDistribution, make_scalar_distribution,
    masked_softmax,
)

B, K = 6, 3


def make_hybrid(seed: int = 0, scalar_dist: str = "beta", scheduling: bool = True):
    gen = torch.Generator().manual_seed(seed)
    dist = HybridPolicyDistribution(
        schedule_logits=torch.randn(B, K, generator=gen),
        power_mean=torch.randn(B, K + 1, generator=gen),
        power_log_std=torch.full((K + 1,), -0.5),
        rate_mean=torch.randn(B, K, generator=gen),
        rate_log_std=torch.full((K,), -0.5),
        rho_r_raw=torch.randn(B, 2, generator=gen),
        scalar_dist=scalar_dist,
        scheduling_enabled=scheduling,
    )
    return dist, gen


# --- masked softmax --------------------------------------------------------
def test_masked_softmax_is_a_simplex_over_the_mask():
    logits = torch.randn(B, K, generator=torch.Generator().manual_seed(0))
    mask = torch.tensor([[True, False, True]] * B)
    out = masked_softmax(logits, mask)
    np.testing.assert_allclose(out.sum(-1).numpy(), 1.0, atol=1e-6)
    np.testing.assert_array_equal(out[:, 1].numpy(), 0.0)
    assert torch.all(out >= 0.0)


def test_masked_softmax_handles_an_empty_mask():
    out = masked_softmax(torch.randn(2, 3), torch.zeros(2, 3, dtype=torch.bool))
    np.testing.assert_allclose(out.numpy(), 0.0)


def test_masked_softmax_is_numerically_stable_for_large_logits():
    out = masked_softmax(torch.tensor([[1e4, -1e4, 5e3]]), torch.ones(1, 3, dtype=torch.bool))
    assert torch.all(torch.isfinite(out))
    assert out.sum().item() == pytest.approx(1.0)


# --- masked logistic-normal ------------------------------------------------
def test_logistic_normal_shapes_and_masking():
    mask = torch.tensor([[True, True, False]] * B)
    dist = MaskedLogisticNormal(torch.zeros(B, K), torch.full((K,), -0.5), mask)
    gen = torch.Generator().manual_seed(1)
    z = dist.sample(gen)
    assert z.shape == (B, K)
    np.testing.assert_array_equal(z[:, 2].numpy(), 0.0)
    assert dist.log_prob(z).shape == (B,)
    assert dist.entropy().shape == (B,)
    assert torch.all(torch.isfinite(dist.log_prob(z)))
    assert torch.all(torch.isfinite(dist.entropy()))


def test_logistic_normal_log_prob_only_counts_unmasked_dimensions():
    mask = torch.tensor([[True, False]])
    dist = MaskedLogisticNormal(torch.zeros(1, 2), torch.zeros(2), mask)
    z = torch.tensor([[0.3, 99.0]])  # the masked coordinate must be irrelevant
    expected = torch.distributions.Normal(0.0, 1.0).log_prob(torch.tensor(0.3))
    assert dist.log_prob(z).item() == pytest.approx(expected.item(), abs=1e-6)
    entropy = torch.distributions.Normal(0.0, 1.0).entropy()
    assert dist.entropy().item() == pytest.approx(entropy.item(), abs=1e-6)


def test_logistic_normal_transform_produces_the_simplex():
    mask = torch.tensor([[True, True, False]] * B)
    dist = MaskedLogisticNormal(torch.randn(B, K), torch.full((K,), -0.5), mask)
    y = dist.transform(dist.sample(torch.Generator().manual_seed(2)))
    np.testing.assert_allclose(y.sum(-1).numpy(), 1.0, atol=1e-6)
    np.testing.assert_array_equal(y[:, 2].numpy(), 0.0)


# --- scalar heads ----------------------------------------------------------
def test_beta_log_prob_matches_scipy():
    raw = torch.tensor([[0.5, -0.2]])
    dist = BetaDistribution(raw)
    value = torch.tensor([[0.37]])
    expected = scipy_beta.logpdf(
        0.37, float(dist.alpha[0, 0]), float(dist.beta[0, 0])
    )
    assert dist.log_prob(value).item() == pytest.approx(expected, rel=1e-6)


def test_beta_samples_lie_strictly_inside_the_unit_interval():
    dist = BetaDistribution(torch.randn(256, 2, generator=torch.Generator().manual_seed(3)))
    samples = dist.sample(torch.Generator().manual_seed(4))
    assert torch.all(samples > 0.0) and torch.all(samples < 1.0)
    assert torch.all(torch.isfinite(dist.log_prob(samples)))
    assert torch.all(torch.isfinite(dist.entropy()))
    mode = dist.mode()
    assert torch.all(mode > 0.0) and torch.all(mode < 1.0)


def test_beta_sampling_is_generator_controlled():
    dist = BetaDistribution(torch.zeros(4, 2))
    a = dist.sample(torch.Generator().manual_seed(11))
    b = dist.sample(torch.Generator().manual_seed(11))
    c = dist.sample(torch.Generator().manual_seed(12))
    np.testing.assert_allclose(a.numpy(), b.numpy())
    assert not np.allclose(a.numpy(), c.numpy())


def test_beta_mean_matches_the_empirical_mean():
    raw = torch.tensor([[1.0, 0.0]])
    dist = BetaDistribution(raw)
    samples = dist.sample(torch.Generator().manual_seed(5)).numpy()
    expected = float(dist.alpha / (dist.alpha + dist.beta))
    big = BetaDistribution(raw.repeat(20_000, 1)).sample(torch.Generator().manual_seed(6))
    assert float(big.mean()) == pytest.approx(expected, abs=0.01)
    assert samples.shape == (1, 1)


def test_squashed_gaussian_includes_the_jacobian():
    """log p_Y(y) = log p_Z(z) - log(y (1 - y)) with y = sigmoid(z)."""
    raw = torch.tensor([[0.3, -0.7]])
    dist = SquashedGaussianDistribution(raw)
    z = torch.tensor([[0.42]])
    base = torch.distributions.Normal(dist.mean, dist.std).log_prob(z).sum()
    y = torch.sigmoid(z)
    expected = base - torch.log(y).sum() - torch.log(1 - y).sum()
    assert dist.log_prob(z).item() == pytest.approx(expected.item(), rel=1e-6)


def test_squashed_gaussian_density_integrates_to_one():
    dist = SquashedGaussianDistribution(torch.tensor([[0.2, -0.3]]))
    y = torch.linspace(1e-4, 1 - 1e-4, 20_001).unsqueeze(-1)
    z = torch.log(y / (1 - y))
    density = torch.exp(dist.log_prob(z))
    integral = torch.trapezoid(density, y.squeeze(-1))
    assert integral.item() == pytest.approx(1.0, abs=1e-3)


def test_squashed_gaussian_transform_and_mode():
    dist = SquashedGaussianDistribution(torch.tensor([[0.5, -1.0]]))
    z = dist.sample(torch.Generator().manual_seed(9))
    y = dist.transform(z)
    assert torch.all((y > 0.0) & (y < 1.0))
    assert dist.mode().item() == pytest.approx(0.5)


def test_scalar_distribution_factory():
    raw = torch.zeros(2, 2)
    assert isinstance(make_scalar_distribution(raw, "beta"), BetaDistribution)
    assert isinstance(
        make_scalar_distribution(raw, "squashed_gaussian"), SquashedGaussianDistribution
    )
    with pytest.raises(ValueError):
        make_scalar_distribution(raw, "bogus")


# --- scheduling head -------------------------------------------------------
def test_bernoulli_schedule_log_prob_and_entropy():
    logits = torch.tensor([[0.0, 2.0, -2.0]])
    dist = BernoulliSchedule(logits)
    sample = torch.tensor([[1.0, 0.0, 1.0]])
    expected = torch.distributions.Bernoulli(logits=logits).log_prob(sample).sum()
    assert dist.log_prob(sample).item() == pytest.approx(expected.item(), rel=1e-6)
    assert torch.isfinite(dist.entropy()).all()
    assert dist.mode().tolist() == [[0.0, 1.0, 0.0]]


def test_bernoulli_sampling_frequency_matches_the_probability():
    dist = BernoulliSchedule(torch.zeros(20_000, 1))
    samples = dist.sample(torch.Generator().manual_seed(10))
    assert float(samples.mean()) == pytest.approx(0.5, abs=0.02)


# --- composite -------------------------------------------------------------
@pytest.mark.parametrize("scalar_dist", ["beta", "squashed_gaussian"])
def test_hybrid_sample_shapes_and_finiteness(scalar_dist):
    dist, gen = make_hybrid(scalar_dist=scalar_dist)
    latent = dist.sample(gen)
    assert latent.schedule_raw.shape == (B, K)
    assert latent.active_mask.shape == (B, K)
    assert latent.power_latent.shape == (B, K + 1)
    assert latent.rate_latent.shape == (B, K)
    assert latent.rho_r_latent.shape == (B, 1)
    log_prob = dist.log_prob(latent)
    entropy = dist.entropy(latent.active_mask)
    assert log_prob.shape == (B,) and entropy.shape == (B,)
    assert torch.all(torch.isfinite(log_prob)) and torch.all(torch.isfinite(entropy))
    assert torch.all(latent.active_mask.any(dim=-1))


def test_ppo_ratio_is_one_for_identical_parameters():
    dist_old, gen = make_hybrid(seed=3)
    latent = dist_old.sample(gen)
    old_log_prob = dist_old.log_prob(latent)
    dist_new, _ = make_hybrid(seed=3)  # same parameters -> identical distribution
    ratio = torch.exp(dist_new.log_prob(latent) - old_log_prob)
    np.testing.assert_allclose(ratio.numpy(), 1.0, atol=1e-6)


def test_ppo_ratio_moves_when_the_parameters_change():
    dist_old, gen = make_hybrid(seed=4)
    latent = dist_old.sample(gen)
    dist_new, _ = make_hybrid(seed=5)
    ratio = torch.exp(dist_new.log_prob(latent) - dist_old.log_prob(latent))
    assert torch.all(ratio > 0.0)
    assert not np.allclose(ratio.numpy(), 1.0)


def test_log_prob_is_invariant_to_the_masked_latent_entries():
    """Re-scoring must depend only on the stored mask and unmasked latents."""
    dist, gen = make_hybrid(seed=6)
    latent = dist.sample(gen)
    perturbed = LatentAction(
        schedule_raw=latent.schedule_raw,
        active_mask=latent.active_mask,
        power_latent=latent.power_latent
        + 100.0 * (~torch.cat([torch.ones(B, 1, dtype=torch.bool), latent.active_mask], -1)),
        rate_latent=latent.rate_latent + 100.0 * (~latent.active_mask),
        rho_r_latent=latent.rho_r_latent,
    )
    np.testing.assert_allclose(
        dist.log_prob(latent).numpy(), dist.log_prob(perturbed).numpy(), atol=1e-6
    )


def test_env_action_is_a_feasible_simplex():
    dist, gen = make_hybrid(seed=7)
    latent = dist.sample(gen)
    action = dist.to_env_action(latent)
    power = action["power_fractions"]
    rate = action["rate_fractions"]
    np.testing.assert_allclose(power.sum(-1).numpy(), 1.0, atol=1e-6)
    np.testing.assert_allclose(rate.sum(-1).numpy(), 1.0, atol=1e-6)
    inactive = ~latent.active_mask
    np.testing.assert_array_equal(power[:, 1:][inactive].numpy(), 0.0)
    np.testing.assert_array_equal(rate[inactive].numpy(), 0.0)


def test_mode_is_deterministic_and_feasible():
    dist, _ = make_hybrid(seed=8)
    a = dist.to_env_action(dist.mode())
    b = dist.to_env_action(dist.mode())
    np.testing.assert_allclose(a["power_fractions"].numpy(), b["power_fractions"].numpy())
    np.testing.assert_allclose(a["power_fractions"].sum(-1).numpy(), 1.0, atol=1e-6)


def test_scheduling_disabled_excludes_the_bernoulli_term():
    with_sched, gen = make_hybrid(seed=9, scheduling=True)
    latent = with_sched.sample(gen)
    without_sched, _ = make_hybrid(seed=9, scheduling=False)
    latent_all = LatentAction(
        schedule_raw=torch.ones_like(latent.schedule_raw),
        active_mask=torch.ones_like(latent.active_mask),
        power_latent=latent.power_latent,
        rate_latent=latent.rate_latent,
        rho_r_latent=latent.rho_r_latent,
    )
    assert torch.all(torch.isfinite(without_sched.log_prob(latent_all)))
    sampled = without_sched.sample(gen)
    assert torch.all(sampled.active_mask)


def test_invalid_shapes_raise():
    with pytest.raises(ValueError):
        MaskedLogisticNormal(torch.zeros(3), torch.zeros(3), torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError):
        BetaDistribution(torch.zeros(2, 3))
    with pytest.raises(ValueError):
        BernoulliSchedule(torch.zeros(3))


# --- subset-categorical scheduling head ------------------------------------
def test_enumerate_subsets_covers_every_non_empty_subset():
    from rsma_aoi.agents.distributions import enumerate_subsets

    for k in (1, 2, 3, 4):
        table = enumerate_subsets(k)
        assert table.shape == (2**k - 1, k)
        assert torch.all(table.any(dim=-1)), "the empty subset must not appear"
        codes = {tuple(row.tolist()) for row in table}
        assert len(codes) == 2**k - 1, "subsets must be unique"
        assert tuple([True] * k) in codes
    with pytest.raises(ValueError):
        enumerate_subsets(0)


def test_subset_categorical_needs_no_repair():
    """Every sampled category is already a feasible (non-empty) schedule."""
    from rsma_aoi.agents.distributions import SubsetCategorical

    dist = SubsetCategorical(torch.randn(64, 15, generator=torch.Generator().manual_seed(0)), 4)
    idx = dist.sample(torch.Generator().manual_seed(1))
    masks = dist.decode(idx)
    assert idx.shape == (64, 1)
    assert masks.shape == (64, 4)
    assert torch.all(masks.any(dim=-1))
    assert torch.all(torch.isfinite(dist.log_prob(idx)))
    assert torch.all(dist.entropy() > 0)


def test_subset_categorical_log_prob_matches_torch():
    from rsma_aoi.agents.distributions import SubsetCategorical

    logits = torch.tensor([[0.5, -1.0, 2.0]])
    dist = SubsetCategorical(logits, 2)
    reference = torch.distributions.Categorical(logits=logits)
    for index in range(3):
        value = torch.tensor([[float(index)]])
        assert dist.log_prob(value).item() == pytest.approx(
            reference.log_prob(torch.tensor([index])).item(), rel=1e-6
        )
    assert dist.entropy().item() == pytest.approx(reference.entropy().item(), rel=1e-6)
    assert dist.mode().item() == 2.0


def test_subset_categorical_sampling_frequencies_match_the_probabilities():
    from rsma_aoi.agents.distributions import SubsetCategorical

    logits = torch.tensor([[0.0, 1.0, -1.0]]).repeat(20_000, 1)
    dist = SubsetCategorical(logits, 2)
    idx = dist.sample(torch.Generator().manual_seed(3)).reshape(-1).long()
    empirical = torch.bincount(idx, minlength=3).float() / idx.numel()
    expected = torch.softmax(torch.tensor([0.0, 1.0, -1.0]), dim=0)
    torch.testing.assert_close(empirical, expected, atol=0.02, rtol=0.05)


def test_subset_categorical_shape_validation():
    from rsma_aoi.agents.distributions import SubsetCategorical

    with pytest.raises(ValueError):
        SubsetCategorical(torch.zeros(2, 4), 3)   # 3 users need 7 logits
    with pytest.raises(ValueError):
        SubsetCategorical(torch.zeros(7), 3)


def test_hybrid_distribution_with_the_categorical_head():
    gen = torch.Generator().manual_seed(4)
    dist = HybridPolicyDistribution(
        schedule_logits=torch.randn(B, 2**K - 1, generator=gen),
        power_mean=torch.randn(B, K + 1, generator=gen),
        power_log_std=torch.full((K + 1,), -0.5),
        rate_mean=torch.randn(B, K, generator=gen),
        rate_log_std=torch.full((K,), -0.5),
        rho_r_raw=torch.randn(B, 2, generator=gen),
        scheduling_head="subset_categorical",
        num_users=K,
    )
    latent = dist.sample(gen)
    assert latent.schedule_raw.shape == (B, 1)
    assert latent.active_mask.shape == (B, K)
    assert torch.all(latent.active_mask.any(dim=-1))

    # the PPO ratio must still be exactly 1 under identical parameters
    ratio = torch.exp(dist.log_prob(latent) - dist.log_prob(latent))
    np.testing.assert_allclose(ratio.numpy(), 1.0, atol=1e-6)
    action = dist.to_env_action(latent)
    np.testing.assert_allclose(action["power_fractions"].sum(-1).numpy(), 1.0, atol=1e-6)
    inactive = ~latent.active_mask
    np.testing.assert_array_equal(action["rate_fractions"][inactive].numpy(), 0.0)
    assert torch.all(torch.isfinite(dist.entropy(latent.active_mask)))


def test_categorical_head_scheduling_disabled_selects_everyone():
    gen = torch.Generator().manual_seed(5)
    dist = HybridPolicyDistribution(
        schedule_logits=torch.randn(B, 2**K - 1, generator=gen),
        power_mean=torch.zeros(B, K + 1), power_log_std=torch.full((K + 1,), -0.5),
        rate_mean=torch.zeros(B, K), rate_log_std=torch.full((K,), -0.5),
        rho_r_raw=torch.zeros(B, 2),
        scheduling_head="subset_categorical", num_users=K, scheduling_enabled=False,
    )
    assert torch.all(dist.sample(gen).active_mask)
    assert torch.all(dist.mode().active_mask)


def test_unknown_scheduling_head_raises():
    with pytest.raises(ValueError):
        HybridPolicyDistribution(
            schedule_logits=torch.zeros(2, 3), power_mean=torch.zeros(2, 4),
            power_log_std=torch.zeros(4), rate_mean=torch.zeros(2, 3),
            rate_log_std=torch.zeros(3), rho_r_raw=torch.zeros(2, 2),
            scheduling_head="bogus", num_users=3,
        )
