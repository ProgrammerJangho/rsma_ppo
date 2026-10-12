"""Channel model: reproducibility, temporal correlation, CSIT error."""

from __future__ import annotations

import numpy as np
import pytest

from rsma_aoi.envs.channel import ChannelModel
from rsma_aoi.utils.config import EnvConfig


def make_channel(**kwargs) -> ChannelModel:
    cfg = EnvConfig(**{"num_users": 3, "num_antennas": 4, **kwargs})
    return ChannelModel.from_config(cfg)


def test_shape_and_dtype_are_complex():
    ch = make_channel()
    h = ch.sample_initial(np.random.default_rng(0))
    assert h.shape == (3, 4)
    assert np.iscomplexobj(h)
    assert np.all(np.isfinite(h))


def test_same_seed_reproduces_the_same_channel():
    ch = make_channel(alpha=0.7)
    a = ch.sample_initial(np.random.default_rng(123))
    b = ch.sample_initial(np.random.default_rng(123))
    np.testing.assert_allclose(a, b)

    rng_a, rng_b = np.random.default_rng(5), np.random.default_rng(5)
    ha = hb = a
    for _ in range(5):
        ha = ch.evolve(ha, rng_a)
        hb = ch.evolve(hb, rng_b)
    np.testing.assert_allclose(ha, hb)


def test_rayleigh_path_is_bit_identical_to_the_legacy_formulas():
    """Adding Rician support must not move any historical Rayleigh trajectory."""
    ch = make_channel(alpha=0.8, csit_error_std=0.3, csit_error_mode="relative")
    seed = 12345
    new_rng = np.random.default_rng(seed)
    old_rng = np.random.default_rng(seed)

    h_new = ch.sample_initial(new_rng)
    real = old_rng.standard_normal(ch.shape)
    imag = old_rng.standard_normal(ch.shape)
    h_old = np.sqrt(ch.path_gain[:, None] / 2.0) * (real + 1j * imag)
    np.testing.assert_array_equal(h_new, h_old)

    evolved_new = ch.evolve(h_new, new_rng)
    real = old_rng.standard_normal(ch.shape)
    imag = old_rng.standard_normal(ch.shape)
    innovation = np.sqrt(ch.path_gain[:, None] / 2.0) * (real + 1j * imag)
    evolved_old = ch.alpha * h_old + np.sqrt(1.0 - ch.alpha**2) * innovation
    np.testing.assert_array_equal(evolved_new, evolved_old)

    estimate_new = ch.estimate(evolved_new, new_rng)
    real = old_rng.standard_normal(ch.shape)
    imag = old_rng.standard_normal(ch.shape)
    error = np.sqrt(ch.path_gain[:, None] / 2.0) * (real + 1j * imag)
    estimate_old = np.sqrt(1.0 - ch.csit_error_std**2) * evolved_old + ch.csit_error_std * error
    np.testing.assert_array_equal(estimate_new, estimate_old)



def test_different_seeds_give_different_channels():
    ch = make_channel()
    a = ch.sample_initial(np.random.default_rng(0))
    b = ch.sample_initial(np.random.default_rng(1))
    assert not np.allclose(a, b)


def test_alpha_zero_is_independent_block_fading():
    ch = make_channel(alpha=0.0)
    rng = np.random.default_rng(0)
    h0 = ch.sample_initial(rng)
    samples = np.stack([ch.evolve(h0, rng).ravel() for _ in range(3000)])
    # the successor must be uncorrelated with the (fixed) predecessor
    corr = np.abs(np.vdot(h0.ravel(), samples.mean(axis=0))) / np.linalg.norm(h0)
    assert corr < 0.2


def test_alpha_positive_keeps_temporal_correlation():
    alpha = 0.9
    ch = make_channel(alpha=alpha)
    rng = np.random.default_rng(1)
    h0 = ch.sample_initial(rng)
    successors = np.stack([ch.evolve(h0, rng).ravel() for _ in range(4000)])
    # E[h(t+1)] = alpha * h(t)
    np.testing.assert_allclose(successors.mean(axis=0), alpha * h0.ravel(), atol=0.06)


def test_stationary_variance_matches_path_gain():
    ch = make_channel(alpha=0.8, path_gains_db=[0.0, -3.0, 3.0])
    rng = np.random.default_rng(2)
    h = ch.sample_initial(rng)
    power = np.zeros(3)
    trials = 4000
    for _ in range(trials):
        h = ch.evolve(h, rng)
        power += np.mean(np.abs(h) ** 2, axis=1)
    power /= trials
    np.testing.assert_allclose(power, ch.path_gain, rtol=0.1)


def test_perfect_csit_returns_the_true_channel():
    ch = make_channel(csit_error_std=0.0)
    rng = np.random.default_rng(3)
    h = ch.sample_initial(rng)
    np.testing.assert_array_equal(ch.estimate(h, rng), h)


def test_csit_error_preserves_average_power():
    ch = make_channel(csit_error_std=0.4)
    rng = np.random.default_rng(4)
    true_power, est_power = 0.0, 0.0
    trials = 3000
    for _ in range(trials):
        h = ch.sample_initial(rng)
        true_power += float(np.mean(np.abs(h) ** 2))
        est_power += float(np.mean(np.abs(ch.estimate(h, rng)) ** 2))
    assert abs(true_power - est_power) / trials < 0.05


def test_invalid_shapes_raise():
    ch = make_channel()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        ch.evolve(np.zeros((2, 2), dtype=complex), rng)
    with pytest.raises(ValueError):
        ch.estimate(np.zeros((2, 2), dtype=complex), rng)


def test_rician_mean_and_total_power_match_the_configured_beta():
    ch = make_channel(
        small_scale_model="rician_ula", rician_k_db=8.0,
        user_angles_deg=[-10.0, 0.0, 10.0],
        path_gains_db=[0.0, -3.0, 3.0],
    )
    rng = np.random.default_rng(20)
    samples = np.stack([ch.sample_initial(rng) for _ in range(6000)])
    np.testing.assert_allclose(samples.mean(axis=0), ch.mean, atol=0.035)
    power = np.mean(np.abs(samples) ** 2, axis=(0, 2))
    np.testing.assert_allclose(power, ch.path_gain, rtol=0.04)


def test_rician_zero_k_limit_is_rayleigh_in_moments():
    # -100 dB is the supported numerical representation of kappa -> 0.
    ch = make_channel(
        small_scale_model="rician_ula", rician_k_db=-100.0,
        user_angles_deg=[-20.0, 0.0, 20.0],
    )
    rng = np.random.default_rng(24)
    samples = np.stack([ch.sample_initial(rng) for _ in range(6000)])
    assert np.max(np.abs(samples.mean(axis=0))) < 0.04
    power = np.mean(np.abs(samples) ** 2, axis=(0, 2))
    np.testing.assert_allclose(power, ch.path_gain, rtol=0.04)


def test_rician_centered_temporal_correlation_is_alpha():
    alpha = 0.8
    ch = make_channel(
        alpha=alpha, small_scale_model="rician_ula", rician_k_db=8.0,
        user_angles_deg=[-10.0, 0.0, 10.0],
    )
    rng = np.random.default_rng(21)
    h0 = ch.sample_initial(rng)
    successors = np.stack([ch.evolve(h0, rng) for _ in range(5000)])
    expected = ch.mean + alpha * (h0 - ch.mean)
    np.testing.assert_allclose(successors.mean(axis=0), expected, atol=0.025)


def test_ula_mean_correlation_matches_the_steering_formula():
    separation = 20.0
    ch = EnvConfig(
        num_users=2, num_antennas=2, small_scale_model="rician_ula",
        rician_k_db=8.0, user_angles_deg=[-separation / 2, separation / 2],
        antenna_spacing_wavelength=0.5,
    )
    model = ChannelModel.from_config(ch)
    mean = model.mean
    observed = abs(np.vdot(mean[0], mean[1])) ** 2 / (
        np.vdot(mean[0], mean[0]).real * np.vdot(mean[1], mean[1]).real
    )
    delta_u = np.sin(np.deg2rad(separation / 2)) - np.sin(np.deg2rad(-separation / 2))
    expected = np.cos(np.pi * delta_u / 2.0) ** 2
    assert observed == pytest.approx(expected, abs=1e-12)


def test_episode_geometry_sampling_is_seeded_and_covers_all_sets():
    cfg = EnvConfig(
        num_users=2,
        num_antennas=2,
        small_scale_model="rician_ula",
        user_angles_deg=[-10.0, 10.0],
        user_angle_sets_deg=[[-2.5, 2.5], [-10.0, 10.0], [-30.0, 30.0]],
    )
    first = ChannelModel.from_config(cfg)
    second = ChannelModel.from_config(cfg)
    rng_a, rng_b = np.random.default_rng(71), np.random.default_rng(71)
    choices_a, choices_b = [], []
    for _ in range(20):
        choices_a.append(first.resample_episode_geometry(rng_a))
        choices_b.append(second.resample_episode_geometry(rng_b))
        np.testing.assert_array_equal(first.user_angles_deg, second.user_angles_deg)
    assert choices_a == choices_b
    assert set(choices_a) == {0, 1, 2}


def test_fixed_geometry_resampling_preserves_rng_stream():
    cfg = EnvConfig(
        num_users=2, num_antennas=2, small_scale_model="rician_ula",
        user_angles_deg=[-10.0, 10.0],
    )
    channel = ChannelModel.from_config(cfg)
    rng_a, rng_b = np.random.default_rng(73), np.random.default_rng(73)
    assert channel.resample_episode_geometry(rng_a) is None
    assert rng_a.random() == rng_b.random()


def test_rician_relative_csit_preserves_mean_and_marginal_power():
    ch = make_channel(
        small_scale_model="rician_ula", rician_k_db=8.0,
        user_angles_deg=[-15.0, 0.0, 15.0], csit_error_std=0.3,
    )
    rng = np.random.default_rng(22)
    true = np.stack([ch.sample_initial(rng) for _ in range(5000)])
    estimate = np.stack([ch.estimate(h, rng) for h in true])
    np.testing.assert_allclose(estimate.mean(axis=0), ch.mean, atol=0.025)
    np.testing.assert_allclose(
        np.mean(np.abs(estimate) ** 2, axis=(0, 2)), ch.path_gain, rtol=0.04
    )


def test_rician_perfect_csit_is_exact():
    ch = make_channel(
        small_scale_model="rician_ula", rician_k_db=8.0,
        user_angles_deg=[-10.0, 0.0, 10.0], csit_error_std=0.0,
    )
    rng = np.random.default_rng(23)
    h = ch.sample_initial(rng)
    np.testing.assert_array_equal(ch.estimate(h, rng), h)


def test_relative_csit_posterior_sampler_has_conditional_moments():
    ch = make_channel(
        small_scale_model="rician_ula", rician_k_db=8.0,
        user_angles_deg=[-15.0, 0.0, 15.0], csit_error_std=0.3,
    )
    rng = np.random.default_rng(31)
    h = ch.sample_initial(rng)
    h_hat = ch.estimate(h, rng)
    samples = ch.sample_true_given_estimate(h_hat, rng, 12000)
    a = np.sqrt(1.0 - ch.csit_error_std**2)
    expected_mean = ch.mean + a * (h_hat - ch.mean)
    np.testing.assert_allclose(samples.mean(axis=0), expected_mean, atol=0.006)
    centered_power = np.mean(np.abs(samples - expected_mean) ** 2, axis=(0, 2))
    np.testing.assert_allclose(
        centered_power, ch.csit_error_std**2 * ch.scatter_variance, rtol=0.05
    )


def test_posterior_sampler_rejects_unsupported_mode_and_bad_count():
    ch = make_channel(csit_error_std=0.2, csit_error_mode="absolute")
    h_hat = np.zeros(ch.shape, dtype=complex)
    with pytest.raises(NotImplementedError):
        ch.sample_true_given_estimate(h_hat, np.random.default_rng(0), 2)
    relative = make_channel(csit_error_std=0.2)
    with pytest.raises(ValueError):
        relative.sample_true_given_estimate(h_hat, np.random.default_rng(0), 0)
