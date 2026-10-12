"""Heterogeneous path loss: geometry, normalisation, CSIT error modes, weights."""

from __future__ import annotations

import numpy as np
import pytest

from rsma_aoi.envs.channel import CSIT_ERROR_MODES, ChannelModel
from rsma_aoi.envs.geometry import (
    GeometrySpec, aoi_weights_for, build_geometry, class_labels, near_far_gains_db,
)

from rsma_aoi.utils.config import ConfigError, EnvConfig


# --- path-loss models ------------------------------------------------------
def test_homogeneous_reproduces_unit_gains():
    g = build_geometry(GeometrySpec(num_users=4, model="homogeneous"))
    np.testing.assert_allclose(g.beta, 1.0)
    assert g.disparity_db() == pytest.approx(0.0)


def test_explicit_near_far_gains():
    g = build_geometry(GeometrySpec(num_users=4, model="explicit_near_far",
                                    path_gains_db=[0.0, 0.0, -10.0, -10.0],
                                    normalization="none"))
    np.testing.assert_allclose(g.beta, [1.0, 1.0, 0.1, 0.1])
    assert g.disparity_db() == pytest.approx(10.0)


def test_distance_pathloss_decays_with_distance():
    g = build_geometry(GeometrySpec(
        num_users=3, model="distance_pathloss", distances=[50.0, 100.0, 200.0],
        pathloss_exponent=3.5, normalization="none",
    ))
    assert g.beta[0] > g.beta[1] > g.beta[2]
    # doubling the distance costs 10 * eta * log10(2) dB
    expected = 10.0 * 3.5 * np.log10(2.0)
    assert 10 * np.log10(g.beta[0] / g.beta[1]) == pytest.approx(expected, rel=1e-9)


def test_random_geometry_is_seed_reproducible_and_in_range():
    spec = GeometrySpec(num_users=8, model="distance_pathloss", cell_radius=400.0,
                        min_distance=40.0, shadowing_std_db=6.0, geometry_seed=5)
    a, b = build_geometry(spec), build_geometry(spec)
    np.testing.assert_allclose(a.beta, b.beta)
    np.testing.assert_allclose(a.distances, b.distances)
    assert np.all(a.distances >= 40.0) and np.all(a.distances <= 400.0)

    other = build_geometry(GeometrySpec(**{**spec.__dict__, "geometry_seed": 6}))
    assert not np.allclose(a.beta, other.beta)


@pytest.mark.parametrize("mode,check", [
    ("none", lambda b: True),
    ("mean_one", lambda b: np.isclose(b.mean(), 1.0)),
    ("cell_edge_reference", lambda b: np.isclose(b.min(), 1.0)),
])
def test_normalisation_modes(mode, check):
    g = build_geometry(GeometrySpec(num_users=4, model="explicit_near_far",
                                    path_gains_db=[0.0, -5.0, -10.0, -15.0],
                                    normalization=mode))
    assert check(g.beta)
    # normalisation must never change the *relative* disparity
    assert g.disparity_db() == pytest.approx(15.0)


def test_average_snr_reflects_beta_and_power():
    g = build_geometry(GeometrySpec(num_users=2, model="explicit_near_far",
                                    path_gains_db=[0.0, -10.0], normalization="none"))
    snr = g.average_snr_db(total_power=10.0, noise_power=1.0)
    np.testing.assert_allclose(snr, [10.0, 0.0], atol=1e-9)


def test_near_far_helpers_split_users_evenly():
    assert near_far_gains_db(4, 10.0, 2) == [0.0, 0.0, -10.0, -10.0]
    assert class_labels(4, 2) == ["near", "near", "far", "far"]
    assert near_far_gains_db(6, 15.0, 3) == [0.0, 0.0, -7.5, -7.5, -15.0, -15.0]
    assert class_labels(6, 3) == ["near", "near", "middle", "middle", "far", "far"]
    assert near_far_gains_db(8, 20.0, 3)[0] == 0.0
    assert len(near_far_gains_db(8, 20.0, 3)) == 8


def test_invalid_geometry_specs_raise():
    for kwargs in ({"model": "bogus"}, {"normalization": "bogus"},
                   {"model": "explicit_near_far"}, {"cell_radius": 0.0},
                   {"min_distance": 600.0}, {"pathloss_exponent": 0.0},
                   {"shadowing_std_db": -1.0}, {"path_gains_db": [0.0]}):
        with pytest.raises(ValueError):
            build_geometry(GeometrySpec(num_users=4, **kwargs))


# --- CSIT error modes ------------------------------------------------------
def _channel(mode: str, sigma: float, beta):
    return ChannelModel(num_users=len(beta), num_antennas=4, alpha=0.0,
                        csit_error_std=sigma, path_gain=np.asarray(beta),
                        csit_error_mode=mode)


@pytest.mark.parametrize("mode", CSIT_ERROR_MODES)
def test_perfect_csit_is_identical_in_every_mode(mode):
    ch = _channel(mode, 0.0, [1.0, 0.1])
    rng = np.random.default_rng(0)
    h = ch.sample_initial(rng)
    np.testing.assert_array_equal(ch.estimate(h, rng), h)


def test_relative_error_preserves_per_user_power():
    beta = [1.0, 0.1]
    ch = _channel("relative", 0.5, beta)
    rng = np.random.default_rng(0)
    power = np.zeros(2)
    trials = 4000
    for _ in range(trials):
        h = ch.sample_initial(rng)
        power += np.mean(np.abs(ch.estimate(h, rng)) ** 2, axis=1)
    np.testing.assert_allclose(power / trials, beta, rtol=0.08)


def test_absolute_error_inflates_the_weak_user_estimate():
    """The documented artefact: a far user looks stronger than it is."""
    beta = [1.0, 0.01]
    sigma = 0.5
    ch = _channel("absolute", sigma, beta)
    rng = np.random.default_rng(1)
    power = np.zeros(2)
    trials = 4000
    for _ in range(trials):
        h = ch.sample_initial(rng)
        power += np.mean(np.abs(ch.estimate(h, rng)) ** 2, axis=1)
    power /= trials
    expected = (1 - sigma**2) * np.asarray(beta) + sigma**2 * np.mean(beta)
    np.testing.assert_allclose(power, expected, rtol=0.08)
    assert power[1] > 10 * beta[1], "the artefact this mode exists to expose is absent"


def test_normalized_error_removes_the_inflation_but_keeps_relative_degradation():
    beta = [1.0, 0.01]
    sigma = 0.5
    ch = _channel("normalized", sigma, beta)
    rng = np.random.default_rng(2)
    power = np.zeros(2)
    correlation = np.zeros(2)
    trials = 4000
    for _ in range(trials):
        h = ch.sample_initial(rng)
        est = ch.estimate(h, rng)
        power += np.mean(np.abs(est) ** 2, axis=1)
        for k in range(2):
            denom = np.linalg.norm(h[k]) * np.linalg.norm(est[k])
            correlation[k] += abs(np.vdot(h[k], est[k])) / max(denom, 1e-30)
    np.testing.assert_allclose(power / trials, beta, rtol=0.08)
    # the weak user is still relatively worse estimated
    assert (correlation / trials)[1] < (correlation / trials)[0]


def test_unknown_csit_mode_raises():
    with pytest.raises(ValueError):
        _channel("bogus", 0.3, [1.0, 1.0])


# --- config integration ----------------------------------------------------
def test_config_resolves_geometry_and_weights():
    cfg = EnvConfig(num_users=4, num_antennas=2, pathloss_model="explicit_near_far",
                    path_gains_db=near_far_gains_db(4, 10.0, 2),
                    user_classes=class_labels(4, 2), aoi_weight_mode="inverse_snr")
    beta = np.asarray(cfg.path_gain_linear)
    assert beta[0] > beta[-1]
    weights = np.asarray(cfg.weight_vector)
    assert weights[-1] > weights[0], "inverse-SNR weighting must favour far users"
    assert weights.mean() == pytest.approx(1.0)
    assert cfg.class_vector == ["near", "near", "far", "far"]


def test_homogeneous_config_is_unchanged_by_the_new_fields():
    """The default path must still give unit gains and equal weights."""
    cfg = EnvConfig(num_users=4, num_antennas=2)
    np.testing.assert_allclose(cfg.path_gain_linear, 1.0)
    np.testing.assert_allclose(cfg.weight_vector, 1.0)
    assert cfg.class_vector == ["all"] * 4


def test_legacy_path_gains_db_still_works_without_a_model():
    cfg = EnvConfig(num_users=2, path_gains_db=[0.0, -10.0])
    np.testing.assert_allclose(cfg.path_gain_linear, [1.0, 0.1])


def test_invalid_heterogeneous_configs_raise():
    for kwargs in (
        {"pathloss_model": "explicit_near_far"},
        {"csit_error_mode": "bogus"},
        {"beta_normalization": "bogus"},
        {"aoi_weight_mode": "class_priority"},
        {"user_classes": ["near"]},
        {"user_distances": [1.0]},
        {"min_distance": 1e6},
        {"shadowing_std_db": -1.0},
    ):
        with pytest.raises(ConfigError):
            EnvConfig(num_users=4, **kwargs)


def test_beta_reaches_the_environment_channel():
    """Near users must actually see larger channel norms in the environment."""
    from rsma_aoi.envs.aoi_env import AoIRSMAEnv

    cfg = EnvConfig(num_users=4, num_antennas=2, pathloss_model="explicit_near_far",
                    path_gains_db=near_far_gains_db(4, 20.0, 2),
                    user_classes=class_labels(4, 2), episode_length=50)
    env = AoIRSMAEnv(cfg)
    env.reset(seed=0)
    norms = np.zeros(4)
    for _ in range(200):
        norms += np.sum(np.abs(env.true_channel) ** 2, axis=1)
        env.step(env.action_space.sample())
    norms /= 200
    assert norms[:2].mean() > 50 * norms[2:].mean()
    env.close()


def test_aoi_weight_modes():
    g = build_geometry(GeometrySpec(num_users=4, model="explicit_near_far",
                                    path_gains_db=[0.0, 0.0, -12.0, -12.0],
                                    normalization="mean_one"))
    classes = class_labels(4, 2)
    equal = aoi_weights_for("equal", g)
    np.testing.assert_allclose(equal, 1.0)

    inverse = aoi_weights_for("inverse_snr", g)
    assert inverse[-1] > inverse[0], "inverse-SNR must favour the far users"
    assert inverse.mean() == pytest.approx(1.0)

    priority = aoi_weights_for("class_priority", g, class_priority={"near": 1.0, "far": 3.0},
                               classes=classes)
    assert priority[-1] == pytest.approx(3.0 * priority[0])
    assert priority.mean() == pytest.approx(1.0)

    explicit = aoi_weights_for("explicit", g, explicit=[1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(explicit, [1.0, 2.0, 3.0, 4.0])

    for bad in ("bogus",):
        with pytest.raises(ValueError):
            aoi_weights_for(bad, g)
    with pytest.raises(ValueError):
        aoi_weights_for("explicit", g)
    with pytest.raises(ValueError):
        aoi_weights_for("class_priority", g)
