"""Beam directions: unit norm, shapes, ZF nulling and ill-conditioned inputs."""

from __future__ import annotations

import numpy as np
import pytest

from rsma_aoi.envs.beamforming import (
    common_direction, default_rzf_regularization, mrt_directions, private_directions,
    maxmin_reweighted_direction, rzf_directions, zf_directions,
)


@pytest.fixture
def channel() -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4))) / np.sqrt(2)


def test_all_active_directions_are_unit_norm(channel):
    active = np.array([True, True, False])
    for beams in (
        mrt_directions(channel, active),
        zf_directions(channel, active),
        rzf_directions(channel, active, reg=0.1),
    ):
        assert beams.shape == channel.shape
        norms = np.linalg.norm(beams, axis=1)
        np.testing.assert_allclose(norms[active], 1.0, atol=1e-10)
        np.testing.assert_allclose(norms[~active], 0.0, atol=1e-12)


def test_mrt_aligns_with_the_channel(channel):
    active = np.ones(3, dtype=bool)
    beams = mrt_directions(channel, active)
    for k in range(3):
        gain = abs(np.vdot(channel[k], beams[k])) ** 2
        np.testing.assert_allclose(gain, np.sum(np.abs(channel[k]) ** 2), rtol=1e-10)


def test_zf_nulls_the_interference(channel):
    active = np.ones(3, dtype=bool)
    beams = zf_directions(channel, active)
    cross = np.abs(np.conj(channel) @ beams.T) ** 2
    off_diagonal = cross - np.diag(np.diag(cross))
    assert np.max(off_diagonal) < 1e-18


def test_zf_only_nulls_towards_active_users(channel):
    active = np.array([True, True, False])
    beams = zf_directions(channel, active)
    cross = np.abs(np.conj(channel[:2]) @ beams[:2].T) ** 2
    assert abs(cross[0, 1]) < 1e-18 and abs(cross[1, 0]) < 1e-18


def test_rzf_stays_finite_for_identical_channels():
    h = np.ones((2, 2), dtype=complex)  # perfectly correlated users
    active = np.ones(2, dtype=bool)
    beams = rzf_directions(h, active, reg=1e-3)
    assert np.all(np.isfinite(beams))
    np.testing.assert_allclose(np.linalg.norm(beams, axis=1), 1.0, atol=1e-10)


def test_zf_stays_finite_for_rank_deficient_and_overloaded_channels():
    h = np.array([[1.0 + 0j, 0.0], [2.0 + 0j, 0.0], [1e-14 + 0j, 0.0]])
    active = np.ones(3, dtype=bool)
    beams = zf_directions(h, active)
    assert np.all(np.isfinite(beams))
    np.testing.assert_allclose(np.linalg.norm(beams, axis=1), 1.0, atol=1e-10)


def test_zero_channel_falls_back_to_a_valid_direction():
    h = np.zeros((2, 3), dtype=complex)
    active = np.ones(2, dtype=bool)
    for beams in (mrt_directions(h, active), zf_directions(h, active),
                  rzf_directions(h, active, 0.1)):
        assert np.all(np.isfinite(beams))
        np.testing.assert_allclose(np.linalg.norm(beams, axis=1), 1.0, atol=1e-10)


@pytest.mark.parametrize("mode", ["mrt_sum", "principal", "maxmin"])
def test_common_direction_is_unit_norm(channel, mode):
    active = np.array([True, False, True])
    u = common_direction(channel, active, mode)
    assert u.shape == (4,)
    np.testing.assert_allclose(np.linalg.norm(u), 1.0, atol=1e-10)
    assert np.all(np.isfinite(u))


@pytest.mark.parametrize("mode", ["mrt_sum", "principal", "maxmin"])
def test_common_direction_reduces_to_mrt_for_one_user(channel, mode):
    active = np.array([False, True, False])
    u = common_direction(channel, active, mode)
    mrt = channel[1] / np.linalg.norm(channel[1])
    # equal up to an arbitrary global phase
    np.testing.assert_allclose(abs(np.vdot(mrt, u)), 1.0, atol=1e-8)


def test_principal_direction_maximises_the_total_common_gain(channel):
    active = np.ones(3, dtype=bool)
    u = common_direction(channel, active, "principal")
    best = float(np.sum(np.abs(np.conj(channel) @ u) ** 2))
    rng = np.random.default_rng(7)
    for _ in range(200):
        v = rng.standard_normal(4) + 1j * rng.standard_normal(4)
        v /= np.linalg.norm(v)
        assert float(np.sum(np.abs(np.conj(channel) @ v) ** 2)) <= best + 1e-9


def test_maxmin_safeguard_never_reduces_estimated_minimum_gain(channel):
    active = np.ones(3, dtype=bool)
    principal = common_direction(channel, active, "principal")
    maxmin = common_direction(channel, active, "maxmin")
    principal_min = float(np.min(np.abs(np.conj(channel) @ principal) ** 2))
    maxmin_min = float(np.min(np.abs(np.conj(channel) @ maxmin) ** 2))
    assert maxmin_min >= principal_min - 1e-10


def test_maxmin_is_deterministic(channel):
    active = np.ones(3, dtype=bool)
    first = maxmin_reweighted_direction(channel, active, iterations=7, damping=0.4)
    second = maxmin_reweighted_direction(channel, active, iterations=7, damping=0.4)
    np.testing.assert_allclose(first, second, atol=1e-12)


@pytest.mark.parametrize(
    "kwargs", [{"iterations": -1}, {"damping": 0.0}, {"damping": 1.1}]
)
def test_maxmin_rejects_invalid_parameters(channel, kwargs):
    with pytest.raises(ValueError):
        maxmin_reweighted_direction(channel, np.ones(3, dtype=bool), **kwargs)


def test_common_direction_with_no_active_user_is_zero(channel):
    u = common_direction(channel, np.zeros(3, dtype=bool), "principal")
    np.testing.assert_allclose(u, 0.0)


def test_default_rzf_regularization_formula():
    assert default_rzf_regularization(4, 1.0, 10.0) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        default_rzf_regularization(2, 1.0, 0.0)


def test_private_directions_dispatch_and_errors(channel):
    active = np.ones(3, dtype=bool)
    for mode in ("mrt", "zf", "rzf"):
        beams = private_directions(channel, active, mode, 1.0, 10.0)
        np.testing.assert_allclose(np.linalg.norm(beams, axis=1), 1.0, atol=1e-10)
    with pytest.raises(ValueError):
        private_directions(channel, active, "bogus", 1.0, 10.0)
    with pytest.raises(ValueError):
        rzf_directions(channel, active, reg=-1.0)
