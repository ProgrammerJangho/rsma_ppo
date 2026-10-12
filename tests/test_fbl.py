"""Finite-blocklength model: monotonicity, limits and numerical robustness."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erfc

from rsma_aoi.envs.fbl import (
    BLER_MAX, BLER_MIN, channel_dispersion, fbl_bler, fbl_max_payload, fbl_rate,
    fixed_bler, link_bler, q_function, q_function_inv, shannon_bler, shannon_capacity,
)


def test_q_function_matches_erfc():
    x = np.linspace(-4.0, 4.0, 17)
    np.testing.assert_allclose(q_function(x), 0.5 * erfc(x / np.sqrt(2.0)), rtol=1e-10)


def test_q_function_inverse_roundtrip():
    eps = np.array([1e-6, 1e-3, 0.1, 0.5, 0.9])
    np.testing.assert_allclose(q_function(q_function_inv(eps)), eps, rtol=1e-8)


def test_capacity_and_dispersion_formulas():
    gamma = np.array([0.5, 1.0, 10.0])
    np.testing.assert_allclose(shannon_capacity(gamma), np.log2(1.0 + gamma))
    expected = (1.0 - 1.0 / (1.0 + gamma) ** 2) * np.log2(np.e) ** 2
    np.testing.assert_allclose(channel_dispersion(gamma), expected)


def test_bler_decreases_with_sinr():
    gamma = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 20.0])
    bler = fbl_bler(gamma, 128, 128.0)
    assert np.all(np.diff(bler) <= 0.0)


def test_bler_decreases_with_blocklength_at_fixed_payload():
    blers = [float(fbl_bler(2.0, n, 128.0)) for n in (100, 200, 400, 800)]
    assert all(b2 <= b1 for b1, b2 in zip(blers, blers[1:]))


def test_bler_increases_with_payload():
    blers = [float(fbl_bler(3.0, 128, d)) for d in (32.0, 64.0, 128.0, 256.0, 512.0)]
    assert all(b2 >= b1 for b1, b2 in zip(blers, blers[1:]))


def test_bler_limits():
    assert float(fbl_bler(1e6, 128, 64.0)) < 1e-9      # very high SINR -> ~0
    assert float(fbl_bler(1e-6, 128, 512.0)) > 1.0 - 1e-6  # very low SINR -> ~1


def test_bler_stays_in_the_unit_interval_and_is_finite():
    gamma = np.array([0.0, 1e-15, 1e-3, 1.0, 1e3, 1e12])
    for n in (1, 8, 128, 10_000):
        for d in (1.0, 64.0, 1e5):
            bler = fbl_bler(gamma, n, d)
            assert np.all(np.isfinite(bler))
            assert np.all(bler >= BLER_MIN) and np.all(bler <= BLER_MAX)


def test_zero_payload_means_no_error():
    assert float(fbl_bler(0.5, 128, 0.0)) == 0.0
    assert float(shannon_bler(0.5, 128, 0.0)) == 0.0


def test_rate_and_payload_are_mutually_consistent():
    gamma, n, target = 4.0, 256, 1e-3
    payload = float(fbl_max_payload(gamma, n, target))
    np.testing.assert_allclose(float(fbl_bler(gamma, n, payload)), target, rtol=1e-6)


def test_fbl_rate_is_below_shannon_capacity():
    gamma = np.array([0.5, 2.0, 10.0])
    assert np.all(fbl_rate(gamma, 128, 1e-3) < shannon_capacity(gamma))
    # and approaches capacity as n grows
    assert np.all(fbl_rate(gamma, 10**7, 1e-3) > 0.99 * shannon_capacity(gamma))


def test_shannon_model_is_a_deterministic_outage():
    # C(3) = 2 bits/cu; n = 100 -> 200 bits is exactly at the boundary
    assert float(shannon_bler(3.0, 100, 150.0)) == BLER_MIN
    assert float(shannon_bler(3.0, 100, 250.0)) == BLER_MAX


def test_fixed_model_and_dispatch():
    np.testing.assert_allclose(fixed_bler((3,), 0.75), 0.25)
    out = link_bler("fixed", np.ones(3), 128, np.full(3, 10.0), fixed_success_prob=0.9)
    np.testing.assert_allclose(out, 0.1)
    with pytest.raises(ValueError):
        link_bler("nope", np.ones(2), 128, np.ones(2))


def test_invalid_blocklength_raises():
    with pytest.raises(ValueError):
        fbl_bler(1.0, 0, 10.0)
