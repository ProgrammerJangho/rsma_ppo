"""RSMA PHY: hand-checked SINRs, payload splitting, masking, power budget."""

from __future__ import annotations

import numpy as np
import pytest

from rsma_aoi.envs.rsma_phy import (
    RSMAPhysicalLayer, SlotAction, action_violation, split_payload,
)
from rsma_aoi.utils.config import EnvConfig

ORTHOGONAL_H = np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]])


def make_phy(**kwargs) -> RSMAPhysicalLayer:
    cfg = EnvConfig(
        **{
            "num_users": 2, "num_antennas": 2, "snr_db": None, "p_max": 10.0,
            "noise_power": 1.0, "private_beam": "mrt", "common_beam": "mrt_sum",
            "payload_bits": 100.0, "blocklength": 100, **kwargs,
        }
    )
    return RSMAPhysicalLayer(cfg)


def test_sinr_matches_the_hand_calculation():
    """Orthogonal unit channels, P_max=10, sigma^2=1, split [0.2, 0.4, 0.4].

    P_c = 2, P_1 = P_2 = 4.  With MRT, |h_k^H u_k|^2 = 1, |h_k^H u_j|^2 = 0 and
    the mrt_sum common beam gives |h_k^H u_c|^2 = 1/2, hence

        gamma_c,k = 2 * 0.5 / (4 * 1 + 4 * 0 + 1) = 0.2
        gamma_p,k = 4 * 1 / (0 + 1)               = 4
    """
    phy = make_phy()
    action = SlotAction(
        active=np.ones(2, dtype=bool),
        power_fractions=np.array([0.2, 0.4, 0.4]),
        rate_fractions=np.array([0.5, 0.5]),
        common_payload_scale=0.5,
    )
    res = phy.transmit(ORTHOGONAL_H, ORTHOGONAL_H, action)
    np.testing.assert_allclose(res.common_sinr, [0.2, 0.2], rtol=1e-12)
    np.testing.assert_allclose(res.private_sinr, [4.0, 4.0], rtol=1e-12)
    np.testing.assert_allclose(res.common_power, 2.0)
    np.testing.assert_allclose(res.private_power, [4.0, 4.0])
    np.testing.assert_allclose(res.total_power, 10.0)


def test_private_sinr_accounts_for_cross_interference():
    """Non-orthogonal channels: h_1 = [1, 0], h_2 = [1, 1]/sqrt(2), MRT beams.

    |h_1^H u_2|^2 = |[1,0]^H [1,1]/sqrt2|^2 = 0.5, so
    gamma_p,1 = P_1 * 1 / (P_2 * 0.5 + 1).
    """
    h = np.array([[1.0 + 0j, 0.0 + 0j], [1.0 + 0j, 1.0 + 0j] / np.sqrt(2)])
    phy = make_phy()
    action = SlotAction(np.ones(2, bool), np.array([0.0, 0.5, 0.5]), np.array([0.5, 0.5]), 0.0)
    res = phy.transmit(h, h, action)
    np.testing.assert_allclose(res.private_sinr[0], 5.0 / (5.0 * 0.5 + 1.0), rtol=1e-12)
    np.testing.assert_allclose(res.common_sinr, [0.0, 0.0])  # no common payload/power


def test_zero_power_stream_gives_zero_sinr():
    phy = make_phy()
    action = SlotAction(np.ones(2, bool), np.array([0.0, 1.0, 0.0]), np.array([0.5, 0.5]), 0.0)
    res = phy.transmit(ORTHOGONAL_H, ORTHOGONAL_H, action)
    assert res.common_sinr[1] == 0.0
    assert res.private_sinr[1] == 0.0
    assert res.private_sinr[0] > 0.0


def test_inactive_users_are_masked_everywhere():
    phy = make_phy()
    active = np.array([True, False])
    action = SlotAction(active, np.array([0.2, 0.8, 0.0]), np.array([1.0, 0.0]), 0.5)
    res = phy.transmit(ORTHOGONAL_H, ORTHOGONAL_H, action)
    assert res.private_power[1] == 0.0
    assert res.private_sinr[1] == 0.0
    assert res.common_sinr[1] == 0.0
    assert res.success_prob[1] == 0.0
    assert res.private_payload[1] == 0.0
    assert res.num_active == 1


def test_single_scheduled_user_is_numerically_stable():
    phy = make_phy(private_beam="rzf")
    for mask in ([True, False], [False, True]):
        active = np.array(mask)
        pf = np.zeros(3)
        pf[0] = 0.3
        pf[1:][active] = 0.7
        rf = np.zeros(2)
        rf[active] = 1.0
        res = phy.transmit(ORTHOGONAL_H, ORTHOGONAL_H, SlotAction(active, pf, rf, 0.5))
        assert np.all(np.isfinite(res.common_sinr))
        assert np.all(np.isfinite(res.private_sinr))
        assert 0.0 <= res.success_prob[int(np.argmax(active))] <= 1.0


def test_no_scheduled_user_produces_an_idle_slot():
    phy = make_phy()
    action = SlotAction(np.zeros(2, bool), np.zeros(3), np.zeros(2), 0.0)
    res = phy.transmit(ORTHOGONAL_H, ORTHOGONAL_H, action)
    assert res.num_active == 0
    assert res.total_power == 0.0
    np.testing.assert_allclose(res.success_prob, 0.0)


def test_total_power_never_exceeds_p_max():
    phy = make_phy()
    rng = np.random.default_rng(0)
    for _ in range(200):
        active = rng.random(2) < 0.7
        if not active.any():
            active[0] = True
        weights = rng.dirichlet(np.ones(int(active.sum()) + 1))
        pf = np.zeros(3)
        pf[0] = weights[0]
        pf[1:][active] = weights[1:]
        rf = np.zeros(2)
        rf[active] = rng.dirichlet(np.ones(int(active.sum())))
        res = phy.transmit(ORTHOGONAL_H, ORTHOGONAL_H,
                           SlotAction(active, pf, rf, float(rng.random())))
        assert res.total_power <= phy.total_power + 1e-9


def test_power_scale_reduces_the_radiated_power():
    phy = make_phy()
    action = SlotAction(np.ones(2, bool), np.array([0.2, 0.4, 0.4]),
                        np.array([0.5, 0.5]), 0.5, power_scale=0.25)
    res = phy.transmit(ORTHOGONAL_H, ORTHOGONAL_H, action)
    np.testing.assert_allclose(res.total_power, 2.5)


# --- payload split ---------------------------------------------------------
def test_split_payload_is_feasible_by_construction():
    payload = np.array([100.0, 100.0])
    active = np.ones(2, dtype=bool)
    rng = np.random.default_rng(1)
    for _ in range(500):
        c = rng.dirichlet(np.ones(2))
        rho = float(rng.random())
        d_c, d_p = split_payload(payload, active, c, rho)
        assert d_c >= 0.0
        assert np.all(d_p >= -1e-12)
        assert np.all(d_p <= payload + 1e-9)
        np.testing.assert_allclose(d_c + d_p.sum(), payload.sum(), rtol=1e-9)


def test_split_payload_hand_example():
    payload = np.array([100.0, 100.0])
    active = np.ones(2, dtype=bool)
    # cap = min(D_k / c_k) = 200, so rho = 0.5 -> D_c_tot = 100, D_c,k = 50
    d_c, d_p = split_payload(payload, active, np.array([0.5, 0.5]), 0.5)
    assert d_c == pytest.approx(100.0)
    np.testing.assert_allclose(d_p, [50.0, 50.0])
    # rho = 1 puts the maximum feasible amount on the common stream
    d_c, d_p = split_payload(payload, active, np.array([0.5, 0.5]), 1.0)
    assert d_c == pytest.approx(200.0)
    np.testing.assert_allclose(d_p, [0.0, 0.0], atol=1e-9)
    # rho = 0 is pure private transmission
    d_c, d_p = split_payload(payload, active, np.array([0.5, 0.5]), 0.0)
    assert d_c == 0.0
    np.testing.assert_allclose(d_p, payload)


def test_split_payload_asymmetric_shares():
    payload = np.array([100.0, 100.0])
    # cap = min(100/0.8, 100/0.2) = 125 -> D_c_tot = 125, D_c = [100, 25]
    d_c, d_p = split_payload(payload, np.ones(2, bool), np.array([0.8, 0.2]), 1.0)
    assert d_c == pytest.approx(125.0)
    np.testing.assert_allclose(d_p, [0.0, 75.0], atol=1e-9)


def test_split_payload_ignores_inactive_users():
    payload = np.array([100.0, 100.0])
    active = np.array([True, False])
    d_c, d_p = split_payload(payload, active, np.array([1.0, 0.0]), 0.5)
    assert d_c == pytest.approx(50.0)
    np.testing.assert_allclose(d_p, [50.0, 0.0])


def test_per_user_fraction_mode():
    payload = np.array([100.0, 200.0])
    d_c, d_p = split_payload(
        payload, np.ones(2, bool), np.array([0.25, 0.5]), 0.0, mode="per_user_fraction"
    )
    assert d_c == pytest.approx(25.0 + 100.0)
    np.testing.assert_allclose(d_p, [75.0, 100.0])
    with pytest.raises(ValueError):
        split_payload(payload, np.ones(2, bool), np.array([0.5, 0.5]), 0.5, mode="bogus")


def test_success_probability_combines_common_and_private_errors():
    phy = make_phy()
    action = SlotAction(np.ones(2, bool), np.array([0.4, 0.3, 0.3]), np.array([0.5, 0.5]), 0.5)
    res = phy.transmit(ORTHOGONAL_H, ORTHOGONAL_H, action)
    expected = (1.0 - res.common_bler) * (1.0 - res.private_bler)
    np.testing.assert_allclose(res.success_prob, expected)
    assert np.all((res.success_prob >= 0.0) & (res.success_prob <= 1.0))


# --- multiple-access variants ---------------------------------------------
def test_sdma_removes_the_common_stream_and_reuses_its_power():
    phy = make_phy(multiple_access="sdma")
    action = SlotAction(np.ones(2, bool), np.array([0.5, 0.25, 0.25]), np.array([0.5, 0.5]), 1.0)
    res = phy.transmit(ORTHOGONAL_H, ORTHOGONAL_H, action)
    assert res.common_power == 0.0
    assert res.common_payload == 0.0
    np.testing.assert_allclose(res.common_sinr, 0.0)
    np.testing.assert_allclose(res.private_power, [5.0, 5.0])  # renormalised budget
    np.testing.assert_allclose(res.private_payload, [100.0, 100.0])


def test_oma_is_interference_free_and_splits_the_blocklength():
    phy = make_phy(multiple_access="oma")
    h = np.array([[1.0 + 0j, 0.0 + 0j], [1.0 + 0j, 1.0 + 0j] / np.sqrt(2)])
    action = SlotAction(np.ones(2, bool), np.array([0.0, 0.5, 0.5]), np.array([0.5, 0.5]), 0.0)
    res = phy.transmit(h, h, action)
    np.testing.assert_allclose(res.effective_blocklength, [50.0, 50.0])
    # single-user MRT SINR = P_max * ||h||^2 / sigma^2, no interference term
    np.testing.assert_allclose(res.private_sinr[0], 10.0 * 1.0, rtol=1e-12)
    assert res.common_power == 0.0


def test_oma_differs_from_sdma_in_physics():
    common = dict(payload_bits=100.0, blocklength=100)
    h = np.array([[1.0 + 0j, 0.2 + 0j], [0.2 + 0j, 1.0 + 0j]])
    action = SlotAction(np.ones(2, bool), np.array([0.0, 0.5, 0.5]), np.array([0.5, 0.5]), 0.0)
    oma = make_phy(multiple_access="oma", **common).transmit(h, h, action)
    sdma = make_phy(multiple_access="sdma", **common).transmit(h, h, action)
    assert not np.allclose(oma.private_sinr, sdma.private_sinr)
    assert not np.allclose(oma.effective_blocklength, sdma.effective_blocklength)


def test_noma_applies_sic_and_is_finite():
    phy = make_phy(multiple_access="noma", payload_bits=50.0)
    h = np.array([[1.0 + 0j, 0.0 + 0j], [0.4 + 0j, 0.0 + 0j]])
    action = SlotAction(np.ones(2, bool), np.array([0.0, 0.3, 0.7]), np.array([0.5, 0.5]), 0.0)
    res = phy.transmit(h, h, action)
    assert np.all(np.isfinite(res.success_prob))
    assert np.all((res.success_prob >= 0.0) & (res.success_prob <= 1.0))
    np.testing.assert_allclose(res.total_power, 10.0)
    assert "noma_order" in res.extras


def test_unknown_multiple_access_raises():
    phy = make_phy()
    phy.mode = "bogus"
    with pytest.raises(ValueError):
        phy.transmit(ORTHOGONAL_H, ORTHOGONAL_H,
                     SlotAction(np.ones(2, bool), np.array([0.2, 0.4, 0.4]),
                                np.array([0.5, 0.5]), 0.5))


def test_shape_mismatches_raise():
    phy = make_phy()
    action = SlotAction(np.ones(2, bool), np.array([0.2, 0.4, 0.4]), np.array([0.5, 0.5]), 0.5)
    with pytest.raises(ValueError):
        phy.transmit(np.zeros((3, 2), dtype=complex), ORTHOGONAL_H, action)
    with pytest.raises(ValueError):
        SlotAction(np.ones(2, bool), np.array([0.5, 0.5]), np.array([0.5, 0.5]), 0.5)


def test_action_violation_is_zero_for_feasible_actions():
    action = SlotAction(np.array([True, False]), np.array([0.3, 0.7, 0.0]),
                        np.array([1.0, 0.0]), 0.5)
    assert action_violation(action) == pytest.approx(0.0)


def test_action_violation_detects_infeasibility():
    over = SlotAction(np.ones(2, bool), np.array([0.6, 0.6, 0.6]), np.array([0.5, 0.5]), 0.5)
    assert action_violation(over) > 0.0
    leak = SlotAction(np.array([True, False]), np.array([0.3, 0.4, 0.3]),
                      np.array([1.0, 0.0]), 0.5)
    assert action_violation(leak) > 0.0
    bad_rate = SlotAction(np.ones(2, bool), np.array([0.2, 0.4, 0.4]),
                          np.array([0.2, 0.2]), 0.5)
    assert action_violation(bad_rate) > 0.0


# --- batch evaluation must never drift from transmit -----------------------
@pytest.mark.parametrize("use_common", [True, False])
@pytest.mark.parametrize("rate_split_mode", ["simplex_scaled", "per_user_fraction"])
def test_batch_evaluation_matches_transmit_exactly(use_common, rate_split_mode):
    """evaluate_allocations is the single source of truth for the RSMA link."""
    phy = make_phy(
        num_users=3, num_antennas=2, private_beam="rzf", common_beam="principal",
        multiple_access="rsma" if use_common else "sdma", rate_split_mode=rate_split_mode,
    )
    rng = np.random.default_rng(0)
    h = (rng.standard_normal((3, 2)) + 1j * rng.standard_normal((3, 2))) / np.sqrt(2)
    h_hat = (rng.standard_normal((3, 2)) + 1j * rng.standard_normal((3, 2))) / np.sqrt(2)
    active = np.array([True, False, True])

    n = 32
    pf = np.zeros((n, 4))
    rf = np.zeros((n, 3))
    for i in range(n):
        weights = rng.dirichlet(np.ones(int(active.sum()) + 1))
        pf[i, 0] = weights[0]
        pf[i, 1:][active] = weights[1:]
        rf[i, active] = rng.dirichlet(np.ones(int(active.sum())))
    rho_r = rng.random(n)
    rho_p = rng.random(n)

    batch = phy.evaluate_allocations(
        h, active, pf, rf, rho_r, rho_p, use_common=use_common, h_hat=h_hat
    )
    for i in range(n):
        single = phy.transmit(
            h, h_hat, SlotAction(active, pf[i], rf[i], rho_r[i], rho_p[i])
        )
        np.testing.assert_array_equal(single.success_prob, batch["success_prob"][i])
        np.testing.assert_array_equal(single.common_sinr, batch["common_sinr"][i])
        np.testing.assert_array_equal(single.private_sinr, batch["private_sinr"][i])
        np.testing.assert_array_equal(single.private_payload, batch["private_payload"][i])
        assert single.common_payload == batch["common_payload"][i]
        assert single.total_power == pytest.approx(batch["total_power"][i])


def test_batch_evaluation_reuses_precomputed_beams():
    phy = make_phy(num_users=3, num_antennas=2, private_beam="rzf")
    rng = np.random.default_rng(1)
    h = (rng.standard_normal((3, 2)) + 1j * rng.standard_normal((3, 2))) / np.sqrt(2)
    active = np.array([True, True, False])
    beams = phy.rsma_beams(h, active)
    u_priv, u_common = beams
    np.testing.assert_allclose(np.linalg.norm(u_priv[active], axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(u_common), 1.0, atol=1e-10)

    pf = np.array([[0.3, 0.35, 0.35, 0.0]])
    rf = np.array([[0.5, 0.5, 0.0]])
    with_beams = phy.evaluate_allocations(h, active, pf, rf, np.array([0.4]), beams=beams)
    with_hhat = phy.evaluate_allocations(h, active, pf, rf, np.array([0.4]), h_hat=h)
    np.testing.assert_array_equal(with_beams["success_prob"], with_hhat["success_prob"])
    with pytest.raises(ValueError):
        phy.evaluate_allocations(h, active, pf, rf, np.array([0.4]))


def test_split_payload_batch_matches_the_scalar_version():
    from rsma_aoi.envs.rsma_phy import split_payload_batch

    payload = np.array([100.0, 150.0, 200.0])
    active = np.array([True, True, False])
    rng = np.random.default_rng(2)
    for mode in ("simplex_scaled", "per_user_fraction"):
        c = np.zeros((16, 3))
        c[:, active] = rng.dirichlet(np.ones(2), size=16)
        rho = rng.random(16)
        d_c, d_p = split_payload_batch(payload, active, c, rho, mode)
        for i in range(16):
            d_c_i, d_p_i = split_payload(payload, active, c[i], rho[i], mode)
            assert d_c_i == pytest.approx(d_c[i])
            np.testing.assert_allclose(d_p_i, d_p[i])


def test_batch_evaluation_handles_degenerate_and_empty_schedules():
    phy = make_phy(num_users=2, num_antennas=2)
    h = ORTHOGONAL_H
    active = np.ones(2, dtype=bool)
    pf = np.array([[0.0, 0.0, 0.0], [0.2, 0.4, 0.4]])
    rf = np.array([[0.5, 0.5], [0.5, 0.5]])
    out = phy.evaluate_allocations(h, active, pf, rf, np.array([0.5, 0.5]), h_hat=h)
    assert bool(out["degenerate"][0]) and not bool(out["degenerate"][1])
    np.testing.assert_allclose(out["success_prob"][0], 0.0)
    assert out["total_power"][0] == 0.0

    empty = phy.evaluate_allocations(
        h, np.zeros(2, dtype=bool), pf, rf, np.array([0.5, 0.5]), h_hat=h
    )
    np.testing.assert_allclose(empty["success_prob"], 0.0)


# --- mode-aware batch evaluation ------------------------------------------
def test_batch_evaluation_derives_the_scheme_from_the_config():
    """evaluate_allocations must not silently evaluate SDMA/OMA as RSMA."""
    rng = np.random.default_rng(3)
    h = (rng.standard_normal((3, 2)) + 1j * rng.standard_normal((3, 2))) / np.sqrt(2)
    active = np.array([True, True, False])
    pf = np.array([[0.4, 0.3, 0.3, 0.0]])
    rf = np.array([[0.5, 0.5, 0.0]])
    rho = np.array([0.6])

    rsma = make_phy(num_users=3, num_antennas=2, multiple_access="rsma")
    sdma = make_phy(num_users=3, num_antennas=2, multiple_access="sdma")
    assert rsma.default_use_common() is True
    assert sdma.default_use_common() is False

    out_rsma = rsma.evaluate_allocations(h, active, pf, rf, rho, h_hat=h)
    out_sdma = sdma.evaluate_allocations(h, active, pf, rf, rho, h_hat=h)
    assert out_rsma["common_power"][0] > 0.0
    assert out_sdma["common_power"][0] == 0.0
    assert out_sdma["common_payload"][0] == 0.0
    assert not np.allclose(out_rsma["success_prob"], out_sdma["success_prob"])


@pytest.mark.parametrize("scheme", ["rsma", "sdma", "oma"])
def test_batch_matches_transmit_for_every_scheme(scheme):
    """The vectorised path must reproduce transmit() for each access scheme."""
    phy = make_phy(num_users=3, num_antennas=2, multiple_access=scheme,
                   private_beam="rzf", common_beam="principal")
    rng = np.random.default_rng(11)
    h = (rng.standard_normal((3, 2)) + 1j * rng.standard_normal((3, 2))) / np.sqrt(2)
    h_hat = (rng.standard_normal((3, 2)) + 1j * rng.standard_normal((3, 2))) / np.sqrt(2)
    active = np.array([True, True, False])

    n = 24
    pf = np.zeros((n, 4))
    rf = np.zeros((n, 3))
    for i in range(n):
        weights = rng.dirichlet(np.ones(int(active.sum()) + 1))
        pf[i, 0] = weights[0]
        pf[i, 1:][active] = weights[1:]
        rf[i, active] = rng.dirichlet(np.ones(int(active.sum())))
    rho_r = rng.random(n)

    batch = phy.evaluate_allocations(h, active, pf, rf, rho_r, h_hat=h_hat)
    for i in range(n):
        single = phy.transmit(h, h_hat, SlotAction(active, pf[i], rf[i], rho_r[i]))
        np.testing.assert_allclose(single.success_prob, batch["success_prob"][i], rtol=1e-12)
        np.testing.assert_allclose(single.private_sinr, batch["private_sinr"][i], rtol=1e-12)
        assert single.total_power == pytest.approx(batch["total_power"][i])


def test_oma_batch_beams_are_mrt_and_blocklength_splits():
    phy = make_phy(num_users=3, num_antennas=2, multiple_access="oma")
    rng = np.random.default_rng(5)
    h = (rng.standard_normal((3, 2)) + 1j * rng.standard_normal((3, 2))) / np.sqrt(2)
    active = np.array([True, True, False])
    u_priv, u_common = phy.rsma_beams(h, active)
    np.testing.assert_allclose(np.linalg.norm(u_priv[active], axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(u_common, 0.0)
    for k in np.flatnonzero(active):
        np.testing.assert_allclose(u_priv[k], h[k] / np.linalg.norm(h[k]), atol=1e-12)

    out = phy.evaluate_allocations(
        h, active, np.array([[0.0, 0.5, 0.5, 0.0]]), np.array([[0.5, 0.5, 0.0]]),
        np.array([0.0]), h_hat=h,
    )
    np.testing.assert_allclose(out["effective_blocklength"][0][active], phy.blocklength / 2)
    assert out["effective_blocklength"][0][2] == 0.0


def test_noma_batch_evaluation_is_refused_rather_than_wrong():
    phy = make_phy(num_users=2, num_antennas=2, multiple_access="noma")
    with pytest.raises(NotImplementedError, match="NOMA"):
        phy.evaluate_allocations(
            ORTHOGONAL_H, np.ones(2, bool), np.array([[0.2, 0.4, 0.4]]),
            np.array([[0.5, 0.5]]), np.array([0.5]), h_hat=ORTHOGONAL_H,
        )
