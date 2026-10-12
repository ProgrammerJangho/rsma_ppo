"""Paired common-beam allocation audit invariants."""

import numpy as np

from rsma_aoi.evaluation.beam_audit import audit_state_allocations
from rsma_aoi.utils.config import EnvConfig


def test_beam_audit_uses_equal_budgets_and_feasible_sdma_boundary():
    cfg = EnvConfig(
        num_users=2, num_antennas=2, snr_db=10.0, blocklength=64,
        payload_bits=64.0, multiple_access="rsma", aoi_cap_enabled=False,
    )
    rng = np.random.default_rng(7)
    h = (rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))) / np.sqrt(2)
    rows = audit_state_allocations(
        cfg, np.array([2.0, 9.0]), h, h, np.ones(2, dtype=bool),
        seed=11, iterations=2, population=24,
    )
    assert {row["beam"] for row in rows} == {"principal", "mrt_sum", "maxmin"}
    assert {row["evaluations"] for row in rows} == {48}
    for row in rows:
        pf = row["boundary_power_fractions"]
        assert pf[0] == 0.0
        np.testing.assert_allclose(pf.sum(), 1.0, atol=1e-12)
        assert row["boundary_identity_max_abs"] <= 1e-10
        assert row["safe_gain_true"] >= min(
            row["raw_gain_true"], row["sdma_gain_true"]
        ) - 1e-12


def test_beam_audit_rejects_single_user_schedule():
    cfg = EnvConfig(num_users=2, num_antennas=2, multiple_access="rsma")
    h = np.eye(2, dtype=complex)
    try:
        audit_state_allocations(
            cfg, np.ones(2), h, h, np.array([True, False]),
            iterations=1, population=8,
        )
    except ValueError as exc:
        assert "at least two" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("single-user schedules must be rejected")
