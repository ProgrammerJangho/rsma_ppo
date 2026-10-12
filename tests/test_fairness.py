"""Resource-parity audit, gain sign convention and experiment manifests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rsma_aoi.evaluation.fairness import (
    PARITY_FIELDS, audit_resource_parity, build_manifest, gain_table,
    measure_scheme_budget, rsma_gain,
)
from rsma_aoi.utils.config import EnvConfig


def test_gain_sign_convention():
    """Positive means RSMA is better (lower AoI)."""
    assert rsma_gain(10.0, 8.0) == pytest.approx(0.2)
    assert rsma_gain(10.0, 12.0) == pytest.approx(-0.2)
    assert rsma_gain(10.0, 10.0) == 0.0
    assert np.isnan(rsma_gain(0.0, 5.0))
    assert np.isnan(rsma_gain(float("nan"), 5.0))


def test_gain_table_uses_the_same_convention():
    table = gain_table({"rsma": 9.0, "sdma": 10.0, "oma": 8.0})
    assert set(table) == {"gain_vs_sdma", "gain_vs_oma"}
    assert table["gain_vs_sdma"] == pytest.approx(0.1)   # RSMA better than SDMA
    assert table["gain_vs_oma"] == pytest.approx(-0.125)  # RSMA worse than OMA
    with pytest.raises(KeyError):
        gain_table({"sdma": 1.0})


@pytest.mark.parametrize(
    "cfg_name", ["configs/m2_hybrid.yaml", "configs/m1_rsma.yaml", "configs/env_default.yaml"]
)
def test_shipped_configs_pass_the_parity_audit(cfg_name):
    cfg = EnvConfig.from_yaml(cfg_name)
    report = audit_resource_parity(cfg, num_states=8)
    assert report.passed, report.mismatches
    report.raise_if_failed()


def test_parity_covers_every_documented_quantity():
    for field in ("channel_uses", "radiated_energy", "peak_power",
                  "offered_payload_bits", "slot_blocklength", "bandwidth_channel_uses"):
        assert field in PARITY_FIELDS


def test_oma_and_rsma_consume_the_same_slot_resources():
    """OMA splits the slot; the totals must still match superposition schemes."""
    cfg = EnvConfig(num_users=4, num_antennas=2, blocklength=128, payload_bits=192.0)
    rng = np.random.default_rng(0)
    h = (rng.standard_normal((4, 2)) + 1j * rng.standard_normal((4, 2))) / np.sqrt(2)
    active = np.array([True, True, False, False])

    rsma = measure_scheme_budget(cfg, "rsma", active, h, h)
    oma = measure_scheme_budget(cfg, "oma", active, h, h)
    assert oma.channel_uses == pytest.approx(rsma.channel_uses)
    assert oma.radiated_energy == pytest.approx(rsma.radiated_energy, rel=1e-9)
    assert oma.peak_power == pytest.approx(rsma.peak_power, rel=1e-9)
    assert oma.offered_payload_bits == pytest.approx(rsma.offered_payload_bits)
    assert oma.num_scheduled == rsma.num_scheduled


def test_parity_audit_detects_a_genuine_mismatch(monkeypatch):
    """A scheme given twice the power must fail the audit."""
    import rsma_aoi.evaluation.fairness as fairness

    cfg = EnvConfig(num_users=2, num_antennas=2)
    original = fairness.measure_scheme_budget

    def biased(cfg_, scheme, *args, **kwargs):
        budget = original(cfg_, scheme, *args, **kwargs)
        if scheme == "oma":
            return fairness.SchemeBudget(
                **{**budget.as_dict(), "radiated_energy": budget.radiated_energy * 2.0}
            )
        return budget

    monkeypatch.setattr(fairness, "measure_scheme_budget", biased)
    report = fairness.audit_resource_parity(cfg, num_states=4)
    assert not report.passed
    assert any("radiated_energy" in m for m in report.mismatches)
    with pytest.raises(AssertionError):
        report.raise_if_failed()


def test_manifest_is_machine_readable(tmp_path):
    cfg = EnvConfig.from_yaml("configs/m2_hybrid.yaml")
    manifest, report = build_manifest("unit_test", cfg, seed=0, note="hello")
    assert report.passed
    path = manifest.save(tmp_path / "manifest.json")
    payload = json.loads(path.read_text())
    assert payload["name"] == "unit_test"
    assert payload["parity"]["passed"] is True
    assert payload["extra"]["note"] == "hello"
    assert "positive means RSMA is better" in payload["gain_convention"]["sign"]
    for key in ("bandwidth", "slot_duration", "energy", "channel_realisation",
                "csit_realisation", "aoi_update_timing"):
        assert key in payload["assumptions"]
    assert payload["env_config"]["blocklength"] == cfg.blocklength
