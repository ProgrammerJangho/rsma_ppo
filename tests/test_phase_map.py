"""Phase-map machinery: point sampling, config construction, manifests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from map_operating_regimes import ARMS, evaluate_point, make_config, sample_points  # noqa: E402

from rsma_aoi.evaluation.fairness import audit_resource_parity  # noqa: E402


def test_sampled_points_stay_inside_the_grid():
    from map_operating_regimes import (
        ANCHOR_POINTS, BLOCKLENGTH, CORRELATION, CSIT_ERROR, PAYLOAD_BITS,
        SCENARIOS, SNR_DB,
    )

    points = sample_points(40, seed=0)
    assert len(points) >= 40
    anchors = [dict(a) for a in ANCHOR_POINTS]
    for anchor in anchors:
        assert anchor in points, "the reference configurations must always be screened"
    for point in points:
        if point in anchors:
            continue  # anchors are deliberately off-grid
        assert (point["num_users"], point["num_antennas"]) in SCENARIOS
        assert point["snr_db"] in SNR_DB
        assert point["csit_error"] in CSIT_ERROR
        assert point["blocklength"] in BLOCKLENGTH
        assert point["payload_bits"] in PAYLOAD_BITS
        assert point["correlation"] in CORRELATION


def test_sampling_is_reproducible():
    assert sample_points(20, seed=3) == sample_points(20, seed=3)
    assert sample_points(20, seed=3) != sample_points(20, seed=4)


def test_every_sampled_point_builds_a_valid_parity_passing_config():
    for point in sample_points(12, seed=1):
        cfg = make_config(point)
        cfg.validate()
        assert cfg.num_users == point["num_users"]
        assert cfg.blocklength == point["blocklength"]
        assert audit_resource_parity(cfg, num_states=3).passed


def test_arms_cover_the_required_schemes():
    schemes = {scheme for _, scheme, _, _ in ARMS}
    assert schemes == {"oma", "sdma", "rsma"}
    names = {arm for arm, *_ in ARMS}
    for required in ("oma_heuristic", "sdma_equal", "rsma_equal", "rsma_heuristic",
                     "rsma_opt", "sdma_opt"):
        assert required in names


def test_evaluate_point_produces_the_documented_columns():
    point = {"num_users": 2, "num_antennas": 2, "snr_db": 10.0, "csit_error": 0.3,
             "blocklength": 64, "payload_bits": 64.0, "correlation": 0.0}
    row = evaluate_point((point, 4, "cem_fast", 0, 4))
    for arm, *_ in ARMS:
        assert f"aoi_{arm}" in row and np.isfinite(row[f"aoi_{arm}"])
        assert f"aoi_{arm}_ci_low" in row and f"aoi_{arm}_ci_high" in row
    for key in ("gain_opt_vs_sdma", "gain_opt_vs_oma", "gain_equal_vs_sdma",
                "parity_ok", "evaluations", "runtime_s", "load_ratio",
                "rate_bits_per_use"):
        assert key in row
    assert row["parity_ok"] is True
    assert row["evaluations"] > 0


def test_rsma_never_loses_to_sdma_because_it_contains_it():
    """RSMA with f_c = 0 reproduces SDMA, so the measured gain cannot be negative."""
    for seed, snr in ((0, 5.0), (1, 15.0)):
        point = {"num_users": 2, "num_antennas": 2, "snr_db": snr, "csit_error": 0.3,
                 "blocklength": 64, "payload_bits": 96.0, "correlation": 0.0}
        row = evaluate_point((point, 6, "cem_fast", seed, 4))
        assert row["gain_opt_vs_sdma"] >= -1e-9, (
            "negative gain means the SDMA fallback for the RSMA arm is broken"
        )


def test_heuristic_recovery_is_nan_when_there_is_no_gain_to_recover():
    """A ratio against a near-zero denominator must not be reported as a fraction."""
    point = {"num_users": 2, "num_antennas": 2, "snr_db": 20.0, "csit_error": 0.0,
             "blocklength": 512, "payload_bits": 32.0, "correlation": 0.0}
    row = evaluate_point((point, 4, "cem_fast", 0, 4))
    if row["gain_opt_vs_sdma"] <= 0.005:
        assert np.isnan(row["heuristic_gain_recovery"])


@pytest.mark.parametrize("config", ["configs/op1_rsma.yaml", "configs/op2_rsma.yaml",
                                    "configs/op3_rsma.yaml"])
def test_selected_operating_points_are_valid_and_fair(config):
    from rsma_aoi.utils.config import EnvConfig

    cfg = EnvConfig.from_yaml(config)
    cfg.validate()
    assert cfg.num_users == 2 and cfg.num_antennas == 2
    assert cfg.multiple_access == "rsma"
    report = audit_resource_parity(cfg, num_states=6)
    assert report.passed, report.mismatches


def test_manifest_records_the_screening_metric_caveat(tmp_path):
    from rsma_aoi.evaluation.fairness import build_manifest
    from rsma_aoi.utils.config import EnvConfig

    cfg = EnvConfig.from_yaml("configs/op1_rsma.yaml")
    manifest, _ = build_manifest(
        "phase_map_test", cfg, metric="expected next-slot weighted AoI (myopic)"
    )
    payload = json.loads(manifest.save(tmp_path / "m.json").read_text())
    assert "myopic" in payload["extra"]["metric"]
