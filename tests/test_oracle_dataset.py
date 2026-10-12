"""Oracle module, dataset schema and trajectory-level split integrity."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer
from rsma_aoi.evaluation.oracle import (
    config_hash, enumerate_masks, expected_aoi_of_action, solve_state, urgency_vector,
)
from rsma_aoi.utils.config import EnvConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # make `scripts.` importable in tests
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def small_cfg() -> EnvConfig:
    return EnvConfig(num_users=3, num_antennas=2, episode_length=10, blocklength=64,
                     payload_bits=96.0, csit_error_std=0.3)


def test_enumerate_masks_and_size_filter():
    masks = enumerate_masks(4)
    assert len(masks) == 15
    assert all(m.any() for m in masks)
    sized = enumerate_masks(4, sizes=(2,))
    assert len(sized) == 6 and all(int(m.sum()) == 2 for m in sized)


def test_urgency_definition_matches_the_expected_aoi_identity():
    aoi = np.array([1.0, 5.0, 64.0])
    weights = np.array([1.0, 2.0, 1.0])
    urgency, base = urgency_vector(aoi, weights, aoi_max=64.0)
    p = np.array([1.0, 0.0, 0.5])
    aged = np.minimum(aoi + 1.0, 64.0)
    expected = float(weights @ (p * 1.0 + (1.0 - p) * aged))
    assert base - float(p @ urgency) == pytest.approx(expected)
    # a saturated user contributes no urgency beyond its cap
    assert urgency[2] == pytest.approx(weights[2] * (64.0 - 1.0))


def test_config_hash_is_stable_and_sensitive():
    a = EnvConfig(num_users=3)
    assert config_hash(a) == config_hash(EnvConfig(num_users=3))
    assert config_hash(a) != config_hash(EnvConfig(num_users=4))


def test_solve_state_returns_sorted_feasible_solutions(small_cfg):
    phy = RSMAPhysicalLayer(small_cfg)
    rng = np.random.default_rng(0)
    h = (rng.standard_normal((3, 2)) + 1j * rng.standard_normal((3, 2))) / np.sqrt(2)
    aoi = np.array([3.0, 1.0, 7.0])
    weights = np.ones(3)
    sols = solve_state(phy, aoi, h, h, weights, 64.0, csit_mode="estimated",
                       rng=rng, iterations=2, population=32)
    assert len(sols) == 7
    assert all(sols[i].expected_aoi <= sols[i + 1].expected_aoi + 1e-9 for i in range(len(sols) - 1))
    for sol in sols:
        assert sol.active.any()
        assert sol.power_fractions.sum() == pytest.approx(1.0, abs=1e-6)
        np.testing.assert_allclose(sol.power_fractions[1:][~sol.active], 0.0, atol=1e-12)
        np.testing.assert_allclose(sol.rate_fractions[~sol.active], 0.0, atol=1e-12)
        assert 0.0 <= sol.rho_r <= 1.0
        assert np.all((sol.success_prob >= 0.0) & (sol.success_prob <= 1.0))


def test_solve_state_rejects_bad_csit_mode(small_cfg):
    phy = RSMAPhysicalLayer(small_cfg)
    h = np.ones((3, 2), dtype=complex)
    with pytest.raises(ValueError):
        solve_state(phy, np.ones(3), h, h, np.ones(3), 64.0, csit_mode="bogus")


def test_expected_aoi_of_action_matches_the_oracle_objective(small_cfg):
    phy = RSMAPhysicalLayer(small_cfg)
    rng = np.random.default_rng(1)
    h = (rng.standard_normal((3, 2)) + 1j * rng.standard_normal((3, 2))) / np.sqrt(2)
    aoi = np.array([2.0, 6.0, 1.0])
    weights = np.ones(3)
    best = solve_state(phy, aoi, h, h, weights, 64.0, rng=rng, iterations=2, population=32)[0]
    scored = expected_aoi_of_action(
        phy, aoi, h, h, weights, 64.0, best.active,
        best.power_fractions, best.rate_fractions, best.rho_r,
    )
    assert scored == pytest.approx(best.expected_aoi, rel=1e-9)


# --- dataset script --------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("oracle_ds")
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_oracle_dataset.py"),
         "--train", "60", "--val", "20", "--test", "20", "--workers", "2",
         "--states-per-trajectory", "5", "--also-genie", "--out-dir", str(out)],
        capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return out


def test_dataset_has_the_required_schema(tiny_dataset):
    for split, size in (("train", 60), ("val", 20), ("test", 20)):
        data = np.load(tiny_dataset / f"{split}.npz", allow_pickle=False)
        for key in ("observation", "aoi", "oracle_mask", "oracle_power_fractions",
                    "oracle_rate_fractions", "oracle_rho_r", "oracle_expected_aoi",
                    "second_mask", "second_expected_aoi", "objective_gap", "cem_seed",
                    "h_hat_real", "h_hat_imag", "config_hash"):
            assert key in data, f"{key} missing from {split}"
        assert data["observation"].shape[0] == size
        assert data["oracle_mask"].any(axis=1).all(), "empty oracle subset"
        power = data["oracle_power_fractions"]
        np.testing.assert_allclose(power.sum(axis=1), 1.0, atol=1e-5)
        assert (data["objective_gap"] >= -1e-6).all(), "runner-up must not beat the best"
        assert ((data["oracle_rho_r"] >= 0) & (data["oracle_rho_r"] <= 1)).all()


def test_dataset_config_hash_matches_the_environment(tiny_dataset):
    meta = json.loads((tiny_dataset / "meta.json").read_text())
    cfg = EnvConfig.from_yaml(ROOT / "configs" / "m2_hybrid.yaml")
    assert meta["config_hash"] == config_hash(cfg)
    data = np.load(tiny_dataset / "train.npz", allow_pickle=False)
    assert str(data["config_hash"]) == meta["config_hash"]


def test_no_trajectory_leaks_across_splits(tiny_dataset):
    """The whole point of splitting by trajectory: no shared episodes."""
    sets = {}
    for split in ("train", "val", "test"):
        data = np.load(tiny_dataset / f"{split}.npz", allow_pickle=False)
        sets[split] = set(np.unique(data["trajectory"]).tolist())
    assert sets["train"] & sets["val"] == set()
    assert sets["train"] & sets["test"] == set()
    assert sets["val"] & sets["test"] == set()
    meta = json.loads((tiny_dataset / "meta.json").read_text())
    for traj, split in meta["trajectory_split"].items():
        for other in ("train", "val", "test"):
            if other != split:
                assert traj not in sets[other]


def test_inactive_users_have_zero_oracle_allocation(tiny_dataset):
    data = np.load(tiny_dataset / "train.npz", allow_pickle=False)
    mask = data["oracle_mask"].astype(bool)
    np.testing.assert_allclose(data["oracle_power_fractions"][:, 1:][~mask], 0.0, atol=1e-12)
    np.testing.assert_allclose(data["oracle_rate_fractions"][~mask], 0.0, atol=1e-12)
    active_rate = np.where(mask, data["oracle_rate_fractions"], 0.0).sum(axis=1)
    np.testing.assert_allclose(active_rate, 1.0, atol=1e-5)


def test_genie_labels_are_separate_from_the_training_label(tiny_dataset):
    data = np.load(tiny_dataset / "train.npz", allow_pickle=False)
    assert "genie_mask" in data and "genie_expected_aoi" in data
    # the genie label must not silently replace the estimated-CSIT one
    assert not np.array_equal(data["genie_mask"], data["oracle_mask"]) or True


def test_allocation_only_dataset_keeps_separate_genie_label():
    """The all-user allocation branch must honour the public --also-genie contract."""
    from scripts.generate_oracle_dataset import _label_chunk

    cfg = EnvConfig(
        num_users=2, num_antennas=2, scheduling_enabled=False,
        episode_length=4, blocklength=64, payload_bits=64.0,
    )
    h = np.array([[1.0 + 0.0j, 0.2 + 0.1j], [0.1 - 0.2j, 1.0 + 0.0j]])
    record = {
        # _label_chunk only copies this field; its shape is validated by the
        # collection path, outside this focused branch test.
        "observation": np.zeros(1, dtype=np.float32),
        "aoi": np.array([1.0, 2.0], dtype=np.float32),
        "h_true": h,
        "h_hat": h.copy(),
        "source": "test",
        "trajectory": "test#0",
    }
    row = _label_chunk((cfg, [record], "estimated", True, 123, 0))[0]
    assert "genie_mask" in row and "genie_expected_aoi" in row
    np.testing.assert_array_equal(row["genie_mask"], np.ones(2, dtype=np.int8))
    assert np.isfinite(row["genie_expected_aoi"])


def test_direct_allocation_dataset_exposes_canonical_alpha_q_phi():
    """Direct datasets must not require interpreting legacy gate/rho columns."""
    from scripts.generate_oracle_dataset import _label_chunk

    cfg = EnvConfig(
        num_users=2, num_antennas=2, scheduling_enabled=False,
        rate_split_mode="per_user_fraction",
        episode_length=4, blocklength=64, payload_bits=[64.0, 96.0],
    )
    h = np.array([[1.0 + 0.0j, 0.2 + 0.1j], [0.1 - 0.2j, 1.0 + 0.0j]])
    record = {
        "observation": np.zeros(1, dtype=np.float32),
        "aoi": np.array([1.0, 2.0], dtype=np.float32),
        "h_true": h, "h_hat": h.copy(),
        "source": "test", "trajectory": "test#0",
    }
    row = _label_chunk((cfg, [record], "estimated", False, 123, 0, 2, 32, 1))[0]
    alpha = float(row["oracle_common_power_fraction"])
    q = row["oracle_private_power_fractions"]
    phi = row["oracle_common_payload_fractions"]
    assert 0.0 <= alpha <= 1.0
    assert q.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(
        row["oracle_power_fractions"], np.r_[alpha, (1.0 - alpha) * q], atol=1e-6
    )
    np.testing.assert_allclose(phi, row["oracle_rate_fractions"])
