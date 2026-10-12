from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from rsma_aoi.utils.config import EnvConfig

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_final_evaluation import (  # noqa: E402
    _environment_difference, build_uncertainty_units, load_protocol,
    paired_differences,
)


def test_frozen_protocol_is_well_formed():
    cfg, protocol = load_protocol(ROOT / "configs/thesis_final_evaluation.yaml")
    assert cfg.episode_length == protocol["episode_steps"] == 8192
    assert protocol["burn_in"] == 2048
    assert len(protocol["eval_seeds"]) == 6
    assert set(protocol["checkpoint_groups"]) == {
        "scratch_joint_gate", "objective_warm_joint_gate", "sdma_power_ppo"
    }


def test_environment_comparison_only_ignores_horizon():
    cfg = EnvConfig(episode_length=10)
    assert _environment_difference(cfg, replace(cfg, episode_length=100)) == {}
    difference = _environment_difference(cfg, replace(cfg, snr_db=5.0))
    assert difference == {"snr_db": (cfg.snr_db, 5.0)}


def test_uncertainty_units_do_not_pseudoreplicate_learned_episodes():
    rows = []
    for algorithm, training_seeds in (("learned", [0, 1]), ("fixed", [-1])):
        for training_seed in training_seeds:
            for eval_seed in (10, 11, 12):
                row = {"algorithm": algorithm, "training_seed": training_seed,
                       "eval_seed": eval_seed}
                row.update({name: float(training_seed + eval_seed) for name in __import__(
                    "run_final_evaluation").METRICS})
                rows.append(row)
    units = build_uncertainty_units(pd.DataFrame(rows))
    assert len(units[units.algorithm == "learned"]) == 2
    assert len(units[units.algorithm == "fixed"]) == 3
    assert set(units[units.algorithm == "learned"].uncertainty_unit) == {"training_seed"}


def test_paired_difference_uses_identical_eval_seeds():
    rows = []
    for algorithm, offset in (("method", -1.0), ("reference", 0.0)):
        for train_seed in (0, 1):
            for eval_seed in (20, 21):
                row = {"algorithm": algorithm, "training_seed": train_seed,
                       "eval_seed": eval_seed}
                row.update({name: 5.0 + offset for name in __import__(
                    "run_final_evaluation").METRICS})
                rows.append(row)
    paired = paired_differences(pd.DataFrame(rows), "reference")
    mean_aoi = paired[(paired.algorithm == "method") & (paired.metric == "mean_aoi")].iloc[0]
    assert mean_aoi.mean_difference == pytest.approx(-1.0)
    assert mean_aoi.paired_trajectories == 2
    assert mean_aoi.wins == 2
