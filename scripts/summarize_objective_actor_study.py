#!/usr/bin/env python3
"""Summarize the objective-aware actor pretraining runs.

The final row in each scalar log contains the held-out, hard-gate PHY audit.
Validation values are taken from the checkpoint selected during training, so the
table records the model-selection criterion separately from the test result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


RUNS = (
    (5, "objective_only", 0, "objective_actor_rician_5deg_seed0_v1", True),
    (5, "action_objective", 0, "objective_actor_rician_5deg_actionmix_seed0_v1", False),
    (20, "objective_only", 0, "objective_actor_rician_20deg_seed0_v1", True),
    (20, "objective_only", 1, "objective_actor_rician_20deg_seed1_v1", True),
    (20, "objective_only", 2, "objective_actor_rician_20deg_seed2_v1", True),
    (60, "objective_only", 0, "objective_actor_rician_60deg_seed0_v1", True),
)


def _finite(value: object) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if pd.notna(parsed) else None


def summarize(root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for angle, loss_variant, pretrain_seed, directory, main_study in RUNS:
        run = root / "runs" / directory
        scalars = pd.read_csv(run / "scalars.csv")
        test = scalars[scalars["test/predicted_expected_aoi"].notna()].iloc[-1]
        checkpoint = torch.load(
            run / "checkpoint.pt", map_location="cpu", weights_only=False
        )
        validation = checkpoint.get("metrics", {})
        selected_value = _finite(validation.get("val/predicted_expected_aoi"))
        valid_rows = scalars[scalars["val/predicted_expected_aoi"].notna()]
        best_epoch = None
        if selected_value is not None and not valid_rows.empty:
            offset = (valid_rows["val/predicted_expected_aoi"] - selected_value).abs()
            best_epoch = int(valid_rows.loc[offset.idxmin(), "step"])
        rows.append(
            {
                "angle_deg": angle,
                "loss_variant": loss_variant,
                "main_study": main_study,
                "pretrain_seed": pretrain_seed,
                "best_epoch": best_epoch,
                "validation_predicted_expected_aoi": _finite(
                    validation.get("val/predicted_expected_aoi")
                ),
                "validation_oracle_expected_aoi": _finite(
                    validation.get("val/oracle_expected_aoi")
                ),
                "test_predicted_expected_aoi": _finite(
                    test["test/predicted_expected_aoi"]
                ),
                "test_oracle_expected_aoi": _finite(
                    test["test/oracle_expected_aoi"]
                ),
                "test_objective_gap": _finite(test["test/objective_gap"]),
                "test_gate_accuracy": _finite(test["test/gate_accuracy"]),
                "test_power_l1": _finite(test["test/power_l1"]),
                "test_rate_l1": _finite(test["test/rate_l1"]),
                "test_rho_mae": _finite(test["test/rho_mae"]),
                "run_directory": str(run.relative_to(root)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir", default="results/rician_objective_actor_study_v1"
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output = root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(root)
    summary.to_csv(output / "summary.csv", index=False)
    metrics = [
        "validation_predicted_expected_aoi",
        "test_predicted_expected_aoi",
        "test_objective_gap",
        "test_gate_accuracy",
    ]
    aggregate = (
        summary[summary.main_study]
        .groupby(["angle_deg", "loss_variant"])[metrics]
        .agg(["count", "mean", "std"])
    )
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate.reset_index().to_csv(output / "aggregate.csv", index=False)

    manifest = {
        "description": "Held-out hard-gate audit of objective-aware actor pretraining",
        "main_study_rule": (
            "Use objective_only at all angles; the 5-degree action_objective run "
            "is retained only as a sensitivity analysis."
        ),
        "rows": len(summary),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
