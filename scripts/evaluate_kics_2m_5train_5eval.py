#!/usr/bin/env python3
"""Target-condition evaluation: 5 training seeds x 5 evaluation seeds x 6 permutations.

Generalises ``evaluate_kics_2m_five_eval_seeds.py`` (seed 0 only) to all five
independently trained policies and adds the action-role statistics of
``audit_feedforward_state_adaptation.py``.

Protocol (20 deg, 15 dB, n=128, CSIT error 0.2, 768 slots, 256 burn-in):
    eval_seed = 990278 + 10000 * eval_repeat + 100 * permutation
Every method sees the same channel/CSIT/ACK seeds.

Aggregation (no trajectory is treated as an independent sample):
    1. mean over the 6 permutations      -> (label, training_seed, eval_repeat)
    2. mean over the 5 evaluation seeds  -> (label, training_seed)
    3. mean and 95% Student-t CI over the 5 training-seed means (df = 4)
Rule baselines have no training seed; their CI is over the 5 evaluation-seed
means and is reported separately. Paired gains match training seed and
evaluation seed before step 2.

Action-role statistics (learned RSMA policies, original actions only) and
closed-loop counterfactuals (alpha fixed at its calibration mean, q replaced by
the calibration AoI-rank rule, overall phi level fixed) are computed per
training seed and summarised with the same training-seed CI.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import permutations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from rsma_aoi.agents.ppo import PPOPolicy  # noqa: E402
from rsma_aoi.envs.aoi_env import AoIRSMAEnv  # noqa: E402
from rsma_aoi.utils.config import EnvConfig  # noqa: E402
from scripts.audit_feedforward_state_adaptation import _replace, calibrate  # noqa: E402
from scripts.evaluate_thesis_rule_baselines import evaluate as evaluate_rule  # noqa: E402

CONFIG = "configs/thesis_rician_k3_nt2_mixed_phy_sinr.yaml"
TRAINING_SEEDS = (0, 1, 2, 3, 4)
EVAL_REPEATS = (0, 1, 2, 3, 4)
ANGLE_SETS = list(permutations((-10.0, 0.0, 10.0)))
LEARNED = {
    "rsma_ppo_2m": "runs/matched_objective_rsma_ppo_k3_20deg_seed{seed}_continue_to_2097k_v1/checkpoint.pt",
    "sdma_ppo_2m": "runs/q_only_ppo_k3_20deg_seed{seed}_continue_to_2097k_v1/checkpoint.pt",
    "rsma_ppo_524k": "runs/matched_objective_rsma_ppo_k3_20deg_from_stage1_seed{seed}_524k_v1/checkpoint.pt",
    "sdma_ppo_524k": "runs/q_only_ppo_k3_20deg_from_stage1_seed{seed}_524k_v1/checkpoint.pt",
}
RULES = {
    "fixed_rule_rsma": ("oldest_common_fixed_alpha", 0.70),
    "max_aoi_single_user": ("max_aoi_single_user", 0.0),
}
ROLE_POLICY = "rsma_ppo_2m"
COUNTERFACTUALS = ("fixed_alpha", "rank_q", "fixed_phi_level")
STEPS, BURN_IN = 768, 256


def eval_seed(repeat: int, permutation: int) -> int:
    return 990278 + 10000 * repeat + 100 * permutation


def _cfg(angles) -> EnvConfig:
    # Identical to scripts/evaluate_mixed_phy_sinr_ppo.py::_job for kind="ppo".
    base = EnvConfig.from_yaml(ROOT / CONFIG)
    return replace(
        base, user_angles_deg=list(angles), user_angle_sets_deg=None,
        snr_db=15.0, snr_db_values=None, blocklength=128, blocklength_values=None,
        multiple_access="rsma", episode_length=STEPS,
    )


def _alignment(h_hat: np.ndarray) -> float:
    norms = np.linalg.norm(h_hat, axis=1)
    values = [
        abs(np.vdot(h_hat[i], h_hat[j])) / max(norms[i] * norms[j], 1e-12)
        for i in range(len(h_hat)) for j in range(i + 1, len(h_hat))
    ]
    return float(np.mean(values))


def run_learned(job: dict) -> dict:
    cfg = _cfg(job["angles"])
    policy = PPOPolicy.from_checkpoint(ROOT / job["checkpoint"], deterministic=True, device="cpu")[0]
    if hasattr(policy, "reset"):
        policy.reset(seed=job["eval_seed"])
    env = AoIRSMAEnv(cfg)
    obs, _ = env.reset(seed=job["eval_seed"])
    record = job["mode"] == "original" and job["record_actions"]
    ages, success, alphas, common_bits = [], [], [], []
    slots = []
    for slot in range(STEPS):
        state = env.get_state()
        action = policy.act(obs[None], [state])[0]
        executed = action if job["mode"] == "original" else _replace(
            action, job["mode"], job["constants"], state
        )
        if record and slot >= BURN_IN:
            aoi = np.asarray(state["aoi"], dtype=np.float64)
            slots.append({
                "slot": slot,
                "alpha": float(np.asarray(action["common_power_fraction"]).reshape(-1)[0]),
                "q": np.asarray(action["private_power_fractions"], dtype=np.float64).copy(),
                "phi": np.asarray(action["common_payload_fractions"], dtype=np.float64).copy(),
                "aoi": aoi,
                "alignment": _alignment(np.asarray(state["h_hat"])),
            })
        obs, _, terminated, truncated, info = env.step(executed)
        if slot >= BURN_IN:
            ages.extend(np.asarray(info["aoi"], dtype=float).tolist())
            success.extend(np.asarray(info["success_probabilities"], dtype=float).tolist())
            common_bits.append(float(info["common_payload"]))
            alphas.append(float(np.asarray(executed["common_power_fraction"]).reshape(-1)[0]))
        if terminated or truncated:
            break
    env.close()
    age = np.asarray(ages)
    return {
        "label": job["label"], "mode": job["mode"], "training_seed": job["training_seed"],
        "eval_repeat": job["eval_repeat"], "permutation": job["permutation"],
        "eval_seed": job["eval_seed"], "mean_aoi": float(age.mean()),
        "p95_aoi": float(np.quantile(age, 0.95)),
        "success_probability": float(np.mean(success)),
        "alpha_mean": float(np.mean(alphas)), "alpha_std": float(np.std(alphas)),
        "common_payload_mean": float(np.mean(common_bits)),
        "slots": slots,
    }


def run_rule(job: dict) -> dict:
    result = evaluate_rule({
        "config": CONFIG, "kind": job["kind"], "phase": "test", "alpha": job["alpha"],
        "permutation": job["permutation"], "angles": job["angles"],
        "geometry_deg": 20, "snr_db": 15.0, "blocklength": 128,
        "steps": STEPS, "burn_in": BURN_IN, "eval_seed": job["eval_seed"],
    })
    return {
        "label": job["label"], "mode": "original", "training_seed": -1,
        "eval_repeat": job["eval_repeat"], "permutation": job["permutation"],
        "eval_seed": job["eval_seed"], "mean_aoi": result["mean_aoi"],
        "p95_aoi": result["p95_aoi"], "success_probability": result["success_probability"],
        "alpha_mean": float(result["alpha"]), "alpha_std": 0.0,
        "common_payload_mean": result["common_payload_mean"], "slots": [],
    }


def dispatch(job: dict) -> dict:
    if job["family"] == "rule":
        return run_rule(job)
    if job["family"] == "calibration":
        return calibrate(job)
    return run_learned(job)


def ci95(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    return float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def summarise_over(frame: pd.DataFrame, unit: str, value: str) -> dict:
    per_unit = frame.groupby(unit)[value].mean()
    return {"mean": float(per_unit.mean()), "ci95": ci95(per_unit), "sd": float(per_unit.std(ddof=1)),
            "n": int(len(per_unit))}


def role_statistics(slots: list[dict]) -> dict:
    alpha = np.array([s["alpha"] for s in slots])
    q = np.stack([s["q"] for s in slots])
    phi = np.stack([s["phi"] for s in slots])
    aoi = np.stack([s["aoi"] for s in slots])
    alignment = np.array([s["alignment"] for s in slots])
    relative = aoi / np.maximum(aoi.mean(axis=1, keepdims=True), 1e-12)
    top = aoi.max(axis=1, keepdims=True)
    unique = (aoi == top).sum(axis=1) == 1
    agree = np.argmax(q, axis=1) == np.argmax(aoi, axis=1)
    order = np.argsort(aoi, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(aoi.shape[1])[None, :].repeat(len(aoi), 0), axis=1)
    out = {
        "alpha_mean": float(alpha.mean()), "alpha_sd": float(alpha.std()),
        "alpha_p05": float(np.quantile(alpha, 0.05)), "alpha_p95": float(np.quantile(alpha, 0.95)),
        "q_relative_aoi_pearson": float(pearsonr(q.ravel(), relative.ravel()).statistic),
        "q_argmax_is_oldest_unique_pct": float(100.0 * agree[unique].mean()),
        "unique_oldest_state_pct": float(100.0 * unique.mean()),
        "phi_mean": float(phi.mean()),
        "phi_alignment_spearman": float(spearmanr(phi.mean(axis=1), alignment).statistic),
        "phi_alignment_pearson": float(pearsonr(phi.mean(axis=1), alignment).statistic),
        "alpha_alignment_spearman": float(spearmanr(alpha, alignment).statistic),
    }
    for r, name in enumerate(("youngest", "middle", "oldest")):
        mask = rank == r
        out[f"q_{name}"] = float(q[mask].mean())
        out[f"phi_{name}"] = float(phi[mask].mean())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-dir", default="results/kics_2m_5train_5eval_v1")
    ap.add_argument("--skip-counterfactuals", action="store_true")
    args = ap.parse_args()
    out = ROOT / args.out_dir
    if out.exists():
        raise FileExistsError(out)

    for label, template in LEARNED.items():
        for seed in TRAINING_SEEDS:
            if not (ROOT / template.format(seed=seed)).is_file():
                raise FileNotFoundError(template.format(seed=seed))
    out.mkdir(parents=True)

    jobs = []
    for repeat in EVAL_REPEATS:
        for permutation, angles in enumerate(ANGLE_SETS):
            common = {"angles": angles, "permutation": permutation, "eval_repeat": repeat,
                      "eval_seed": eval_seed(repeat, permutation)}
            for label, template in LEARNED.items():
                for seed in TRAINING_SEEDS:
                    jobs.append({**common, "family": "ppo", "label": label, "mode": "original",
                                 "training_seed": seed, "checkpoint": template.format(seed=seed),
                                 "record_actions": label == ROLE_POLICY})
            for label, (kind, alpha) in RULES.items():
                jobs.append({**common, "family": "rule", "label": label, "kind": kind, "alpha": alpha})

    calibration = {}
    if not args.skip_counterfactuals:
        cal_jobs = []
        for seed in TRAINING_SEEDS:
            for permutation, angles in enumerate(ANGLE_SETS):
                cal_jobs.append({
                    "family": "calibration", "policy": seed,
                    "checkpoint": LEARNED[ROLE_POLICY].format(seed=seed), "config": CONFIG,
                    "permutation": permutation, "angles": angles, "steps": STEPS,
                    "burn_in": BURN_IN, "seed": 980278 + 100 * permutation,
                })
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            cal_rows = list(pool.map(dispatch, cal_jobs))
        pd.DataFrame(cal_rows).to_csv(out / "calibration_constants.csv", index=False)
        calibration = {(r["policy"], r["permutation"]): r for r in cal_rows}
        for repeat in EVAL_REPEATS:
            for permutation, angles in enumerate(ANGLE_SETS):
                for seed in TRAINING_SEEDS:
                    for mode in COUNTERFACTUALS:
                        jobs.append({
                            "family": "ppo", "label": ROLE_POLICY, "mode": mode,
                            "training_seed": seed, "checkpoint": LEARNED[ROLE_POLICY].format(seed=seed),
                            "angles": angles, "permutation": permutation, "eval_repeat": repeat,
                            "eval_seed": eval_seed(repeat, permutation), "record_actions": False,
                            "constants": calibration[(seed, permutation)],
                        })

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(dispatch, jobs, chunksize=2))

    role_rows = []
    for seed in TRAINING_SEEDS:
        slots = [s for r in rows if r["label"] == ROLE_POLICY and r["mode"] == "original"
                 and r["training_seed"] == seed for s in r["slots"]]
        role_rows.append({"training_seed": seed, "slots": len(slots), **role_statistics(slots)})
    roles = pd.DataFrame(role_rows)
    roles.to_csv(out / "action_roles_by_training_seed.csv", index=False)
    role_summary = pd.DataFrame([
        {"statistic": c, "mean": roles[c].mean(), "sd_over_training_seeds": roles[c].std(ddof=1),
         "ci95_halfwidth": ci95(roles[c]), "min": roles[c].min(), "max": roles[c].max()}
        for c in roles.columns if c not in ("training_seed", "slots")
    ])
    role_summary.to_csv(out / "action_roles_summary.csv", index=False)

    traj = pd.DataFrame([{k: v for k, v in r.items() if k != "slots"} for r in rows])
    traj.to_csv(out / "target_trajectories.csv", index=False)
    original = traj[traj["mode"] == "original"]

    metrics = ["mean_aoi", "p95_aoi", "success_probability", "alpha_mean", "common_payload_mean"]
    repeat_means = original.groupby(["label", "training_seed", "eval_repeat"], as_index=False)[metrics].mean()
    repeat_means.to_csv(out / "training_seed_eval_seed_means.csv", index=False)
    seed_means = repeat_means.groupby(["label", "training_seed"], as_index=False)[metrics].mean()
    seed_means.to_csv(out / "training_seed_means.csv", index=False)

    agg_rows = []
    for label, group in seed_means.groupby("label"):
        learned = label in LEARNED
        source = group if learned else repeat_means[repeat_means.label == label]
        row = {"label": label, "ci_unit": "training_seed" if learned else "evaluation_seed",
               "n_units": len(source)}
        for m in metrics:
            row[m] = float(source[m].mean())
            row[f"{m}_ci95"] = ci95(source[m])
            row[f"{m}_sd"] = float(source[m].std(ddof=1))
        agg_rows.append(row)
    aggregate = pd.DataFrame(agg_rows)
    aggregate.to_csv(out / "target_aggregate.csv", index=False)

    # Evaluation-seed variation: average each evaluation seed over training seeds.
    eval_var = repeat_means.groupby(["label", "eval_repeat"], as_index=False)["mean_aoi"].mean()
    eval_var.to_csv(out / "evaluation_seed_means.csv", index=False)
    eval_summary = eval_var.groupby("label")["mean_aoi"].agg(["mean", "std", "min", "max"]).reset_index()
    eval_summary["ci95_over_evaluation_seeds"] = eval_summary["label"].map(
        lambda lab: ci95(eval_var.loc[eval_var.label == lab, "mean_aoi"]))
    eval_summary.to_csv(out / "evaluation_seed_variation.csv", index=False)

    wide = repeat_means.pivot_table(index=["training_seed", "eval_repeat"], columns="label",
                                    values="mean_aoi").reset_index()
    rules = repeat_means[repeat_means.training_seed == -1].pivot_table(
        index="eval_repeat", columns="label", values="mean_aoi")
    learned_wide = wide[wide.training_seed >= 0].drop(columns=[c for c in RULES if c in wide.columns])
    learned_wide = learned_wide.merge(rules.reset_index(), on="eval_repeat")
    comparisons = [
        ("rsma_ppo_2m", "sdma_ppo_2m"), ("rsma_ppo_2m", "fixed_rule_rsma"),
        ("rsma_ppo_2m", "max_aoi_single_user"), ("rsma_ppo_2m", "rsma_ppo_524k"),
        ("sdma_ppo_2m", "sdma_ppo_524k"), ("rsma_ppo_524k", "sdma_ppo_524k"),
        ("sdma_ppo_2m", "fixed_rule_rsma"),
    ]
    paired_rows = []
    per_pair = []
    for method, baseline in comparisons:
        gain = 100.0 * (learned_wide[baseline] - learned_wide[method]) / learned_wide[baseline]
        diff = learned_wide[baseline] - learned_wide[method]
        frame = pd.DataFrame({"training_seed": learned_wide.training_seed,
                              "eval_repeat": learned_wide.eval_repeat, "gain_pct": gain, "diff": diff})
        frame["comparison"] = f"{method}_vs_{baseline}"
        per_pair.append(frame)
        by_seed = frame.groupby("training_seed")[["gain_pct", "diff"]].mean()
        paired_rows.append({
            "comparison": f"{method}_vs_{baseline}",
            "gain_pct": float(by_seed.gain_pct.mean()), "gain_ci95_training_seed": ci95(by_seed.gain_pct),
            "abs_diff": float(by_seed["diff"].mean()), "abs_diff_ci95_training_seed": ci95(by_seed["diff"]),
            "training_seed_wins": int((by_seed.gain_pct > 0).sum()), "training_seeds": len(by_seed),
            "seed_eval_pair_wins": int((frame.gain_pct > 0).sum()), "seed_eval_pairs": len(frame),
        })
    traj_learned = original[original.training_seed >= 0].pivot_table(
        index=["training_seed", "eval_repeat", "permutation"], columns="label", values="mean_aoi")
    traj_rules = original[original.training_seed == -1].pivot_table(
        index=["eval_repeat", "permutation"], columns="label", values="mean_aoi")
    traj_all = traj_learned.reset_index().merge(traj_rules.reset_index(), on=["eval_repeat", "permutation"])
    for row in paired_rows:
        method, baseline = row["comparison"].split("_vs_")
        row["trajectory_wins"] = int((traj_all[method] < traj_all[baseline]).sum())
        row["trajectories"] = int(len(traj_all))
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out / "paired_summary.csv", index=False)
    pd.concat(per_pair).to_csv(out / "paired_seed_eval_pairs.csv", index=False)

    if not args.skip_counterfactuals:
        cf = traj[traj.label == ROLE_POLICY].pivot_table(
            index=["training_seed", "eval_repeat", "permutation"], columns="mode", values="mean_aoi")
        cf_rows = []
        for mode in COUNTERFACTUALS:
            change = 100.0 * (cf[mode] - cf["original"]) / cf["original"]
            per_seed = change.groupby(level="training_seed").mean()
            cf_rows.append({"mode": mode, "aoi_increase_pct": float(per_seed.mean()),
                            "ci95_training_seed": ci95(per_seed), "min_seed": float(per_seed.min()),
                            "max_seed": float(per_seed.max()),
                            "training_seeds_worse": int((per_seed > 0).sum())})
        pd.DataFrame(cf_rows).to_csv(out / "counterfactual_summary.csv", index=False)
        print("Counterfactuals:\n", pd.DataFrame(cf_rows).to_string(index=False))

    print("Aggregate:\n", aggregate[["label", "ci_unit", "mean_aoi", "mean_aoi_ci95", "p95_aoi",
                                     "success_probability", "alpha_mean"]].to_string(index=False))
    print("\nPaired:\n", paired.to_string(index=False))
    print("\nEvaluation-seed variation:\n", eval_summary.to_string(index=False))
    print("\nAction roles:\n", role_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
