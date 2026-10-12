#!/usr/bin/env python3
"""Audit reward convergence and state-dependent direct PPO actions, without CEM."""
from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rsma_aoi.agents.ppo import PPOPolicy
from rsma_aoi.envs.aoi_env import AoIRSMAEnv


def corr(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def alignment(h: np.ndarray) -> tuple[float, float]:
    values = []
    for i, j in combinations(range(h.shape[0]), 2):
        denom = np.vdot(h[i], h[i]).real * np.vdot(h[j], h[j]).real
        values.append(float(abs(np.vdot(h[i], h[j])) ** 2 / max(denom, 1e-30)))
    return float(np.mean(values)), float(np.max(values))


def collect(checkpoint: Path, eval_seeds: list[int], steps: int, burn_in: int):
    policy, trained_cfg, _ = PPOPolicy.from_checkpoint(checkpoint, deterministic=True)
    cfg = replace(trained_cfg, episode_length=steps)
    rows = []
    for eval_seed in eval_seeds:
        env = AoIRSMAEnv(cfg)
        obs, _ = env.reset(seed=eval_seed)
        for slot in range(steps):
            state = env.get_state()
            h_hat = np.asarray(state["h_hat"])
            aoi = np.asarray(state["aoi"], float)
            align_mean, align_max = alignment(h_hat)
            gain = np.sum(np.abs(h_hat) ** 2, axis=1)
            action = policy.act(obs[None], [state])[0]
            alpha = float(action["common_power_fraction"][0])
            q = np.asarray(action["private_power_fractions"], float)
            phi = np.asarray(action["common_payload_fractions"], float)
            obs, reward, terminated, truncated, info = env.step(action)
            if slot >= burn_in:
                row = {
                    "checkpoint": checkpoint.parent.name,
                    "training_seed": int(checkpoint.parent.name.split("seed")[1].split("_")[0]),
                    "eval_seed": eval_seed,
                    "slot": slot,
                    "alignment_mean": align_mean,
                    "alignment_max": align_max,
                    "aoi_mean": float(aoi.mean()),
                    "aoi_spread": float(aoi.max() - aoi.min()),
                    "alpha": alpha,
                    "reward": float(reward),
                    "mean_aoi_next": float(info["mean_aoi"]),
                    "common_on": float(alpha > 1e-12 and info["common_payload"] > 1e-12),
                }
                for k in range(cfg.num_users):
                    row.update({
                        f"aoi_{k}": aoi[k], f"gain_{k}": gain[k],
                        f"q_{k}": q[k], f"phi_{k}": phi[k],
                    })
                rows.append(row)
            if terminated or truncated:
                break
        env.close()
    return pd.DataFrame(rows), cfg.num_users


def action_summary(trace: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, users = [], []
    for seed, frame in trace.groupby("training_seed"):
        aoi_flat = np.concatenate([frame[f"aoi_{u}"].to_numpy() for u in range(k)])
        gain_flat = np.concatenate([frame[f"gain_{u}"].to_numpy() for u in range(k)])
        q_flat = np.concatenate([frame[f"q_{u}"].to_numpy() for u in range(k)])
        phi_flat = np.concatenate([frame[f"phi_{u}"].to_numpy() for u in range(k)])
        summary.append({
            "training_seed": seed,
            "states": len(frame),
            "alpha_mean": frame.alpha.mean(),
            "alpha_std": frame.alpha.std(),
            "alpha_zero_rate": (frame.alpha == 0).mean(),
            "alpha_one_rate": (frame.alpha == 1).mean(),
            "common_on_rate": frame.common_on.mean(),
            "phi_mean": phi_flat.mean(),
            "phi_std": phi_flat.std(),
            "phi_zero_rate": (phi_flat == 0).mean(),
            "phi_one_rate": (phi_flat == 1).mean(),
            "q_std": q_flat.std(),
            "corr_alpha_alignment": corr(frame.alpha, frame.alignment_mean),
            "corr_alpha_aoi_mean": corr(frame.alpha, frame.aoi_mean),
            "corr_alpha_aoi_spread": corr(frame.alpha, frame.aoi_spread),
            "corr_q_own_aoi": corr(q_flat, aoi_flat),
            "corr_q_own_gain": corr(q_flat, gain_flat),
            "corr_phi_own_aoi": corr(phi_flat, aoi_flat),
            "corr_phi_own_gain": corr(phi_flat, gain_flat),
        })
        for user in range(k):
            users.append({
                "training_seed": seed, "user": user,
                "q_mean": frame[f"q_{user}"].mean(),
                "q_std": frame[f"q_{user}"].std(),
                "phi_mean": frame[f"phi_{user}"].mean(),
                "phi_std": frame[f"phi_{user}"].std(),
                "corr_q_aoi": corr(frame[f"q_{user}"], frame[f"aoi_{user}"]),
                "corr_q_gain": corr(frame[f"q_{user}"], frame[f"gain_{user}"]),
                "corr_phi_aoi": corr(frame[f"phi_{user}"], frame[f"aoi_{user}"]),
                "corr_phi_gain": corr(frame[f"phi_{user}"], frame[f"gain_{user}"]),
            })
    return pd.DataFrame(summary), pd.DataFrame(users)


def curve_summary(run_dirs: list[Path]) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    rows, curves = [], {}
    for run in run_dirs:
        seed = int(run.name.split("seed")[1].split("_")[0])
        data = pd.read_csv(run / "scalars.csv")
        data["rollout_aoi_ma32"] = data["env/mean_aoi"].rolling(32, min_periods=8).mean()
        data["episode_reward_per_slot"] = data["episode/r"] / 512.0
        data["episode_aoi"] = -data["episode_reward_per_slot"]
        curve = data.rollout_aoi_ma32.dropna()
        tail = data.iloc[int(0.75 * len(data)):]
        x = tail["perf/global_step"].to_numpy(float)
        y = tail["env/mean_aoi"].to_numpy(float)
        slope = float(np.polyfit(x, y, 1)[0] * 100_000)
        best_idx = int(curve.idxmin())
        rows.append({
            "training_seed": seed,
            "initial_10pct_aoi": data["env/mean_aoi"].iloc[: max(1, len(data)//10)].mean(),
            "final_10pct_aoi": data["env/mean_aoi"].iloc[-max(1, len(data)//10):].mean(),
            "best_ma32_aoi": float(curve.loc[best_idx]),
            "best_ma32_step": int(data.loc[best_idx, "perf/global_step"]),
            "final_ma32_aoi": float(curve.iloc[-1]),
            "final_minus_best_pct": float(100*(curve.iloc[-1]/curve.min()-1)),
            "tail_slope_aoi_per_100k": slope,
            "entropy_initial": data["train/entropy"].iloc[0],
            "entropy_final": data["train/entropy"].iloc[-1],
            "tail_kl_mean": tail["train/approx_kl"].mean(),
            "tail_explained_variance_mean": tail["train/explained_variance"].mean(),
        })
        curves[seed] = data
    return pd.DataFrame(rows), curves


def plot_curves(curves: dict[int, pd.DataFrame], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for seed, data in curves.items():
        x = data["perf/global_step"]
        axes[0].plot(x, data["rollout_aoi_ma32"], label=f"seed {seed}")
        axes[1].plot(x, data["episode_reward_per_slot"], label=f"seed {seed}")
    axes[0].set(title="Training rollout AoI (32-update moving mean)", xlabel="steps", ylabel="mean AoI")
    axes[1].set(title="Episode reward per slot (50-episode moving mean)", xlabel="steps", ylabel="reward")
    for ax in axes:
        ax.grid(alpha=.25); ax.legend()
    fig.tight_layout()
    fig.savefig(out / "ppo_reward_convergence.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[991000, 991001, 991002, 991003, 991004, 991005])
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--burn-in", type=int, default=256)
    parser.add_argument("--skip-curves", action="store_true")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    traces, k = [], None
    for checkpoint in map(Path, args.checkpoints):
        trace, users = collect(checkpoint, args.eval_seeds, args.steps, args.burn_in)
        traces.append(trace); k = users if k is None else k
    trace = pd.concat(traces, ignore_index=True)
    actions, users = action_summary(trace, int(k))
    trace.to_csv(out / "action_trace.csv.gz", index=False)
    actions.to_csv(out / "action_state_summary.csv", index=False)
    users.to_csv(out / "per_user_state_summary.csv", index=False)
    if not args.skip_curves:
        run_dirs = [p.parent for p in map(Path, args.checkpoints)]
        curves, curve_data = curve_summary(run_dirs)
        curves.to_csv(out / "learning_curve_summary.csv", index=False)
        plot_curves(curve_data, out)
        print("LEARNING CURVES")
        print(curves.to_string(index=False))
    print("\nACTION STATE DEPENDENCE")
    print(actions.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
