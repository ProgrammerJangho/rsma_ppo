#!/usr/bin/env python3
"""Static equal-private/equal-payload PHY landscape, NOT a long-run AoI optimum.

Independent perfect-CSIT channel samples. Full common code rate is K*D*rho/n
for equal payloads and c_k=1/K; this reveals poorly decodable initial rates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rsma_aoi.envs.channel import ChannelModel
from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer
from rsma_aoi.utils.config import EnvConfig, dump_config_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/thesis_ablation.yaml")
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    cfg = EnvConfig.from_yaml(args.config)
    if cfg.csit_error_std != 0 or cfg.multiple_access != "rsma":
        parser.error("this isolated probe requires perfect-CSIT RSMA")
    if args.channels < 1:
        parser.error("channels must be positive")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    active = np.ones(cfg.num_users, dtype=bool)
    power_axis = np.linspace(0, 1, 21)
    rho_axis = np.linspace(0, 1, 21)
    pairs = np.array([(p, r) for r in rho_axis for p in power_axis] + [(1/3, .5), (.25, .3)])
    pf = np.column_stack([pairs[:, 0], np.repeat(((1-pairs[:, 0])/cfg.num_users)[:, None], cfg.num_users, axis=1)])
    rf = np.full((len(pairs), cfg.num_users), 1/cfg.num_users)
    phy = RSMAPhysicalLayer(cfg)
    channel = ChannelModel.from_config(cfg)
    rng = np.random.default_rng(810000)
    success = np.zeros(len(pairs))
    common_error = np.zeros(len(pairs))
    for _ in range(args.channels):
        h = channel.sample_initial(rng)
        res = phy.evaluate_allocations(h, active, pf, rf, pairs[:, 1], h_hat=h)
        success += res["success_prob"].mean(axis=1)
        common_error += res["common_bler"].mean(axis=1)
    frame = pd.DataFrame({"common_power": pairs[:, 0], "rho": pairs[:, 1],
                          "mean_success_probability": success/args.channels,
                          "mean_common_bler": common_error/args.channels})
    frame.to_csv(out / "landscape.csv", index=False)
    dump_config_snapshot(out / "config_snapshot.yaml", env=cfg)
    (out / "manifest.json").write_text(json.dumps({
        "channels": args.channels, "channel_seed": 810000,
        "purpose": "independent-channel static PHY probe; equal private power and common shares",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in list((ROOT / "src").rglob("*.py")) + [Path(__file__).resolve()]},
    }, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
    im = ax.pcolormesh(power_axis, rho_axis, (success[:441]/args.channels).reshape(21, 21),
                       shading="nearest", vmin=0, vmax=1, cmap="viridis")
    ax.scatter([1/3, .25, 0], [.5, .3, 0], c=["red", "orange", "white"], edgecolors="black", s=65)
    for x, y, label in [(1/3, .5, "Initial modal action"), (.25, .3, "Fixed RSMA"), (0, 0, "SDMA boundary")]:
        ax.annotate(label, (x,y), xytext=(8,8), textcoords="offset points", color="white", fontsize=8)
    ax.set(xlabel="Common power fraction", ylabel="Common payload scale rho",
           title="Equal private powers and common payload shares\nMean packet success over independent channels")
    fig.colorbar(im, ax=ax, label="Packet success probability")
    fig.savefig(out / "landscape.png", dpi=180)
    plt.close(fig)
    print(frame.iloc[[-2, -1, 0]].to_string(index=False))
    print("Static PHY probe only; rates can fail, no long-run AoI optimality claim.")


if __name__ == "__main__":
    main()
