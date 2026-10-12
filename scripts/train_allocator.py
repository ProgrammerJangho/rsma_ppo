#!/usr/bin/env python3
"""Compare action-imitation and objective-aware training of a learned allocator.

Behaviour cloning matches the reference *action*; under FBL the decoding
threshold makes that a poor proxy for the *outcome*.  This script trains the
same network under several loss mixtures and ranks them by **expected-AoI
regret against the optimiser reference** - not by action L1.

Examples:
    python scripts/train_allocator.py --dataset datasets/oracle_m2 --epochs 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsma_aoi.agents.allocator_net import (  # noqa: E402
    LOSS_PRESETS, AllocatorNet, allocator_losses, evaluate_allocator, precompute_batch,
)
from rsma_aoi.agents.imitation import OracleDataset  # noqa: E402
from rsma_aoi.utils.config import EnvConfig  # noqa: E402
from rsma_aoi.utils.logging import get_logger  # noqa: E402
from rsma_aoi.utils.seeding import seed_everything  # noqa: E402


def load_batch(path: Path, cfg: EnvConfig, limit: int | None):
    data = OracleDataset.load(path)
    n = len(data.aoi) if limit is None else min(limit, len(data.aoi))
    return precompute_batch(
        cfg,
        observation=data.observation[:n].numpy(),
        aoi=data.aoi[:n].numpy(),
        h_true=data.h_true[:n],
        h_hat=data.h_hat[:n],
        active=data.oracle_mask[:n].numpy(),
        reference_power=data.oracle_power[:n].numpy(),
        reference_rate=data.oracle_rate[:n].numpy(),
        reference_rho=data.oracle_rho_r[:n].numpy(),
        total_power=(None if data.total_power is None else data.total_power[:n]),
    )


def train_one(preset: str, train, val, test, cfg, args, logger) -> dict:
    seed_everything(args.seed)
    weights = LOSS_PRESETS[preset]
    model = AllocatorNet(train.features.shape[1], cfg.num_users, args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    best = {"expected_aoi_regret": float("inf")}
    best_state = None

    for epoch in range(args.epochs):
        order = rng.permutation(len(train))
        for start in range(0, len(train), args.batch_size):
            idx = torch.as_tensor(order[start : start + args.batch_size])
            chunk = type(train)(**{k: v[idx] for k, v in train.__dict__.items()})
            loss, _ = allocator_losses(model, chunk, cfg, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{preset}: non-finite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        metrics = evaluate_allocator(model, val, cfg)
        if metrics["expected_aoi_regret"] < best["expected_aoi_regret"]:
            best = metrics
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % max(1, args.epochs // 4) == 0:
            logger.info("  %-24s epoch %3d | val regret %.4f | val AoI %.4f",
                        preset, epoch + 1, metrics["expected_aoi_regret"],
                        metrics["expected_aoi"])

    model.load_state_dict(best_state)
    row = {"loss": preset, **{f"test_{k}": v for k, v in evaluate_allocator(model, test, cfg).items()}}
    row["val_regret"] = best["expected_aoi_regret"]
    return row, model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="datasets/oracle_m2")
    parser.add_argument("--config", default="configs/m2_hybrid.yaml")
    parser.add_argument("--presets", nargs="+", default=list(LOSS_PRESETS))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="results/allocator")
    args = parser.parse_args(argv)

    logger = get_logger()
    cfg = EnvConfig.from_yaml(args.config)
    root = Path(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("precomputing beam gains")
    train = load_batch(root / "train.npz", cfg, args.limit)
    val = load_batch(root / "val.npz", cfg, args.limit)
    test = load_batch(root / "test.npz", cfg, args.limit)
    logger.info("train/val/test = %d/%d/%d", len(train), len(val), len(test))

    rows = []
    for preset in args.presets:
        logger.info("training %s", preset)
        row, model = train_one(preset, train, val, test, cfg, args, logger)
        torch.save({"model": model.state_dict(), "preset": preset,
                    "env_config": cfg.to_dict(), "hidden": args.hidden},
                   out_dir / f"allocator_{preset}.pt")
        rows.append(row)

    # non-learned references on the same test states
    reference_rows = []
    for name, power_common, rho in (("equal_sdma", 0.0, 0.0), ("equal_rsma", 0.25, 0.3)):
        k = cfg.num_users
        active = test.active
        pf = torch.zeros(len(test), k + 1)
        pf[:, 0] = power_common
        counts = active.sum(-1, keepdim=True).clamp_min(1)
        pf[:, 1:] = active.float() * (1.0 - power_common) / counts
        rf = active.float() / counts
        rho_t = torch.full((len(test),), rho)

        from rsma_aoi.envs import torch_phy as tphy
        payload = torch.as_tensor(np.asarray(cfg.payload_vector), dtype=torch.float32)
        out = tphy.rsma_success_probability(
            test.gain_priv, test.gain_common, active, pf, rf, rho_t, payload,
            float(cfg.blocklength), cfg.noise_power, cfg.total_power,
        )
        gain = (out["success_prob"] * test.urgency).sum(-1)
        reference_rows.append({
            "loss": name,
            "test_expected_aoi_regret": float((test.reference_gain - gain).mean()),
            "test_expected_aoi": float((test.base_aoi - gain).mean()),
            "test_reference_expected_aoi": float((test.base_aoi - test.reference_gain).mean()),
        })

    df = pd.DataFrame(rows + reference_rows).sort_values("test_expected_aoi_regret")
    df.to_csv(out_dir / "allocator_comparison.csv", index=False)
    pd.set_option("display.width", 220)
    cols = [c for c in ("loss", "test_expected_aoi_regret", "test_expected_aoi",
                        "test_reference_expected_aoi", "test_success_regret",
                        "test_power_l1", "test_rate_l1", "test_rho_mae",
                        "test_threshold_crossing_rate", "test_feasibility_error")
            if c in df.columns]
    print("\n=== learned allocator: ranked by expected-AoI regret (primary metric) ===")
    print(df[cols].round(4).to_string(index=False))
    logger.info("results written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
