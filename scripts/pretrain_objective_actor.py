#!/usr/bin/env python3
"""Pretrain the PPO actor on the differentiable finite-blocklength AoI objective.

Unlike action-only behaviour cloning, this trains the PPO-compatible power,
common-rate, payload, and gate heads on their physical outcome.  The resulting
checkpoint is consumed directly by PPO's existing ``pretrained_actor`` path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pretrain_oracle import build_model  # noqa: E402
from rsma_aoi.agents.allocator_net import (  # noqa: E402
    LOSS_PRESETS, hybrid_actor_objective_losses, precompute_batch,
    sdma_actor_objective_losses,
)
from rsma_aoi.agents.imitation import (  # noqa: E402
    OracleDataset, dataset_meta, evaluate_imitation, save_imitation_checkpoint,
)
from rsma_aoi.envs.aoi_env import AoIRSMAEnv  # noqa: E402
from rsma_aoi.envs.channel import ChannelModel  # noqa: E402
from rsma_aoi.evaluation.oracle import config_hash  # noqa: E402
from rsma_aoi.utils.config import (  # noqa: E402
    apply_overrides_from_strings, dump_config_snapshot, load_run_configs,
)
from rsma_aoi.utils.logging import RunLogger, get_logger  # noqa: E402
from rsma_aoi.utils.seeding import seed_everything  # noqa: E402
from rsma_aoi.utils.action_mapping import DIRECT_ACTION_SCHEMA  # noqa: E402


def _physical_batch(
    data: OracleDataset, cfg, posterior_samples=1, seed=0, posterior_saa=False
):
    h_true = data.h_true
    if posterior_saa or posterior_samples > 1:
        channel = ChannelModel.from_config(cfg)
        rng = np.random.default_rng(seed)
        samples = []
        for index, h_hat in enumerate(data.h_hat):
            if data.user_angles_deg is not None:
                channel.user_angles_deg = data.user_angles_deg[index].copy()
            samples.append(channel.sample_true_given_estimate(h_hat, rng, posterior_samples))
        h_true = np.stack(samples)
    return precompute_batch(
        cfg,
        observation=data.observation.cpu().numpy(),
        aoi=data.aoi.cpu().numpy(),
        h_true=h_true,
        h_hat=data.h_hat,
        active=data.oracle_mask.cpu().numpy(),
        reference_power=data.oracle_power.cpu().numpy(),
        reference_rate=data.oracle_rate.cpu().numpy(),
        reference_rho=data.oracle_rho_r.cpu().numpy(),
        total_power=data.total_power,
        blocklength=data.blocklength,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--agent-config", default="configs/ppo_default.yaml")
    parser.add_argument("--preset", choices=sorted(LOSS_PRESETS), default="objective_only")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--posterior-samples", type=int, default=1,
        help="SAA samples from p(h_true|h_hat) per observed state",
    )
    parser.add_argument(
        "--posterior-saa", action="store_true",
        help="sample the posterior even when --posterior-samples=1",
    )
    parser.add_argument("--posterior-seed", type=int, default=10000)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    logger = get_logger()
    env_cfg, ppo_cfg = load_run_configs(args.env_config, args.agent_config)
    if args.set:
        ppo_cfg = apply_overrides_from_strings(ppo_cfg, args.set)
    if env_cfg.scheduling_enabled:
        parser.error("first objective-actor study requires scheduling_enabled=false")
    is_sdma = ppo_cfg.allocation_control == "sdma_power"
    if is_sdma and ppo_cfg.common_gate_enabled:
        parser.error("sdma_power objective pretraining requires common_gate_enabled=false")
    if is_sdma and args.preset != "objective_only":
        parser.error("sdma_power objective pretraining currently supports objective_only")
    if env_cfg.common_payload_mapping != "direct":
        parser.error("objective pretraining currently mirrors the direct payload mapping")
    if args.posterior_samples < 1:
        parser.error("posterior-samples must be positive")
    if (args.posterior_saa or args.posterior_samples > 1) and env_cfg.csit_error_mode != "relative":
        parser.error("posterior SAA currently supports relative CSIT only")
    seed_everything(args.seed)

    meta = dataset_meta(args.dataset)
    if meta.get("config_hash") and meta["config_hash"] != config_hash(env_cfg):
        parser.error("dataset and environment config hashes differ")
    if (
        env_cfg.rate_split_mode == "per_user_fraction"
        and meta.get("action_schema") != DIRECT_ACTION_SCHEMA
    ):
        parser.error(
            "direct-action pretraining requires a freshly generated "
            f"{DIRECT_ACTION_SCHEMA!r} dataset"
        )
    train = OracleDataset.load(Path(args.dataset) / "train.npz")
    val = OracleDataset.load(Path(args.dataset) / "val.npz")
    test = OracleDataset.load(Path(args.dataset) / "test.npz")

    probe = AoIRSMAEnv(env_cfg)
    obs_dim = int(np.prod(probe.observation_space.shape))
    probe.close()
    if train.observation.shape[1] != obs_dim:
        parser.error("dataset observation width does not match the environment")

    obs_rms = train.fit_normalizer() if ppo_cfg.normalize_observation else None
    if obs_rms is not None:
        for split in (train, val, test):
            split.normalize_(obs_rms)
    train_phy = _physical_batch(
        train, env_cfg, args.posterior_samples, args.posterior_seed, args.posterior_saa
    )
    val_phy = _physical_batch(
        val, env_cfg, args.posterior_samples, args.posterior_seed + 1, args.posterior_saa
    )
    model = build_model(env_cfg, ppo_cfg, obs_dim)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    weights = LOSS_PRESETS[args.preset]
    loss_fn = sdma_actor_objective_losses if is_sdma else hybrid_actor_objective_losses
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dump_config_snapshot(out / "config_snapshot.yaml", env=env_cfg, ppo=ppo_cfg)
    run_logger = RunLogger(out)
    best_score, best_path = float("inf"), out / "checkpoint.pt"

    for epoch in range(args.epochs):
        model.train()
        components: dict[str, list[float]] = {}
        order = rng.permutation(len(train_phy))
        for start in range(0, len(order), args.batch_size):
            idx = torch.as_tensor(order[start : start + args.batch_size])
            chunk = type(train_phy)(**{
                key: value[idx] for key, value in train_phy.__dict__.items()
            })
            loss, metrics = loss_fn(model, chunk, env_cfg, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite objective loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for key, value in metrics.items():
                components.setdefault(f"train/{key}", []).append(float(value))

        with torch.no_grad():
            _, val_loss = loss_fn(model, val_phy, env_cfg, weights)
        metrics = {key: float(np.mean(value)) for key, value in components.items()}
        metrics.update({f"val/{key}": float(value) for key, value in val_loss.items()})
        # Real-PHY hard-gate validation is the checkpoint criterion.
        metrics.update(evaluate_imitation(model, val, env_cfg, max_phy_states=5000))
        # Under imperfect CSIT, select on the same posterior-averaged physical
        # objective used for training.  The single-realisation real-PHY metric
        # remains logged as an independent held-out diagnostic.
        score = (
            metrics["val/expected_aoi"]
            if args.posterior_saa or args.posterior_samples > 1
            else metrics["val/predicted_expected_aoi"]
        )
        run_logger.log_scalars(epoch, metrics)
        if score < best_score:
            best_score = score
            save_imitation_checkpoint(
                best_path, model, env_cfg, ppo_cfg, metrics, meta,
                obs_rms=obs_rms,
                kind=("objective_actor_pretrain_sdma" if is_sdma else "objective_actor_pretrain"),
            )
        if model.direct_fraction and (epoch + 1) % max(1, args.epochs // 6) == 0:
            logger.info("epoch %d | selection AoI %.4f | common bits %.4f | regret %.4f",
                        epoch+1, score, metrics["val/common_bits"], metrics["val/regret"])
        elif (epoch + 1) % max(1, args.epochs // 6) == 0:
            logger.info(
                "epoch %d | selection AoI %.4f (single-realisation oracle %.4f) | gate %.3f | regret %.4f",
                epoch + 1, score, metrics["val/oracle_expected_aoi"],
                metrics["val/gate_probability"], metrics["val/regret"],
            )

    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate_imitation(model, test, env_cfg, max_phy_states=5000)
    run_logger.log_scalars(args.epochs, {
        key.replace("val/", "test/"): value for key, value in test_metrics.items()
    })
    run_logger.close()
    logger.info("best hard val AoI %.4f", best_score)
    for key, value in sorted(test_metrics.items()):
        logger.info("test %-32s %.4f", key.replace("val/", ""), value)
    logger.info("checkpoint %s", best_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
