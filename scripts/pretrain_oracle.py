#!/usr/bin/env python3
"""Behaviour-clone the myopic oracle into a hybrid actor (curriculum stage 0).

The resulting checkpoint is a *warm start* for PPO, not a final policy: the
oracle is myopic, so imitating it places the policy in the multi-user basin
without claiming it is long-run optimal.

Examples:
    python scripts/pretrain_oracle.py --dataset datasets/oracle_m2 \
        --env-config configs/m2_hybrid.yaml --scheduling-head count_then_subset \
        --epochs 30 --out runs/bc_cts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsma_aoi.agents.imitation import (  # noqa: E402
    ImitationLossWeights, OracleDataset, dataset_meta, evaluate_imitation,
    imitation_loss, save_imitation_checkpoint,
)
from rsma_aoi.agents.networks import HybridActorCritic  # noqa: E402
from rsma_aoi.envs.aoi_env import AoIRSMAEnv  # noqa: E402
from rsma_aoi.evaluation.oracle import config_hash  # noqa: E402
from rsma_aoi.utils.config import (  # noqa: E402
    EnvConfig, PPOConfig, apply_overrides_from_strings, dump_config_snapshot, load_run_configs,
)
from rsma_aoi.utils.logging import RunLogger, get_logger  # noqa: E402
from rsma_aoi.utils.seeding import seed_everything  # noqa: E402


def resolve_schedule_spec(env_cfg: EnvConfig, ppo_cfg: PPOConfig):
    """Default ``schedule_size`` / ``schedule_sizes`` to the multiplexing limit."""
    limit = max(1, min(env_cfg.num_users, env_cfg.num_antennas))
    size = ppo_cfg.schedule_size or limit
    sizes = tuple(ppo_cfg.schedule_sizes or range(1, limit + 1))
    return size, sizes


def build_model(env_cfg: EnvConfig, ppo_cfg: PPOConfig, obs_dim: int) -> HybridActorCritic:
    size, sizes = resolve_schedule_spec(env_cfg, ppo_cfg)
    return HybridActorCritic(
        obs_dim=obs_dim,
        num_users=env_cfg.num_users,
        hidden_sizes=ppo_cfg.hidden_sizes,
        activation=ppo_cfg.activation,
        log_std_init=ppo_cfg.log_std_init,
        scalar_dist=ppo_cfg.scalar_dist,
        scheduling_enabled=env_cfg.scheduling_enabled,
        power_scale_action=env_cfg.power_scale_action,
        scheduling_head=ppo_cfg.scheduling_head,
        conditional_allocation=ppo_cfg.conditional_allocation,
        schedule_size=size,
        schedule_sizes=sizes,
        count_prior=ppo_cfg.count_prior,
        allocation_control=ppo_cfg.allocation_control,
        fixed_common_power_fraction=ppo_cfg.fixed_common_power_fraction,
        fixed_common_payload_scale=ppo_cfg.fixed_common_payload_scale,
        initial_common_power_fraction=ppo_cfg.initial_common_power_fraction,
        initial_common_payload_scale=ppo_cfg.effective_initial_common_fraction,
        initial_scalar_concentration=ppo_cfg.initial_scalar_concentration,
        common_gate_enabled=ppo_cfg.common_gate_enabled,
            rate_split_mode=env_cfg.rate_split_mode,
        initial_common_on_probability=ppo_cfg.initial_common_on_probability,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="datasets/oracle_m2")
    parser.add_argument("--env-config", default="configs/m2_hybrid.yaml")
    parser.add_argument("--agent-config", default="configs/ppo_default.yaml")
    parser.add_argument("--scheduling-head", default=None,
                        help="override ppo.scheduling_head for the cloned actor")
    parser.add_argument("--conditional-allocation", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lambda-schedule", type=float, default=1.0)
    parser.add_argument("--lambda-count", type=float, default=0.5)
    parser.add_argument("--lambda-power", type=float, default=1.0)
    parser.add_argument("--lambda-rate", type=float, default=0.5)
    parser.add_argument("--lambda-rho", type=float, default=0.2)
    parser.add_argument("--lambda-gate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/bc_oracle")
    parser.add_argument("--set", nargs="*", default=None, metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    logger = get_logger()
    env_cfg, ppo_cfg = load_run_configs(args.env_config, args.agent_config)
    if args.set:
        ppo_cfg = apply_overrides_from_strings(ppo_cfg, args.set)
    if args.scheduling_head:
        ppo_cfg = apply_overrides_from_strings(
            ppo_cfg, [f"scheduling_head={args.scheduling_head}"]
        )
    if args.conditional_allocation:
        ppo_cfg = apply_overrides_from_strings(ppo_cfg, ["conditional_allocation=true"])
    seed_everything(args.seed)

    meta = dataset_meta(args.dataset)
    expected = config_hash(env_cfg)
    if meta.get("config_hash") and meta["config_hash"] != expected:
        raise SystemExit(
            f"dataset was generated for env config hash {meta['config_hash']} but "
            f"{args.env_config} hashes to {expected}; regenerate the dataset or "
            "point --env-config at the config it was built from"
        )

    train = OracleDataset.load(Path(args.dataset) / "train.npz")
    val = OracleDataset.load(Path(args.dataset) / "val.npz")
    test = OracleDataset.load(Path(args.dataset) / "test.npz")
    logger.info("dataset: train %d | val %d | test %d (hash %s)",
                len(train), len(val), len(test), train.config_hash)

    probe_env = AoIRSMAEnv(env_cfg)
    obs_dim = int(np.prod(probe_env.observation_space.shape))
    probe_env.close()
    if train.observation.shape[1] != obs_dim:
        raise SystemExit(
            f"dataset observations are {train.observation.shape[1]}-dim but the env "
            f"produces {obs_dim}; the dataset and env config disagree"
        )

    # PPO normalises observations, so clone under the same scale and ship the
    # normaliser with the checkpoint
    obs_rms = train.fit_normalizer() if ppo_cfg.normalize_observation else None
    if obs_rms is not None:
        for split in (train, val, test):
            split.normalize_(obs_rms)
        logger.info("fitted observation normaliser on the training split")

    model = build_model(env_cfg, ppo_cfg, obs_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = ImitationLossWeights(
        schedule=args.lambda_schedule, count=args.lambda_count, power=args.lambda_power,
        rate=args.lambda_rate, rho=args.lambda_rho, gate=args.lambda_gate,
    )
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_logger = RunLogger(out_dir)
    dump_config_snapshot(out_dir / "config_snapshot.yaml", env=env_cfg, ppo=ppo_cfg)

    best_metric, best_path = float("inf"), out_dir / "checkpoint.pt"
    for epoch in range(args.epochs):
        model.train()
        epoch_components: dict[str, list[float]] = {}
        for idx in train.batches(args.batch_size, rng):
            loss, components = imitation_loss(
                model, train.observation[idx], train.oracle_mask[idx],
                train.oracle_power[idx], train.oracle_rate[idx],
                train.oracle_rho_r[idx], weights,
                None if train.oracle_common_gate is None else train.oracle_common_gate[idx],
            )
            if not torch.isfinite(loss):
                raise SystemExit("behaviour-cloning loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            for key, value in components.items():
                epoch_components.setdefault(key, []).append(value)

        metrics = {k: float(np.mean(v)) for k, v in epoch_components.items()}
        metrics.update(evaluate_imitation(model, val, env_cfg, max_phy_states=500))
        run_logger.log_scalars(epoch, metrics)
        logger.info(
            "epoch %2d | loss %.4f | subset acc %.3f | count acc %.3f | F1 %.3f | "
            "pred AoI %.3f (oracle %.3f)",
            epoch + 1, metrics["loss/total"], metrics["val/exact_subset_accuracy"],
            metrics["val/count_accuracy"], metrics["val/scheduling_f1"],
            metrics.get("val/predicted_expected_aoi", float("nan")),
            metrics.get("val/oracle_expected_aoi", float("nan")),
        )
        # selection criterion: the PHY-evaluated objective, not supervised accuracy
        score = metrics.get("val/predicted_expected_aoi", metrics["loss/total"])
        if score < best_metric:
            best_metric = score
            save_imitation_checkpoint(
                best_path, model, env_cfg, ppo_cfg, metrics, meta, obs_rms=obs_rms
            )

    # final test-set report using the best checkpoint
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate_imitation(model, test, env_cfg, max_phy_states=1000)
    run_logger.log_scalars(args.epochs, {k.replace("val/", "test/"): v
                                         for k, v in test_metrics.items()})
    run_logger.close()
    logger.info("BEST val predicted expected AoI: %.4f", best_metric)
    for key, value in sorted(test_metrics.items()):
        logger.info("  test %-32s %.4f", key.replace("val/", ""), value)
    logger.info("checkpoint: %s", best_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
