"""Behaviour cloning of the myopic oracle into the hybrid actor.

The oracle labels (see :mod:`rsma_aoi.evaluation.oracle`) are expressed in the
policy's own action space - a scheduling subset plus a power simplex, a
common-rate simplex and ``rho_r`` - so they can be regressed directly onto the
existing :class:`~rsma_aoi.agents.networks.HybridActorCritic` heads.

Losses
------
* scheduling: BCE on the oracle mask (Bernoulli head) or cross-entropy on the
  oracle subset index (categorical heads).  ``count_then_subset`` additionally
  gets an explicit count cross-entropy.
* power / rate: cross-entropy between the oracle simplex and the predicted
  masked softmax, computed **only over the active coordinates** (the inactive
  ones are exactly zero in both by construction).
* ``rho_r``: Beta negative log-likelihood (Beta head) or MSE on the squashed
  mean (squashed-Gaussian head).

Allocation losses are teacher-forced on the *oracle* mask, which is also what a
schedule-conditioned actor should receive.

Validation deliberately goes beyond supervised accuracy: the predicted action is
pushed through the real PHY and scored by its expected next weighted AoI, and
compared against the oracle's own objective.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from rsma_aoi.agents.distributions import masked_softmax
from rsma_aoi.agents.networks import HybridActorCritic
from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer
from rsma_aoi.evaluation.oracle import urgency_vector
from rsma_aoi.utils.config import EnvConfig

_EPS = 1e-8


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
@dataclass
class OracleDataset:
    """One split of an oracle dataset, held in memory as tensors."""

    observation: torch.Tensor          # (N, obs_dim)
    aoi: torch.Tensor                  # (N, K)
    oracle_mask: torch.Tensor          # (N, K) bool
    oracle_power: torch.Tensor         # (N, K+1)
    oracle_rate: torch.Tensor          # (N, K)
    oracle_rho_r: torch.Tensor         # (N,)
    oracle_expected_aoi: torch.Tensor  # (N,)
    objective_gap: torch.Tensor        # (N,)
    h_hat: np.ndarray                  # (N, K, Nt) complex
    h_true: np.ndarray                 # (N, K, Nt) complex - metadata only
    trajectory: np.ndarray
    source: np.ndarray
    config_hash: str
    oracle_common_gate: torch.Tensor | None = None  # (N,) bool; legacy data derives it
    snr_db: np.ndarray | None = None
    total_power: np.ndarray | None = None
    blocklength: np.ndarray | None = None
    user_angles_deg: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.observation.shape[0])

    @property
    def num_users(self) -> int:
        return int(self.oracle_mask.shape[1])

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "OracleDataset":
        """Load a ``.npz`` split written by ``scripts/generate_oracle_dataset.py``."""
        data = np.load(Path(path), allow_pickle=False)
        t = lambda k, dtype=torch.float32: torch.as_tensor(  # noqa: E731
            data[k], dtype=dtype, device=device
        )
        if all(k in data for k in (
            "oracle_common_power_fraction",
            "oracle_private_power_fractions",
            "oracle_common_payload_fractions",
        )):
            alpha = t("oracle_common_power_fraction").reshape(-1, 1)
            q = t("oracle_private_power_fractions")
            power = torch.cat([alpha, (1.0 - alpha) * q], dim=-1)
            rate = t("oracle_common_payload_fractions")
            # Compatibility placeholder only.  Direct actions carry no rho.
            rho = torch.ones(alpha.shape[0], dtype=alpha.dtype, device=alpha.device)
        else:
            power = t("oracle_power_fractions")
            rate = t("oracle_rate_fractions")
            rho = t("oracle_rho_r").reshape(-1)
        gate = (
            t("oracle_common_gate", torch.bool).reshape(-1)
            if "oracle_common_gate" in data
            else ((power[:, 0] > 1e-8) & (rho > 1e-8))
        )
        return cls(
            observation=t("observation"),
            aoi=t("aoi"),
            oracle_mask=t("oracle_mask", torch.bool),
            oracle_power=power,
            oracle_rate=rate,
            oracle_rho_r=rho,
            oracle_expected_aoi=t("oracle_expected_aoi").reshape(-1),
            objective_gap=t("objective_gap").reshape(-1),
            h_hat=data["h_hat_real"] + 1j * data["h_hat_imag"],
            h_true=data["h_true_real"] + 1j * data["h_true_imag"],
            trajectory=data["trajectory"],
            source=data["source"],
            config_hash=str(data["config_hash"]),
            oracle_common_gate=gate,
            snr_db=(data["episode_snr_db"].astype(np.float64) if "episode_snr_db" in data else None),
            total_power=(
                data["episode_total_power"].astype(np.float64)
                if "episode_total_power" in data else None
            ),
            blocklength=(
                data["episode_blocklength"].astype(np.int64)
                if "episode_blocklength" in data else None
            ),
            user_angles_deg=(
                data["episode_user_angles_deg"].astype(np.float64)
                if "episode_user_angles_deg" in data else None
            ),
        )

    def fit_normalizer(self):
        """Fit a :class:`RunningMeanStd` on this split's observations.

        PPO normalises observations, so the cloned actor must be trained on the
        *same* scale or the warm start silently sees a shifted input
        distribution.  The fitted normaliser travels with the checkpoint.
        """
        from rsma_aoi.utils.running_stats import RunningMeanStd

        rms = RunningMeanStd(shape=(self.observation.shape[1],))
        rms.update(self.observation.cpu().numpy())
        return rms

    def normalize_(self, rms) -> None:
        """Replace the stored observations with their normalised version."""
        self.observation = torch.as_tensor(
            rms.normalize(self.observation.cpu().numpy()),
            dtype=torch.float32, device=self.observation.device,
        )

    def batches(self, batch_size: int, rng: np.random.Generator, shuffle: bool = True):
        """Yield index tensors for minibatching."""
        order = rng.permutation(len(self)) if shuffle else np.arange(len(self))
        for start in range(0, len(order), batch_size):
            yield torch.as_tensor(order[start : start + batch_size], dtype=torch.long)


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def simplex_cross_entropy(
    target: torch.Tensor, logits: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """``-sum_i target_i log softmax_mask(logits)_i`` over the masked support.

    Both the target and the prediction are exactly zero outside ``mask``, so the
    sum runs over the active coordinates only.
    """
    pred = masked_softmax(logits, mask)
    logp = torch.log(pred.clamp_min(_EPS))
    return -(target * mask.to(target.dtype) * logp).sum(dim=-1)


def beta_nll(raw: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of a Beta head at ``value in (0, 1)``."""
    dim = raw.shape[-1] // 2
    alpha = 1.0 + F.softplus(raw[..., :dim])
    beta = 1.0 + F.softplus(raw[..., dim:])
    v = value.reshape(alpha.shape).clamp(1e-6, 1.0 - 1e-6)
    return -torch.distributions.Beta(alpha, beta).log_prob(v).sum(dim=-1)


def scheduling_loss(
    model: HybridActorCritic,
    schedule_logits: torch.Tensor,
    oracle_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Scheduling loss for whichever head the actor uses.

    Returns ``(loss_per_sample, extras)``; ``extras`` may carry the separate
    count loss of the hierarchical head.
    """
    head = model.scheduling_head
    target = oracle_mask.to(schedule_logits.dtype)
    if head in ("bernoulli", "fixed_top_m"):
        loss = F.binary_cross_entropy_with_logits(
            schedule_logits, target, reduction="none"
        ).sum(dim=-1)
        return loss, {}

    from rsma_aoi.agents.distributions import (
        CountThenSubset, SubsetCategorical, TopMCategorical,
    )

    if head == "subset_categorical":
        dist = SubsetCategorical(schedule_logits, model.num_users)
    elif head == "topm_categorical":
        dist = TopMCategorical(schedule_logits, model.num_users, model.schedule_size)
    elif head == "count_then_subset":
        n_count = len(model.schedule_sizes)
        dist = CountThenSubset(
            schedule_logits[:, :n_count], schedule_logits[:, n_count:],
            model.num_users, model.schedule_sizes,
        )
    else:  # pragma: no cover - guarded by the config validator
        raise ValueError(f"unsupported scheduling head for imitation: {head!r}")

    index = match_subset_index(dist.table, oracle_mask)
    loss = -dist.log_prob(index.to(schedule_logits.dtype).unsqueeze(-1))
    extras: dict[str, torch.Tensor] = {}
    if head == "count_then_subset":
        block = dist.size_of_index[index]
        extras["count"] = F.cross_entropy(dist.count_logits, block, reduction="none")
    return loss, extras


def match_subset_index(table: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Index of each row of ``mask`` in a subset ``table``.

    Rows whose subset is not representable by the head (e.g. an oracle subset of
    size 3 under a ``topm_categorical(m=2)`` head) fall back to the table entry
    with the largest overlap, so those states still supply a useful target.
    """
    m = mask.to(torch.float32)
    t = table.to(torch.float32)
    overlap = m @ t.T - 0.5 * (t.sum(dim=-1)[None, :] + m.sum(dim=-1)[:, None])
    return overlap.argmax(dim=-1)


@dataclass
class ImitationLossWeights:
    """Relative weights of the behaviour-cloning terms."""

    schedule: float = 1.0
    count: float = 0.5
    power: float = 1.0
    rate: float = 0.5
    rho: float = 0.2
    gate: float = 1.0


def imitation_loss(
    model: HybridActorCritic,
    batch_obs: torch.Tensor,
    oracle_mask: torch.Tensor,
    oracle_power: torch.Tensor,
    oracle_rate: torch.Tensor,
    oracle_rho: torch.Tensor,
    weights: ImitationLossWeights,
    oracle_common_gate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Total behaviour-cloning loss and its components.

    Allocation terms are teacher-forced on the oracle mask, so a
    schedule-conditioned actor sees the schedule it is being asked to serve.
    """
    if model.direct_fraction:
        from rsma_aoi.agents.allocator_net import hybrid_actor_allocations
        power, rate, _, _ = hybrid_actor_allocations(model, batch_obs, oracle_mask)
        loss_power = -(oracle_power * power.clamp_min(1e-8).log()).sum(-1).mean()
        loss_rate = (((rate-oracle_rate)**2)*oracle_mask).sum(-1).mean()
        sched = torch.zeros((), device=batch_obs.device)
        if model.scheduling_enabled:
            sched, _ = scheduling_loss(model, model.schedule_head(model.encoder(batch_obs)), oracle_mask)
            sched = sched.mean()
        total = weights.power*loss_power + weights.rate*loss_rate + weights.schedule*sched
        return total, {"loss/total": float(total.detach()), "loss/power": float(loss_power.detach()), "loss/rate": float(loss_rate.detach())}
    feat = model.encoder(batch_obs)
    schedule_logits = model.schedule_head(feat)
    params = (
        model._allocation_params(feat, oracle_mask)
        if model.conditional_allocation
        else {
            "power_mean": model.power_head(feat),
            "rate_mean": model.rate_head(feat),
            "rho_r_raw": model.rho_r_head(feat),
        }
    )

    if model.scheduling_enabled:
        sched, extras = scheduling_loss(model, schedule_logits, oracle_mask)
    else:
        sched = torch.zeros(oracle_mask.shape[0], device=oracle_mask.device)
        extras = {}
    if oracle_common_gate is None:
        gate_target = (oracle_power[:, 0] > _EPS) & (oracle_rho.reshape(-1) > _EPS)
    else:
        gate_target = oracle_common_gate.reshape(-1).to(torch.bool)
    common_mask = (
        gate_target[:, None]
        if model.common_gate_enabled
        else torch.ones(oracle_mask.shape[0], 1, dtype=torch.bool, device=oracle_mask.device)
    )
    power_mask = torch.cat([common_mask, oracle_mask], dim=-1)
    power = simplex_cross_entropy(oracle_power, params["power_mean"], power_mask)
    rate = simplex_cross_entropy(oracle_rate, params["rate_mean"], oracle_mask)
    if model.scalar_dist == "beta":
        rho = beta_nll(params["rho_r_raw"], oracle_rho)
    else:
        pred = torch.sigmoid(params["rho_r_raw"][..., :1]).reshape(-1)
        rho = (pred - oracle_rho.reshape(-1)) ** 2

    gate_float = gate_target.to(rate.dtype)
    gate_denom = gate_float.sum().clamp_min(1.0)
    gated_rate = (rate * gate_float).sum() / gate_denom
    gated_rho = (rho * gate_float).sum() / gate_denom
    total = (
        weights.schedule * sched.mean()
        + weights.power * power.mean()
        + weights.rate * gated_rate
        + weights.rho * gated_rho
    )
    components = {
        "loss/schedule": sched.mean().detach().item(),
        "loss/power": power.mean().detach().item(),
        "loss/rate": gated_rate.detach().item(),
        "loss/rho": gated_rho.detach().item(),
    }
    if model.common_gate_enabled:
        if model.common_gate_head is None:  # pragma: no cover - model invariant
            raise ValueError("common gate is enabled but the model has no gate head")
        gate_logits = model.common_gate_head(feat).reshape(-1)
        gate_loss = F.binary_cross_entropy_with_logits(
            gate_logits, gate_target.to(gate_logits.dtype)
        )
        total = total + weights.gate * gate_loss
        components["loss/gate"] = gate_loss.detach().item()
    if "count" in extras:
        total = total + weights.count * extras["count"].mean()
        components["loss/count"] = extras["count"].mean().detach().item()
    components["loss/total"] = total.detach().item()
    return total, components


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def fixed_top_m_masks(
    aoi: torch.Tensor, weights: np.ndarray, num_scheduled: int
) -> torch.Tensor:
    """Deterministic max-weighted-AoI-first masks, matching the trainer's rule."""
    priority = aoi * torch.as_tensor(weights, dtype=aoi.dtype, device=aoi.device)
    order = torch.argsort(priority, dim=-1, descending=True, stable=True)
    mask = torch.zeros_like(aoi, dtype=torch.bool)
    mask.scatter_(1, order[:, : max(1, num_scheduled)], True)
    return mask


@torch.no_grad()
def predict_actions(
    model: HybridActorCritic, obs: torch.Tensor, deterministic: bool = True,
    forced_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Greedy policy prediction: schedule, then allocation conditioned on it."""
    dist = model.distribution(obs, forced_mask=forced_mask)
    latent = dist.mode() if deterministic else dist.sample()
    action = dist.to_env_action(latent)
    # Compatibility-only storage for legacy dataset diagnostics, not an actor action.
    if model.direct_fraction:
        action["common_payload_scale"] = torch.ones_like(action["power_fractions"][:, :1])
    action["active_mask"] = latent.active_mask
    return action


@torch.no_grad()
def evaluate_imitation(
    model: HybridActorCritic,
    dataset: OracleDataset,
    env_cfg: EnvConfig,
    batch_size: int = 512,
    max_phy_states: int = 2000,
) -> dict[str, float]:
    """Supervised **and** physical validation metrics.

    Supervised: exact subset accuracy, per-user scheduling F1, active-count
    accuracy, power/rate L1, ``rho_r`` MAE.
    Physical: the predicted action is evaluated by the real PHY and reported as
    an expected next weighted AoI, alongside the oracle's own value.
    """
    model.eval()
    phy = RSMAPhysicalLayer(env_cfg)
    w = np.asarray(env_cfg.weight_vector, dtype=np.float64)
    aoi_max = float(env_cfg.aoi_max)

    exact, tp, fp, fn, count_ok, gate_ok, n = 0, 0, 0, 0, 0, 0, 0
    power_l1, rate_l1, rho_ae = 0.0, 0.0, 0.0
    pred_aoi, oracle_aoi = [], []

    for start in range(0, len(dataset), batch_size):
        sl = slice(start, start + batch_size)
        obs = dataset.observation[sl]
        target = dataset.oracle_mask[sl]
        forced = (
            fixed_top_m_masks(dataset.aoi[sl], w, model.schedule_size or 1)
            if model.scheduling_head == "fixed_top_m"
            else None
        )
        action = predict_actions(model, obs, forced_mask=forced)
        mask = action["active_mask"]
        predicted_gate = (
            (action["power_fractions"][:, 0] > _EPS)
            & (action["common_payload_scale"].reshape(-1) > _EPS)
        )
        target_gate = (
            dataset.oracle_common_gate[sl]
            if dataset.oracle_common_gate is not None
            else (
                (dataset.oracle_power[sl, 0] > _EPS)
                & (dataset.oracle_rho_r[sl] > _EPS)
            )
        )

        exact += int((mask == target).all(dim=-1).sum())
        tp += int((mask & target).sum())
        fp += int((mask & ~target).sum())
        fn += int((~mask & target).sum())
        count_ok += int((mask.sum(-1) == target.sum(-1)).sum())
        gate_ok += int((predicted_gate == target_gate).sum())
        n += int(mask.shape[0])

        power_l1 += float((action["power_fractions"] - dataset.oracle_power[sl]).abs().sum())
        rate_l1 += float((action["rate_fractions"] - dataset.oracle_rate[sl]).abs().sum())
        rho_ae += float(
            (action["common_payload_scale"].reshape(-1) - dataset.oracle_rho_r[sl]).abs().sum()
        )

        limit = max(0, min(max_phy_states - len(pred_aoi), int(mask.shape[0])))
        for i in range(limit):
            idx = start + i
            if dataset.total_power is not None:
                phy.total_power = float(dataset.total_power[idx])
            if dataset.blocklength is not None:
                phy.blocklength = int(dataset.blocklength[idx])
            urgency, base = urgency_vector(
                dataset.aoi[idx].cpu().numpy().astype(np.float64), w, aoi_max
            )
            out = phy.evaluate_allocations(
                dataset.h_true[idx],
                mask[i].cpu().numpy(),
                action["power_fractions"][i].cpu().numpy()[None, :],
                action["rate_fractions"][i].cpu().numpy()[None, :],
                np.asarray([float(action["common_payload_scale"][i])]),
                h_hat=dataset.h_hat[idx],
            )
            pred_aoi.append(base - float(out["success_prob"][0] @ urgency))
            oracle_aoi.append(float(dataset.oracle_expected_aoi[idx]))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    metrics = {
        "val/exact_subset_accuracy": exact / max(n, 1),
        "val/scheduling_precision": precision,
        "val/scheduling_recall": recall,
        "val/scheduling_f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "val/count_accuracy": count_ok / max(n, 1),
        "val/gate_accuracy": gate_ok / max(n, 1),
        "val/power_l1": power_l1 / max(n, 1),
        "val/rate_l1": rate_l1 / max(n, 1),
        "val/rho_mae": rho_ae / max(n, 1),
    }
    if pred_aoi:
        metrics["val/predicted_expected_aoi"] = float(np.mean(pred_aoi))
        metrics["val/oracle_expected_aoi"] = float(np.mean(oracle_aoi))
        metrics["val/objective_gap"] = float(np.mean(pred_aoi) - np.mean(oracle_aoi))
    model.train()
    if model.direct_fraction:
        metrics.pop("val/gate_accuracy", None)
        metrics.pop("val/rho_mae", None)
    return metrics


# ---------------------------------------------------------------------------
# probe set for forgetting diagnostics during PPO fine-tuning
# ---------------------------------------------------------------------------
class OracleProbe:
    """Fixed probe set used to track oracle agreement over PPO fine-tuning."""

    def __init__(self, dataset: OracleDataset, env_cfg: EnvConfig, size: int = 512) -> None:
        self.size = min(size, len(dataset))
        self.obs = dataset.observation[: self.size]
        self.mask = dataset.oracle_mask[: self.size]
        self.dataset = dataset
        self.env_cfg = env_cfg
        self.aoi = dataset.aoi[: self.size]
        self.weights = np.asarray(env_cfg.weight_vector, dtype=np.float64)

    @torch.no_grad()
    def evaluate(self, model: HybridActorCritic, obs_rms=None) -> dict[str, float]:
        """Subset agreement and the active-count distribution on the probe set."""
        obs = self.obs
        if obs_rms is not None:
            obs = torch.as_tensor(
                obs_rms.normalize(obs.cpu().numpy()), dtype=torch.float32, device=obs.device
            )
        forced = (
            fixed_top_m_masks(self.aoi, self.weights, model.schedule_size or 1)
            if model.scheduling_head == "fixed_top_m"
            else None
        )
        action = predict_actions(model, obs, forced_mask=forced)
        mask = action["active_mask"]
        counts = mask.sum(-1)
        out = {
            "probe/exact_subset_accuracy": float((mask == self.mask).all(dim=-1).float().mean()),
            "probe/count_accuracy": float((counts == self.mask.sum(-1)).float().mean()),
            "probe/mean_active": float(counts.float().mean()),
            "probe/common_power_fraction": float(action["power_fractions"][:, 0].mean()),
            "probe/rho_r": float(action["common_payload_scale"].mean()),
        }
        k = int(self.mask.shape[1])
        for c in range(0, k + 1):
            out[f"probe/p_count_{c}"] = float((counts == c).float().mean())
        return out


def save_imitation_checkpoint(
    path: str | Path,
    model: HybridActorCritic,
    env_cfg: EnvConfig,
    ppo_cfg: Any,
    metrics: dict[str, float],
    dataset_meta: dict[str, Any] | None = None,
    obs_rms: Any = None,
    kind: str = "imitation",
) -> Path:
    """Persist a behaviour-cloned actor together with its provenance.

    ``obs_rms`` is the observation normaliser the actor was trained under; PPO
    must start from it, otherwise the warm-started actor sees a different input
    scale than it was cloned on.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "env_config": env_cfg.to_dict(),
            "ppo_config": ppo_cfg.to_dict(),
            "metrics": metrics,
            "dataset_meta": dataset_meta or {},
            "obs_rms": None if obs_rms is None else obs_rms.state_dict(),
            "kind": kind,
        },
        path,
    )
    return path


def load_pretrained_actor(
    model: HybridActorCritic, path: str | Path, strict: bool = False
) -> dict[str, Any]:
    """Load actor weights from an imitation checkpoint into ``model``.

    The critic is intentionally *not* restored: the behaviour-cloned actor has no
    value function, so PPO must fit one from scratch.

    Returns the checkpoint's metadata dict.
    """
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=False)
    source_mode = ckpt.get("env_config", {}).get("rate_split_mode", "simplex_scaled")
    if source_mode != model.rate_split_mode:
        raise ValueError("pretrained actor action parameterization differs; explicit conversion/retraining required")
    state = ckpt["model"]
    actor_state = {
        k: v for k, v in state.items()
        if not k.startswith("critic_trunk") and not k.startswith("value_head")
    }
    missing, unexpected = model.load_state_dict(actor_state, strict=False)
    critic_keys = [k for k in missing if k.startswith(("critic_trunk", "value_head"))]
    other_missing = [k for k in missing if k not in critic_keys]
    if strict and (other_missing or unexpected):
        raise ValueError(
            f"pretrained actor does not match the model: missing={other_missing}, "
            f"unexpected={list(unexpected)}"
        )
    return {
        "metrics": ckpt.get("metrics", {}),
        "dataset_meta": ckpt.get("dataset_meta", {}),
        "obs_rms": ckpt.get("obs_rms"),
        "missing": other_missing,
        "unexpected": list(unexpected),
    }


def dataset_meta(path: str | Path) -> dict[str, Any]:
    """Read a dataset directory's ``meta.json`` (empty dict when absent)."""
    meta_path = Path(path) / "meta.json"
    if not meta_path.is_file():
        return {}
    with meta_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)
