"""Objective-aware learned allocator.

Behaviour cloning of a reference allocation minimises an **action** distance
(L1 / cross-entropy).  Under finite blocklength that is the wrong objective: the
BLER is nearly a step function of SINR, so two allocations a small L1 apart can
sit on opposite sides of a decoding threshold and differ enormously in AoI.

This module trains the same network against the *outcome* instead, using the
differentiable PHY in :mod:`rsma_aoi.envs.torch_phy`:

    L = lambda_action * L_action        # simplex cross-entropy + rho MSE
      + lambda_objective * L_objective  # -E[ sum_k urgency_k p_k ]
      + lambda_margin * L_margin        # hinge on negative FBL decoding margins

``L_regret = J(reference) - J(prediction)`` differs from ``L_objective`` only by
a per-sample constant, so it has the same gradient; it is reported as the
selection **metric** while ``L_objective`` supplies the gradient.

The network emits latents and maps them through the project's single
constraint-preserving transform, so its output is always feasible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rsma_aoi.agents.distributions import make_scalar_distribution, scalar_to_unit
from rsma_aoi.agents.networks import layer_init, make_mlp
from rsma_aoi.envs import torch_phy as tphy
from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer
from rsma_aoi.utils.action_mapping import hard_sigmoid_torch
from rsma_aoi.utils.config import EnvConfig

_EPS = 1e-8


class AllocatorNet(nn.Module):
    """Maps ``(features, active mask)`` to a feasible RSMA allocation.

    Output latents are ``[power (K+1) | rate (K) | rho_r (1)]``; the masked
    softmax and sigmoid make the action feasible by construction.
    """

    def __init__(
        self,
        feature_dim: int,
        num_users: int,
        hidden_sizes: list[int] | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.num_users = num_users
        hidden_sizes = hidden_sizes or [256, 256]
        self.trunk, feat = make_mlp(feature_dim + num_users, hidden_sizes, activation)
        self.head = layer_init(nn.Linear(feat, 2 * num_users + 2), std=0.01)

    def latents(self, features: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        joint = torch.cat([features, active.to(features.dtype)], dim=-1)
        return self.head(self.trunk(joint))

    def forward(
        self, features: torch.Tensor, active: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(power_fractions, rate_fractions, rho_r)``."""
        k = self.num_users
        latent = self.latents(features, active)
        ones = torch.ones(active.shape[0], 1, dtype=torch.bool, device=active.device)
        power_mask = torch.cat([ones, active], dim=-1)
        power = tphy.masked_softmax(latent[:, : k + 1], power_mask)
        rate = tphy.masked_softmax(latent[:, k + 1 : 2 * k + 1], active)
        rho = torch.sigmoid(latent[:, -1])
        return power, rate, rho


@dataclass
class AllocatorBatch:
    """Everything needed to score an allocation without touching NumPy."""

    features: torch.Tensor
    active: torch.Tensor
    gain_priv: torch.Tensor
    gain_common: torch.Tensor
    urgency: torch.Tensor
    base_aoi: torch.Tensor
    reference_power: torch.Tensor
    reference_rate: torch.Tensor
    reference_rho: torch.Tensor
    reference_gate: torch.Tensor
    reference_gain: torch.Tensor
    total_power: torch.Tensor
    blocklength: torch.Tensor

    def to(self, device: torch.device | str) -> "AllocatorBatch":
        return AllocatorBatch(**{k: v.to(device) for k, v in self.__dict__.items()})

    def __len__(self) -> int:
        return int(self.features.shape[0])


def precompute_batch(
    cfg: EnvConfig,
    observation: np.ndarray,
    aoi: np.ndarray,
    h_true: np.ndarray,
    h_hat: np.ndarray,
    active: np.ndarray,
    reference_power: np.ndarray,
    reference_rate: np.ndarray,
    reference_rho: np.ndarray,
    total_power: np.ndarray | None = None,
    blocklength: np.ndarray | None = None,
    dtype: torch.dtype = torch.float32,
) -> AllocatorBatch:
    """Precompute beam gains once so training never re-runs the beamformer.

    The beams depend only on ``(h_hat, active)``, so they are fixed for a given
    sample - which makes the whole objective differentiable in the allocation
    alone.
    """
    from rsma_aoi.evaluation.oracle import urgency_vector

    phy = RSMAPhysicalLayer(cfg)
    weights = np.asarray(cfg.weight_vector, dtype=np.float64)
    n, k = active.shape
    h_true = np.asarray(h_true)
    if h_true.ndim == 3:
        h_true = h_true[:, None, ...]
    if h_true.ndim != 4 or h_true.shape[0] != n or h_true.shape[2:] != (k, cfg.num_antennas):
        raise ValueError("h_true must have shape (N,K,Nt) or (N,S,K,Nt)")
    samples = h_true.shape[1]
    gain_priv = np.zeros((n, samples, k, k))
    gain_common = np.zeros((n, samples, k))
    urgency = np.zeros((n, k))
    base = np.zeros(n)
    ref_gain = np.zeros((n, samples))
    total_power_array = (
        np.full(n, cfg.total_power, dtype=np.float64)
        if total_power is None else np.asarray(total_power, dtype=np.float64).reshape(-1)
    )
    if total_power_array.shape != (n,) or np.any(total_power_array <= 0.0):
        raise ValueError("total_power must contain one positive value per state")
    blocklength_array = (
        np.full(n, cfg.blocklength, dtype=np.float64)
        if blocklength is None else np.asarray(blocklength, dtype=np.float64).reshape(-1)
    )
    if blocklength_array.shape != (n,) or np.any(blocklength_array < 1.0):
        raise ValueError("blocklength must contain one positive value per state")

    for i in range(n):
        phy.total_power = float(total_power_array[i])
        phy.blocklength = int(blocklength_array[i])
        mask = active[i].astype(bool)
        u_priv, u_common = phy.rsma_beams(h_hat[i], mask, use_common=True)
        urgency[i], base[i] = urgency_vector(aoi[i], weights, float(cfg.aoi_max))
        for s in range(samples):
            gain_priv[i, s] = np.abs(np.conj(h_true[i, s]) @ u_priv.T) ** 2
            gain_common[i, s] = np.abs(np.conj(h_true[i, s]) @ u_common) ** 2
            out = phy.evaluate_allocations(
                h_true[i, s], mask, reference_power[i][None, :], reference_rate[i][None, :],
                np.asarray([reference_rho[i]]), beams=(u_priv, u_common),
            )
            ref_gain[i, s] = float(out["success_prob"][0] @ urgency[i])

    tensor = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype)  # noqa: E731
    return AllocatorBatch(
        features=tensor(observation),
        active=torch.as_tensor(active.astype(bool)),
        gain_priv=tensor(gain_priv),
        gain_common=tensor(gain_common),
        urgency=tensor(urgency),
        base_aoi=tensor(base),
        reference_power=tensor(reference_power),
        reference_rate=tensor(reference_rate),
        reference_rho=tensor(reference_rho),
        reference_gate=torch.as_tensor(
            (np.asarray(reference_power)[:, 0] > _EPS)
            & (np.asarray(reference_rho).reshape(-1) > _EPS),
            dtype=torch.bool,
        ),
        reference_gain=tensor(ref_gain.mean(axis=1)),
        total_power=tensor(total_power_array),
        blocklength=tensor(blocklength_array),
    )


def _expand_over_channel_samples(
    batch: AllocatorBatch,
    power: torch.Tensor,
    rate: torch.Tensor,
    rho: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Broadcast one actor action over the posterior-channel sample axis."""
    samples = batch.gain_priv.shape[1]
    return (
        power[:, None, :].expand(-1, samples, -1).reshape(-1, power.shape[-1]),
        rate[:, None, :].expand(-1, samples, -1).reshape(-1, rate.shape[-1]),
        rho[:, None].expand(-1, samples).reshape(-1),
        batch.active[:, None, :].expand(-1, samples, -1).reshape(-1, batch.active.shape[-1]),
    )


def _sampled_phy_output(batch: AllocatorBatch, output: dict[str, torch.Tensor]):
    """Restore flattened PHY outputs to ``(state, sample, user)``."""
    n, samples = batch.gain_priv.shape[:2]
    return {key: value.reshape(n, samples, *value.shape[1:]) for key, value in output.items()}


def _expanded_total_power(batch: AllocatorBatch) -> torch.Tensor:
    """Repeat each state's power budget over posterior channel samples."""
    samples = batch.gain_priv.shape[1]
    return batch.total_power[:, None].expand(-1, samples).reshape(-1)


def _expanded_blocklength(batch: AllocatorBatch) -> torch.Tensor:
    """Repeat each state's blocklength over posterior channel samples."""
    samples = batch.gain_priv.shape[1]
    return batch.blocklength[:, None].expand(-1, samples).reshape(-1)


@dataclass
class LossWeights:
    """Relative weight of each loss term."""

    action: float = 1.0
    objective: float = 0.0
    margin: float = 0.0
    name: str = "action_only"


#: The loss configurations compared in the study.
LOSS_PRESETS: dict[str, LossWeights] = {
    "action_only": LossWeights(1.0, 0.0, 0.0, "action_only"),
    "objective_only": LossWeights(0.0, 1.0, 0.0, "objective_only"),
    "action_objective": LossWeights(0.5, 1.0, 0.0, "action_objective"),
    "action_objective_margin": LossWeights(0.5, 1.0, 0.05, "action_objective_margin"),
}


def simplex_cross_entropy(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Masked cross-entropy between two simplexes (inactive dims excluded)."""
    logp = torch.log(predicted.clamp_min(_EPS))
    return -(target * logp * mask.to(predicted.dtype)).sum(dim=-1)


def allocator_losses(
    model: AllocatorNet,
    batch: AllocatorBatch,
    cfg: EnvConfig,
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Total loss and its components for one batch."""
    if cfg.rate_split_mode != "simplex_scaled":
        raise ValueError("Use the PPO actor objective path for direct-fraction allocation")
    power, rate, rho = model(batch.features, batch.active)
    ones = torch.ones(batch.active.shape[0], 1, dtype=torch.bool, device=batch.active.device)
    power_mask = torch.cat([ones, batch.active], dim=-1)

    l_action = (
        simplex_cross_entropy(power, batch.reference_power, power_mask)
        + simplex_cross_entropy(rate, batch.reference_rate, batch.active)
        + (rho - batch.reference_rho) ** 2
    ).mean()

    payload = torch.as_tensor(
        np.asarray(cfg.payload_vector), dtype=power.dtype, device=power.device
    )
    power_phy, rate_phy, rho_phy, active_phy = _expand_over_channel_samples(
        batch, power, rate, rho
    )
    out = tphy.rsma_success_probability(
        batch.gain_priv.flatten(0, 1), batch.gain_common.flatten(0, 1),
        active_phy, power_phy, rate_phy, rho_phy,
        payload, _expanded_blocklength(batch), cfg.noise_power, _expanded_total_power(batch),
        use_common=True, rate_split_mode=cfg.rate_split_mode,
    )
    out = _sampled_phy_output(batch, out)
    active_phy = active_phy.reshape(batch.active.shape[0], batch.gain_priv.shape[1], -1)
    gain = (out["success_prob"] * batch.urgency[:, None, :]).sum(dim=-1).mean(dim=1)
    l_objective = -gain.mean()

    active_f = active_phy.to(power.dtype)
    negative_margin = (
        torch.relu(-out["private_margin"]) * active_f
        + torch.relu(-out["common_margin"]) * active_f
    )
    l_margin = negative_margin.sum(dim=-1).mean(dim=1).mean()

    total = (
        weights.action * l_action
        + weights.objective * l_objective
        + weights.margin * l_margin
    )
    metrics = {
        "loss_action": l_action.detach(),
        "loss_objective": l_objective.detach(),
        "loss_margin": l_margin.detach(),
        "gain": gain.mean().detach(),
        "regret": (batch.reference_gain - gain).mean().detach(),
        "expected_aoi": (batch.base_aoi - gain).mean().detach(),
        "success": (out["success_prob"].sum(dim=-1) / active_f.sum(dim=-1).clamp_min(1)).mean().detach(),
    }
    return total, metrics


def hybrid_actor_allocations(
    model,
    features: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable modal allocation of the PPO actor.

    The common gate is returned as a probability so objective pretraining can
    optimise the expected common-on/common-off outcome without applying the
    non-differentiable 0.5 threshold used by deterministic deployment.
    """
    if model.direct_fraction:
        feat = model.encoder(features)
        joint = torch.cat([feat, active.to(feat.dtype)], -1) if model.conditional_allocation else feat
        raw = model.power_head(joint)
        # Keep objective pretraining on exactly the same feasible-action map as
        # PPO deployment and CEM.  In particular, the clipped linear map can
        # represent the exact alpha=0 SDMA boundary without a separate gate.
        private = tphy.masked_softmax(raw[:, 1:], active)
        if model.allocation_control == "sdma_power":
            alpha = torch.zeros_like(raw[:, :1])
            rate = torch.zeros_like(model.rate_head(joint))
        else:
            alpha = hard_sigmoid_torch(raw[:, :1])
            rate = hard_sigmoid_torch(model.rate_head(joint)) * active
        power = torch.cat([alpha, (1-alpha)*private], -1)
        ones = torch.ones(features.shape[0], dtype=features.dtype, device=features.device)
        return power, rate, ones, ones
    feat = model.encoder(features)
    params = (
        model._allocation_params(feat, active)
        if model.conditional_allocation
        else {
            "power_mean": model.power_head(feat),
            "rate_mean": model.rate_head(feat),
            "rho_r_raw": model.rho_r_head(feat),
            "common_gate_logits": (
                None if model.common_gate_head is None else model.common_gate_head(feat)
            ),
        }
    )
    common = torch.ones(active.shape[0], 1, dtype=torch.bool, device=active.device)
    power = tphy.masked_softmax(
        params["power_mean"], torch.cat([common, active], dim=-1)
    )
    rate = tphy.masked_softmax(params["rate_mean"], active)
    rho_dist = make_scalar_distribution(params["rho_r_raw"], model.scalar_dist)
    rho_latent = rho_dist.mode()
    rho = scalar_to_unit(rho_latent, model.scalar_dist).reshape(-1)
    gate_probability = (
        torch.ones_like(rho)
        if params["common_gate_logits"] is None
        else torch.sigmoid(params["common_gate_logits"]).reshape(-1)
    )
    return power, rate, rho, gate_probability


def hybrid_actor_objective_losses(
    model,
    batch: AllocatorBatch,
    cfg: EnvConfig,
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train the PPO actor heads on differentiable expected next-slot AoI.

    For a learned common gate, the physical objective is the exact expectation
    over its Bernoulli on/off choice, evaluated at the modal continuous action.
    This keeps gradients for the gate while deployment can retain an exact
    common-off boundary.
    """
    power, rate, rho, gate_probability = hybrid_actor_allocations(
        model, batch.features, batch.active
    )
    payload = torch.as_tensor(
        np.asarray(cfg.payload_vector), dtype=power.dtype, device=power.device
    )
    power_phy, rate_phy, rho_phy, active_phy = _expand_over_channel_samples(
        batch, power, rate, rho
    )
    common = tphy.rsma_success_probability(
        batch.gain_priv.flatten(0, 1), batch.gain_common.flatten(0, 1),
        active_phy, power_phy, rate_phy, rho_phy,
        payload, _expanded_blocklength(batch), cfg.noise_power, _expanded_total_power(batch),
        use_common=True, rate_split_mode=cfg.rate_split_mode,
    )
    common = _sampled_phy_output(batch, common)
    boundary = tphy.rsma_success_probability(
        batch.gain_priv.flatten(0, 1), batch.gain_common.flatten(0, 1),
        active_phy, power_phy, rate_phy, rho_phy,
        payload, _expanded_blocklength(batch), cfg.noise_power, _expanded_total_power(batch),
        use_common=False, rate_split_mode=cfg.rate_split_mode,
    )
    boundary = _sampled_phy_output(batch, boundary)
    active_phy = active_phy.reshape(batch.active.shape[0], batch.gain_priv.shape[1], -1)
    urgency_phy = batch.urgency[:, None, :]
    gain_on = (common["success_prob"] * urgency_phy).sum(dim=-1).mean(dim=1)
    gain_off = (boundary["success_prob"] * urgency_phy).sum(dim=-1).mean(dim=1)
    gain = gate_probability * gain_on + (1.0 - gate_probability) * gain_off
    l_objective = -gain.mean()

    active = batch.active
    active_f = active_phy.to(power.dtype)
    power_mask = torch.cat([
        torch.ones(active.shape[0], 1, dtype=torch.bool, device=active.device), active
    ], dim=-1)
    gate_target = batch.reference_gate.to(power.dtype)
    gate_denom = gate_target.sum().clamp_min(1.0)
    l_power = simplex_cross_entropy(power, batch.reference_power, power_mask).mean()
    l_rate = (
        simplex_cross_entropy(rate, batch.reference_rate, active) * gate_target
    ).sum() / gate_denom
    l_rho = (((rho - batch.reference_rho) ** 2) * gate_target).sum() / gate_denom
    if model.common_gate_head is not None:
        l_gate = F.binary_cross_entropy(
            gate_probability.clamp(1e-7, 1.0 - 1e-7), gate_target
        )
    else:
        l_gate = torch.zeros((), dtype=power.dtype, device=power.device)
    if model.direct_fraction:
        l_rate = (((rate - batch.reference_rate)**2) * active).sum(-1).mean()
        l_rho = torch.zeros_like(l_rate)
    l_action = l_power + 0.5 * l_rate + 0.2 * l_rho + l_gate

    common_margin = (
        torch.relu(-common["private_margin"]) + torch.relu(-common["common_margin"])
    ) * active_f
    boundary_margin = torch.relu(-boundary["private_margin"]) * active_f
    l_margin = (
        gate_probability * common_margin.sum(dim=-1).mean(dim=1)
        + (1.0 - gate_probability) * boundary_margin.sum(dim=-1).mean(dim=1)
    ).mean()

    total = (
        weights.action * l_action
        + weights.objective * l_objective
        + weights.margin * l_margin
    )
    hard_gate = gate_probability >= 0.5
    hard_gain = torch.where(hard_gate, gain_on, gain_off)
    metrics = {
        "loss_action": l_action.detach(),
        "loss_objective": l_objective.detach(),
        "loss_margin": l_margin.detach(),
        "gain": gain.mean().detach(),
        "hard_gain": hard_gain.mean().detach(),
        "regret": (batch.reference_gain - hard_gain).mean().detach(),
        "expected_aoi": (batch.base_aoi - hard_gain).mean().detach(),
        "gate_probability": gate_probability.mean().detach(),
        "gate_accuracy": (hard_gate == batch.reference_gate).to(power.dtype).mean().detach(),
        "common_power": power[:, 0].mean().detach(),
        "rho": rho.mean().detach(),
    }
    if model.direct_fraction:
        metrics.pop("rho", None)
        metrics.pop("gate_probability", None)
        metrics.pop("gate_accuracy", None)
        metrics["common_fraction"] = rate.mean().detach()
        metrics["common_bits"] = common["common_payload"].mean().detach()
    return total, metrics


def sdma_actor_objective_losses(
    model,
    batch: AllocatorBatch,
    cfg: EnvConfig,
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train an SDMA-power actor on the differentiable FBL AoI objective.

    This is the matched common-off counterpart of
    :func:`hybrid_actor_objective_losses`.  The common-power coordinate and
    common payload are exactly zero, while the private powers form a simplex
    over active users.  Only ``objective_only`` is currently meaningful because
    the stored oracle actions belong to the larger RSMA action class.
    """
    if model.allocation_control != "sdma_power":
        raise ValueError("SDMA objective pretraining requires allocation_control='sdma_power'")
    if model.common_gate_enabled:
        raise ValueError("SDMA objective pretraining requires common_gate_enabled=false")
    if weights.action != 0.0 or weights.margin != 0.0:
        raise ValueError("SDMA objective pretraining currently supports objective_only")

    feat = model.encoder(batch.features)
    params = (
        model._allocation_params(feat, batch.active)
        if model.conditional_allocation
        else {"power_mean": model.power_head(feat)}
    )
    common_off = torch.zeros(
        batch.active.shape[0], 1, dtype=torch.bool, device=batch.active.device
    )
    power = tphy.masked_softmax(
        params["power_mean"], torch.cat([common_off, batch.active], dim=-1)
    )
    # These values are physically inactive under use_common=False, but keeping
    # their shapes explicit lets the shared PHY path enforce the same interface.
    active_f = batch.active.to(power.dtype)
    rate = active_f / active_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    rho = torch.zeros(batch.active.shape[0], dtype=power.dtype, device=power.device)

    payload = torch.as_tensor(
        np.asarray(cfg.payload_vector), dtype=power.dtype, device=power.device
    )
    power_phy, rate_phy, rho_phy, active_phy = _expand_over_channel_samples(
        batch, power, rate, rho
    )
    boundary = tphy.rsma_success_probability(
        batch.gain_priv.flatten(0, 1), batch.gain_common.flatten(0, 1),
        active_phy, power_phy, rate_phy, rho_phy,
        payload, _expanded_blocklength(batch), cfg.noise_power, _expanded_total_power(batch),
        use_common=False, rate_split_mode=cfg.rate_split_mode,
    )
    boundary = _sampled_phy_output(batch, boundary)
    active_samples = active_phy.reshape(
        batch.active.shape[0], batch.gain_priv.shape[1], -1
    )
    urgency = batch.urgency[:, None, :]
    gain = (boundary["success_prob"] * urgency).sum(dim=-1).mean(dim=1)
    l_objective = -gain.mean()
    total = weights.objective * l_objective

    metrics = {
        "loss_action": torch.zeros((), dtype=power.dtype, device=power.device),
        "loss_objective": l_objective.detach(),
        "loss_margin": torch.zeros((), dtype=power.dtype, device=power.device),
        "gain": gain.mean().detach(),
        # The reference may be an RSMA oracle; this is intentionally the gap to
        # that common action-class reference rather than an SDMA action target.
        "hard_gain": gain.mean().detach(),
        "regret": (batch.reference_gain - gain).mean().detach(),
        "expected_aoi": (batch.base_aoi - gain).mean().detach(),
        "gate_probability": torch.zeros((), dtype=power.dtype, device=power.device),
        "gate_accuracy": (~batch.reference_gate).to(power.dtype).mean().detach(),
        "common_power": power[:, 0].mean().detach(),
        "rho": rho.mean().detach(),
        "success": (
            boundary["success_prob"].sum(dim=-1)
            / active_samples.to(power.dtype).sum(dim=-1).clamp_min(1.0)
        ).mean().detach(),
    }
    if model.direct_fraction:
        metrics.pop("rho", None)
        metrics.pop("gate_probability", None)
        metrics.pop("gate_accuracy", None)
        metrics["common_fraction"] = torch.zeros(
            (), dtype=power.dtype, device=power.device
        )
        metrics["common_bits"] = torch.zeros(
            (), dtype=power.dtype, device=power.device
        )
    return total, metrics


@torch.no_grad()
def evaluate_allocator(
    model: AllocatorNet, batch: AllocatorBatch, cfg: EnvConfig
) -> dict[str, float]:
    """Selection metrics; the primary one is expected-AoI regret, not action L1."""
    model.eval()
    if cfg.rate_split_mode != "simplex_scaled":
        raise ValueError("Use the PPO actor objective path for direct-fraction allocation")
    power, rate, rho = model(batch.features, batch.active)
    payload = torch.as_tensor(
        np.asarray(cfg.payload_vector), dtype=power.dtype, device=power.device
    )
    power_phy, rate_phy, rho_phy, active_phy = _expand_over_channel_samples(
        batch, power, rate, rho
    )
    out = tphy.rsma_success_probability(
        batch.gain_priv.flatten(0, 1), batch.gain_common.flatten(0, 1),
        active_phy, power_phy, rate_phy, rho_phy,
        payload, _expanded_blocklength(batch), cfg.noise_power, _expanded_total_power(batch),
    )
    out = _sampled_phy_output(batch, out)
    active_phy = active_phy.reshape(batch.active.shape[0], batch.gain_priv.shape[1], -1)
    active_f = active_phy.to(power.dtype)
    gain = (out["success_prob"] * batch.urgency[:, None, :]).sum(dim=-1).mean(dim=1)
    regret = batch.reference_gain - gain

    reference = tphy.rsma_success_probability(
        batch.gain_priv.flatten(0, 1), batch.gain_common.flatten(0, 1),
        active_phy.flatten(0, 1),
        batch.reference_power[:, None, :].expand(-1, batch.gain_priv.shape[1], -1).flatten(0, 1),
        batch.reference_rate[:, None, :].expand(-1, batch.gain_priv.shape[1], -1).flatten(0, 1),
        batch.reference_rho[:, None].expand(-1, batch.gain_priv.shape[1]).reshape(-1),
        payload, _expanded_blocklength(batch), cfg.noise_power, _expanded_total_power(batch),
    )
    reference = _sampled_phy_output(batch, reference)
    predicted_success = ((out["success_prob"] * active_f).sum(-1) / active_f.sum(-1).clamp_min(1)).mean(1)
    reference_success = ((reference["success_prob"] * active_f).sum(-1) / active_f.sum(-1).clamp_min(1)).mean(1)
    crossing = (
        ((out["private_margin"] > 0) != (reference["private_margin"] > 0)).to(power.dtype)
        * active_f
    ).sum() / active_f.sum().clamp_min(1)

    ones = torch.ones(batch.active.shape[0], 1, dtype=torch.bool, device=batch.active.device)
    power_mask = torch.cat([ones, batch.active], dim=-1)
    model.train()
    return {
        "expected_aoi_regret": float(regret.mean()),
        "expected_aoi": float((batch.base_aoi - gain).mean()),
        "reference_expected_aoi": float((batch.base_aoi - batch.reference_gain).mean()),
        "success_regret": float((reference_success - predicted_success).mean()),
        "predicted_success": float(predicted_success.mean()),
        "power_l1": float((((power - batch.reference_power).abs()) * power_mask).sum(-1).mean()),
        "rate_l1": float((((rate - batch.reference_rate).abs()) * active_f).sum(-1).mean()),
        "rho_mae": float((rho - batch.reference_rho).abs().mean()),
        "threshold_crossing_rate": float(crossing),
        "feasibility_error": float(
            (power.sum(-1) - 1.0).abs().max() + (rate.sum(-1) - 1.0).abs().max()
        ),
        "inactive_leak": float(
            (power[:, 1:] * (~batch.active)).abs().max() + (rate * (~batch.active)).abs().max()
        ),
    }
