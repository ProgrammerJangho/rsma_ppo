"""Constraint-preserving action distributions for the hybrid RSMA policy.

Design (this is the part that makes the PPO objective correct)
--------------------------------------------------------------
The environment needs private-power simplexes and bounded allocation variables.
PPO stores and scores the sampled latent variables, then applies deterministic
feasibility transforms.  Old and new log probabilities therefore always refer
to the same random variable even when the action transform is non-injective.

``MaskedLogisticNormal``
    The policy is defined **on the latent space**: ``z ~ N(mu, sigma)`` over the
    unmasked coordinates, and the environment action is the deterministic,
    parameter-free transform ``y = masked_softmax(z)``.  PPO uses
    ``log pi(z)``; the ratio ``pi_new(z)/pi_old(z)`` is evaluated on the *same
    stored latent* ``z`` and the *same stored mask*, so it is exact.
    A density directly on the simplex is deliberately *not* used: the softmax is
    not injective (``z`` and ``z + c1`` give the same ``y``), so a
    full-dimensional Gaussian has no well-defined push-forward density there.
    Because the transform carries no learnable parameters, the latent-space
    policy gradient is the gradient of the induced action distribution.

``hard_sigmoid`` (canonical direct action)
    The latent Gaussian variables for ``alpha`` and ``phi_k`` are mapped by
    ``clip(1/2 + z/6, 0, 1)``.  Unlike a logistic sigmoid, this deterministic
    transform includes the exact SDMA and fully-common endpoints.  No action
    density is claimed after clipping: PPO scores the stored Gaussian latent,
    just as it does for the non-injective softmax transform above.

``BetaDistribution`` (default for scalars)
    Natural support ``(0, 1)``; ``alpha, beta = 1 + softplus(.) >= 1`` keeps the
    density unimodal and finite at the boundaries.  The "latent" *is* the action.

``SquashedGaussianDistribution`` (alternative)
    ``z ~ N(mu, sigma)``, ``y = sigmoid(z)``.  ``log_prob`` returns the density
    of ``y`` **including** the ``-log|dy/dz|`` Jacobian term.  Because ``z`` is
    what gets stored and re-scored, the Jacobian is identical for the old and
    new policy and cancels inside the PPO ratio - the result is the same as
    scoring the latent, but the reported log-probability is a proper density.

``BernoulliSchedule``
    Independent Bernoulli per user.  ``log_prob`` scores the **raw** sample; the
    "at least one active user" repair is a deterministic post-transform whose
    output (``active_mask``) is stored so that the masked heads can be rebuilt
    bit-identically at update time.

    KNOWN FAILURE MODE (measured, see the README): the repair creates a
    degenerate attractor.  Driving every probability towards 0 *maximises* the
    log-probability of the all-zero sample that is actually scored, while the
    repair still hands the environment a reliable single-user schedule.  The
    policy therefore collapses to ``|A| = 1`` with the schedule chosen by the
    non-differentiable argmax fallback rather than by the policy.  Prefer
    ``SubsetCategorical`` unless K is large.

``SubsetCategorical``
    A single categorical over all ``2^K - 1`` non-empty subsets.  Every outcome
    is feasible by construction, so there is no repair rule and no attractor;
    the sampled category *is* the executed schedule and *is* what gets scored.
    Enumeration costs ``2^K - 1`` logits, which is fine up to about ``K = 12``.

Every ``sample`` accepts an explicit ``torch.Generator`` so that nothing depends
on the global RNG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import torch
import torch.distributions as td
from scipy.special import betaincinv

from rsma_aoi.utils.action_mapping import hard_sigmoid_torch

_LOG_STD_MIN = -5.0
_LOG_STD_MAX = 2.0
_BETA_EPS = 1e-6
_NEG_INF = -1e9


class ActionDistribution(Protocol):
    """Minimal interface shared by every head."""

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor: ...

    def mode(self) -> torch.Tensor: ...

    def log_prob(self, value: torch.Tensor) -> torch.Tensor: ...

    def entropy(self) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over the unmasked coordinates; masked entries are exactly 0.

    Args:
        logits: ``(..., D)`` real tensor.
        mask: ``(..., D)`` boolean tensor; at least one True per row is expected.

    Returns:
        ``(..., D)`` tensor whose unmasked entries are non-negative and sum to 1
        (rows with no unmasked entry return all zeros).
    """
    mask_f = mask.to(logits.dtype)
    filled = torch.where(mask, logits, torch.full_like(logits, _NEG_INF))
    filled = filled - filled.max(dim=-1, keepdim=True).values
    exp = torch.exp(filled) * mask_f
    denom = exp.sum(dim=-1, keepdim=True)
    out = torch.where(denom > 0, exp / denom.clamp_min(1e-30), torch.zeros_like(exp))
    return out * mask_f


def _clamped_log_std(log_std: torch.Tensor) -> torch.Tensor:
    return torch.clamp(log_std, _LOG_STD_MIN, _LOG_STD_MAX)


# ---------------------------------------------------------------------------
# masked logistic-normal (simplex heads)
# ---------------------------------------------------------------------------
class MaskedLogisticNormal:
    """Gaussian latent + masked softmax transform.

    Args:
        mean: ``(B, D)`` latent means.
        log_std: ``(B, D)`` or ``(D,)`` latent log std-devs.
        mask: ``(B, D)`` boolean mask of coordinates that take part.

    ``log_prob`` and ``entropy`` are summed over the **unmasked** coordinates
    only, so the score is invariant to whatever happens in the masked ones.
    """

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor, mask: torch.Tensor) -> None:
        if mean.dim() != 2:
            raise ValueError("mean must have shape (B, D)")
        self.mask = mask.to(torch.bool)
        if self.mask.shape != mean.shape:
            raise ValueError("mask must have the same shape as mean")
        self.mean = torch.where(self.mask, mean, torch.zeros_like(mean))
        log_std = _clamped_log_std(log_std)
        if log_std.dim() == 1:
            log_std = log_std.unsqueeze(0).expand_as(mean)
        self.log_std = log_std
        self.std = torch.exp(log_std)
        self._normal = td.Normal(self.mean, self.std)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw a latent ``z``; masked coordinates are set to 0."""
        noise = torch.randn(
            self.mean.shape, generator=generator, device=self.mean.device, dtype=self.mean.dtype
        )
        z = self.mean + self.std * noise
        return torch.where(self.mask, z, torch.zeros_like(z))

    def mode(self) -> torch.Tensor:
        """Deterministic latent (the mean), used for evaluation."""
        return torch.where(self.mask, self.mean, torch.zeros_like(self.mean))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """``(B,)`` log-density of the latent, summed over unmasked coordinates."""
        lp = self._normal.log_prob(value)
        return (lp * self.mask.to(lp.dtype)).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """``(B,)`` latent entropy summed over unmasked coordinates."""
        ent = self._normal.entropy()
        return (ent * self.mask.to(ent.dtype)).sum(dim=-1)

    def transform(self, value: torch.Tensor) -> torch.Tensor:
        """Map a latent to the simplex (exact zeros outside the mask)."""
        return masked_softmax(value, self.mask)


# ---------------------------------------------------------------------------
# scalar heads on (0, 1)
# ---------------------------------------------------------------------------
class BetaDistribution:
    """Beta distribution on ``(0, 1)`` with ``alpha, beta = 1 + softplus(raw)``.

    Args:
        raw: ``(B, 2*D)`` raw head outputs (first ``D`` -> alpha, rest -> beta).
    """

    def __init__(self, raw: torch.Tensor) -> None:
        if raw.dim() != 2 or raw.shape[-1] % 2 != 0:
            raise ValueError("raw must have shape (B, 2*D)")
        dim = raw.shape[-1] // 2
        self.alpha = 1.0 + torch.nn.functional.softplus(raw[..., :dim])
        self.beta = 1.0 + torch.nn.functional.softplus(raw[..., dim:])
        self._dist = td.Beta(self.alpha, self.beta)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Inverse-CDF sampling driven by an explicit uniform generator."""
        u = torch.rand(
            self.alpha.shape, generator=generator, device=self.alpha.device, dtype=self.alpha.dtype
        )
        a = self.alpha.detach().cpu().numpy().astype(np.float64)
        b = self.beta.detach().cpu().numpy().astype(np.float64)
        un = np.clip(u.detach().cpu().numpy().astype(np.float64), _BETA_EPS, 1.0 - _BETA_EPS)
        value = betaincinv(a, b, un)
        value = np.clip(value, _BETA_EPS, 1.0 - _BETA_EPS)
        return torch.as_tensor(value, dtype=self.alpha.dtype, device=self.alpha.device)

    def mode(self) -> torch.Tensor:
        """Mode ``(a-1)/(a+b-2)``; well defined because ``a, b >= 1``."""
        num = self.alpha - 1.0
        den = (self.alpha + self.beta - 2.0).clamp_min(1e-6)
        return torch.clamp(num / den, _BETA_EPS, 1.0 - _BETA_EPS)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        v = torch.clamp(value, _BETA_EPS, 1.0 - _BETA_EPS)
        return self._dist.log_prob(v).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return self._dist.entropy().sum(dim=-1)

    def transform(self, value: torch.Tensor) -> torch.Tensor:
        """The stored value already lives in ``(0, 1)``."""
        return torch.clamp(value, _BETA_EPS, 1.0 - _BETA_EPS)


class SquashedGaussianDistribution:
    """``y = sigmoid(z)``, ``z ~ N(mu, sigma)``, with explicit Jacobian correction.

    ``sample``/``mode`` return the **latent** ``z`` (that is what the buffer
    stores and what gets re-scored), while :meth:`transform` maps it to ``(0, 1)``.
    ``log_prob(z)`` returns ``log p_Y(sigmoid(z))``, i.e. the Gaussian log-density
    minus ``log sigma'(z) = log(y) + log(1-y)``.
    """

    def __init__(self, raw: torch.Tensor) -> None:
        if raw.dim() != 2 or raw.shape[-1] % 2 != 0:
            raise ValueError("raw must have shape (B, 2*D)")
        dim = raw.shape[-1] // 2
        self.mean = raw[..., :dim]
        self.log_std = _clamped_log_std(raw[..., dim:])
        self.std = torch.exp(self.log_std)
        self._normal = td.Normal(self.mean, self.std)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.randn(
            self.mean.shape, generator=generator, device=self.mean.device, dtype=self.mean.dtype
        )
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        return self.mean

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Density of the squashed variable (Jacobian included)."""
        base = self._normal.log_prob(value)
        y = torch.sigmoid(value)
        log_det = torch.log(y.clamp_min(_BETA_EPS)) + torch.log((1.0 - y).clamp_min(_BETA_EPS))
        return (base - log_det).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """Latent Gaussian entropy (an upper bound on the squashed entropy)."""
        return self._normal.entropy().sum(dim=-1)

    def transform(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(value)


def scalar_to_unit(value: torch.Tensor, kind: str) -> torch.Tensor:
    """Map a scalar head's stored variable into ``(0, 1)``.

    ``'beta'`` stores the value itself (a clamp), ``'squashed_gaussian'`` stores
    the pre-squash latent (a sigmoid).  Both end up in the same interval, which
    is what the environment sees.
    """
    if kind == "beta":
        return torch.clamp(value, _BETA_EPS, 1.0 - _BETA_EPS)
    if kind == "squashed_gaussian":
        return torch.sigmoid(value)
    raise ValueError(f"unknown scalar distribution: {kind!r}")


def latent_to_feasible_action(
    power_latent: torch.Tensor,
    rate_latent: torch.Tensor,
    rho_r_latent: torch.Tensor,
    active_mask: torch.Tensor,
    scalar_kind: str = "beta",
    rho_p_latent: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Deterministic latent -> feasible environment action.

    This is the **single** constraint-preserving transform in the project: the
    PPO distribution and the TD3 actor both route through it, so a comparison
    between them can never be confounded by different action handling.

    Args:
        power_latent: ``(B, K+1)`` logits ``[common, private_1..K]``.
        rate_latent: ``(B, K)`` common-rate logits.
        rho_r_latent: ``(B, 1)`` scalar variable (see :func:`scalar_to_unit`).
        active_mask: ``(B, K)`` boolean schedule.
        scalar_kind: how the scalar variables are stored.
        rho_p_latent: optional ``(B, 1)`` total-power scalar.

    Returns:
        ``schedule (B,K)``, ``power_fractions (B,K+1)`` (non-negative, rows sum
        to 1, exact zeros for inactive users), ``rate_fractions (B,K)`` (simplex
        over the active users), ``common_payload_scale (B,1)`` and optionally
        ``power_scale (B,1)``.
    """
    active = active_mask.to(torch.bool)
    ones = torch.ones(active.shape[0], 1, dtype=torch.bool, device=active.device)
    power_mask = torch.cat([ones, active], dim=-1)
    out = {
        "schedule": active.to(power_latent.dtype),
        "power_fractions": masked_softmax(power_latent, power_mask),
        "rate_fractions": masked_softmax(rate_latent, active),
        "common_payload_scale": scalar_to_unit(rho_r_latent, scalar_kind),
    }
    if rho_p_latent is not None:
        out["power_scale"] = scalar_to_unit(rho_p_latent, scalar_kind)
    return out


def make_scalar_distribution(raw: torch.Tensor, kind: str):
    """Factory for the scalar heads (``'beta'`` or ``'squashed_gaussian'``)."""
    if kind == "beta":
        return BetaDistribution(raw)
    if kind == "squashed_gaussian":
        return SquashedGaussianDistribution(raw)
    raise ValueError(f"unknown scalar distribution: {kind!r}")


# ---------------------------------------------------------------------------
# scheduling head
# ---------------------------------------------------------------------------
class BernoulliSchedule:
    """Independent per-user Bernoulli activity decisions.

    ``log_prob`` scores the raw sample.  :meth:`repair` implements the
    documented "if nothing is scheduled, activate the most probable user"
    post-transform; the repaired mask is stored separately and is *not* scored.
    """

    def __init__(self, logits: torch.Tensor) -> None:
        if logits.dim() != 2:
            raise ValueError("logits must have shape (B, K)")
        self.logits = logits
        self._dist = td.Bernoulli(logits=logits)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        probs = torch.sigmoid(self.logits)
        u = torch.rand(probs.shape, generator=generator, device=probs.device, dtype=probs.dtype)
        return (u < probs).to(probs.dtype)

    def mode(self) -> torch.Tensor:
        return (self.logits > 0).to(self.logits.dtype)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self._dist.log_prob(value).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        return self._dist.entropy().sum(dim=-1)

    def repair(self, raw: torch.Tensor) -> torch.Tensor:
        """Ensure at least one active user per row (argmax logit fallback)."""
        mask = raw > 0.5
        empty = ~mask.any(dim=-1)
        if bool(empty.any()):
            best = self.logits.argmax(dim=-1)
            fix = torch.zeros_like(mask)
            fix[torch.arange(mask.shape[0], device=mask.device), best] = True
            mask = torch.where(empty.unsqueeze(-1), fix, mask)
        return mask


# ---------------------------------------------------------------------------
# composite hybrid distribution
# ---------------------------------------------------------------------------
@dataclass
class LatentAction:
    """Everything PPO must store to re-score an action exactly.

    Attributes:
        schedule_raw: ``(B, K)`` raw Bernoulli sample (scored by the policy).
        active_mask: ``(B, K)`` repaired boolean mask (defines the head masks).
        power_latent: ``(B, K+1)`` Gaussian latent of the power simplex.
        rate_latent: ``(B, K)`` Gaussian latent of the common-rate simplex.
        rho_r_latent: ``(B, 1)`` scalar latent / value.
        rho_p_latent: ``(B, 1)`` optional total-power scalar latent / value.
    """

    schedule_raw: torch.Tensor
    active_mask: torch.Tensor
    power_latent: torch.Tensor
    rate_latent: torch.Tensor
    rho_r_latent: torch.Tensor
    rho_p_latent: torch.Tensor | None = None
    common_gate_raw: torch.Tensor | None = None

    def detach(self) -> "LatentAction":
        return LatentAction(
            schedule_raw=self.schedule_raw.detach(),
            active_mask=self.active_mask.detach(),
            power_latent=self.power_latent.detach(),
            rate_latent=self.rate_latent.detach(),
            rho_r_latent=self.rho_r_latent.detach(),
            rho_p_latent=None if self.rho_p_latent is None else self.rho_p_latent.detach(),
            common_gate_raw=(None if self.common_gate_raw is None else self.common_gate_raw.detach()),
        )


class HybridPolicyDistribution:
    """Joint distribution over (schedule, power simplex, rate simplex, scalars).

    The masked heads are built from ``active_mask``: at sampling time it comes
    from the freshly repaired schedule sample, at update time it comes from the
    buffer, which guarantees that old and new log-probabilities refer to the
    same latent variables *and* the same masks.
    """

    def __init__(
        self,
        schedule_logits: torch.Tensor,
        power_mean: torch.Tensor,
        power_log_std: torch.Tensor,
        rate_mean: torch.Tensor,
        rate_log_std: torch.Tensor,
        rho_r_raw: torch.Tensor,
        rho_p_raw: torch.Tensor | None = None,
        scalar_dist: str = "beta",
        scheduling_enabled: bool = True,
        scheduling_head: str = "bernoulli",
        num_users: int | None = None,
        continuous_fn: "Callable[[torch.Tensor], dict[str, torch.Tensor]] | None" = None,
        has_power_scale: bool | None = None,
        schedule_size: int | None = None,
        schedule_sizes: tuple[int, ...] | None = None,
        forced_mask: torch.Tensor | None = None,
        allocation_control: str = "joint",
        fixed_common_power_fraction: float = 0.25,
        fixed_common_payload_scale: float = 0.3,
        common_gate_logits: torch.Tensor | None = None,
        common_gate_enabled: bool = False,
        rate_split_mode: str = "simplex_scaled",
    ) -> None:
        if allocation_control not in ("joint", "power_only", "split_only", "sdma_power"):
            raise ValueError("unknown allocation_control")
        if not 0 <= fixed_common_power_fraction <= 1 or not 0 <= fixed_common_payload_scale <= 1:
            raise ValueError("fixed allocation fractions must lie in [0, 1]")
        self.direct_fraction = rate_split_mode == "per_user_fraction"
        if self.direct_fraction and (
            common_gate_enabled or allocation_control not in ("joint", "sdma_power")
        ):
            raise ValueError(
                "direct fractions require joint or sdma_power allocation without gate"
            )
        self.learn_rho = not self.direct_fraction and allocation_control not in ("power_only", "sdma_power")
        self.learn_power = allocation_control != "split_only"
        self.learn_split = allocation_control not in ("power_only", "sdma_power")
        self.force_common_off = allocation_control == "sdma_power"
        self.fixed_common_power_fraction = fixed_common_power_fraction
        self.fixed_common_payload_scale = fixed_common_payload_scale
        self.common_gate_enabled = bool(common_gate_enabled)
        if self.common_gate_enabled and common_gate_logits is None:
            raise ValueError("common_gate_logits are required when the common gate is enabled")
        self.common_gate = None if common_gate_logits is None else BernoulliSchedule(common_gate_logits)
        self.scheduling_head = scheduling_head
        self.forced_mask = forced_mask
        if scheduling_head == "bernoulli":
            self.num_users = num_users or schedule_logits.shape[-1]
            self.schedule = BernoulliSchedule(schedule_logits)
        elif scheduling_head == "subset_categorical":
            if num_users is None:
                raise ValueError("num_users is required for the subset-categorical head")
            self.num_users = num_users
            self.schedule = SubsetCategorical(schedule_logits, num_users)
        elif scheduling_head == "topm_categorical":
            if num_users is None or schedule_size is None:
                raise ValueError("num_users and schedule_size are required for topm_categorical")
            self.num_users = num_users
            self.schedule = TopMCategorical(schedule_logits, num_users, schedule_size)
        elif scheduling_head == "count_then_subset":
            if num_users is None or not schedule_sizes:
                raise ValueError("num_users and schedule_sizes are required for count_then_subset")
            self.num_users = num_users
            n_count = len(schedule_sizes)
            self.schedule = CountThenSubset(
                schedule_logits[:, :n_count], schedule_logits[:, n_count:],
                num_users, tuple(schedule_sizes),
            )
        elif scheduling_head == "fixed_top_m":
            # the mask is supplied by the caller (a deterministic heuristic), so
            # the scheduling term contributes neither log-probability nor entropy
            if num_users is None:
                raise ValueError("num_users is required for the fixed_top_m head")
            self.num_users = num_users
            self.schedule = BernoulliSchedule(torch.zeros_like(schedule_logits[:, :num_users]))
        else:
            raise ValueError(f"unknown scheduling head: {scheduling_head!r}")
        #: heads whose sample is a flat index decoded through a subset table
        self.index_head = scheduling_head in (
            "subset_categorical", "topm_categorical", "count_then_subset"
        )
        self._unconditional = {
            "power_mean": power_mean,
            "power_log_std": power_log_std,
            "rate_mean": rate_mean,
            "rate_log_std": rate_log_std,
            "rho_r_raw": rho_r_raw,
            "rho_p_raw": rho_p_raw,
            "common_gate_logits": common_gate_logits,
        }
        #: When given, the continuous head parameters are a function of the
        #: sampled schedule (an autoregressive factorisation
        #: ``pi(a|s) = pi(A|s) pi(alloc|s, A)``).  The mask is stored in the
        #: rollout buffer, so re-scoring at update time is bit-identical.
        self._continuous_fn = continuous_fn
        self._cache: tuple[int, dict[str, torch.Tensor]] | None = None
        self.has_power_scale = (
            (rho_p_raw is not None) if has_power_scale is None else bool(has_power_scale)
        )
        self.scheduling_enabled = scheduling_enabled
        self.scalar_kind = scalar_dist

    # -- masked heads -----------------------------------------------------
    def _continuous(self, active: torch.Tensor) -> dict[str, torch.Tensor]:
        """Continuous head parameters, optionally conditioned on the schedule."""
        if self._continuous_fn is None:
            return self._unconditional
        key = id(active)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        params = self._continuous_fn(active)
        self._cache = (key, params)
        return params

    def _power_dist(
        self, active: torch.Tensor, common_gate_raw: torch.Tensor | None = None
    ) -> MaskedLogisticNormal:
        if self.force_common_off:
            common = torch.zeros(active.shape[0], 1, dtype=torch.bool, device=active.device)
        elif self.common_gate_enabled and common_gate_raw is not None:
            common = common_gate_raw.reshape(-1, 1) > 0.5
        else:
            common = torch.ones(active.shape[0], 1, dtype=torch.bool, device=active.device)
        mask = torch.cat([common, active], dim=-1)
        params = self._continuous(active)
        return MaskedLogisticNormal(params["power_mean"], params["power_log_std"], mask)

    def _rate_dist(self, active: torch.Tensor) -> MaskedLogisticNormal:
        params = self._continuous(active)
        return MaskedLogisticNormal(params["rate_mean"], params["rate_log_std"], active)

    def _rho_r_dist(self, active: torch.Tensor):
        return make_scalar_distribution(self._continuous(active)["rho_r_raw"], self.scalar_kind)

    def _rho_p_dist(self, active: torch.Tensor):
        raw = self._continuous(active)["rho_p_raw"]
        return None if raw is None else make_scalar_distribution(raw, self.scalar_kind)

    def _active_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        """Decode the scheduling sample into the executed mask.

        Bernoulli: apply the all-zero repair.  Subset-categorical: decode the
        category (always non-empty, so no repair is involved).
        """
        if not self.scheduling_enabled:
            return torch.ones(
                raw.shape[0], self.num_users, dtype=torch.bool, device=raw.device
            )
        if self.scheduling_head == "fixed_top_m":
            if self.forced_mask is None:
                raise ValueError("the fixed_top_m head requires a forced_mask")
            return self.forced_mask.to(torch.bool)
        if self.index_head:
            return self.schedule.decode(raw)
        return self.schedule.repair(raw)

    # -- sampling ---------------------------------------------------------
    def _zero_latent(self, active: torch.Tensor, width: int) -> torch.Tensor:
        return torch.zeros(active.shape[0], width, device=active.device,
                           dtype=self._unconditional["power_log_std"].dtype)

    def sample(self, generator: torch.Generator | None = None) -> LatentAction:
        if not self.scheduling_enabled:
            raw = self._all_active_raw()
        elif self.scheduling_head == "fixed_top_m":
            raw = torch.zeros_like(self.schedule.logits)
        else:
            raw = self.schedule.sample(generator)
        active = self._active_from_raw(raw)
        common_gate = None if not self.common_gate_enabled else self.common_gate.sample(generator)
        return LatentAction(
            schedule_raw=raw,
            active_mask=active,
            power_latent=(self._power_dist(active, common_gate).sample(generator) if self.learn_power
                          else self._zero_latent(active, self.num_users + 1)),
            rate_latent=(self._rate_dist(active).sample(generator) if self.learn_split
                         else self._zero_latent(active, self.num_users)),
            rho_r_latent=(self._rho_r_dist(active).sample(generator) if self.learn_rho
                          else self._zero_latent(active, 1)),
            rho_p_latent=(
                None if not self.has_power_scale else self._rho_p_dist(active).sample(generator)
            ),
            common_gate_raw=common_gate,
        )

    def _all_active_raw(self) -> torch.Tensor:
        """Scheduling variable standing for "everyone active" (M1 / TD3 setting)."""
        if self.scheduling_head == "subset_categorical":
            last = self.schedule.logits.shape[-1] - 1  # the all-ones subset
            return torch.full(
                (self.schedule.logits.shape[0], 1), float(last),
                dtype=self.schedule.logits.dtype, device=self.schedule.logits.device,
            )
        if self.index_head:
            table = self.schedule.table
            full = int(table.sum(dim=-1).argmax())
            ref = self.schedule.logits if hasattr(self.schedule, "logits") else self.schedule.subset_logits
            return torch.full((ref.shape[0], 1), float(full), dtype=ref.dtype, device=ref.device)
        return torch.ones_like(self.schedule.logits)

    def mode(self) -> LatentAction:
        """Deterministic action used for evaluation."""
        if not self.scheduling_enabled:
            raw = self._all_active_raw()
        elif self.scheduling_head == "fixed_top_m":
            raw = torch.zeros_like(self.schedule.logits)
        else:
            raw = self.schedule.mode()
        active = self._active_from_raw(raw)
        common_gate = None if not self.common_gate_enabled else self.common_gate.mode()
        return LatentAction(
            schedule_raw=raw,
            active_mask=active,
            power_latent=(self._power_dist(active, common_gate).mode() if self.learn_power
                          else self._zero_latent(active, self.num_users + 1)),
            rate_latent=(self._rate_dist(active).mode() if self.learn_split
                         else self._zero_latent(active, self.num_users)),
            rho_r_latent=(self._rho_r_dist(active).mode() if self.learn_rho
                          else self._zero_latent(active, 1)),
            rho_p_latent=(
                None if not self.has_power_scale else self._rho_p_dist(active).mode()
            ),
            common_gate_raw=common_gate,
        )

    # -- scoring ----------------------------------------------------------
    def log_prob(self, latent: LatentAction) -> torch.Tensor:
        """Joint ``(B,)`` log-probability of a stored :class:`LatentAction`."""
        active = latent.active_mask.to(torch.bool)
        total = self._zero_latent(active, 1).squeeze(-1)
        if self.learn_power:
            total = total + self._power_dist(active, latent.common_gate_raw).log_prob(latent.power_latent)
        if self.learn_split:
            split_lp = self._rate_dist(active).log_prob(latent.rate_latent)
            if self.learn_rho:
                split_lp = split_lp + self._rho_r_dist(active).log_prob(latent.rho_r_latent)
            if self.common_gate_enabled:
                if latent.common_gate_raw is None:
                    raise ValueError("stored action is missing common_gate_raw")
                total = total + latent.common_gate_raw.reshape(-1) * split_lp
            else:
                total = total + split_lp
        if self.common_gate_enabled:
            if latent.common_gate_raw is None:
                raise ValueError("stored action is missing common_gate_raw")
            total = total + self.common_gate.log_prob(latent.common_gate_raw)
        if self.has_power_scale and latent.rho_p_latent is not None:
            total = total + self._rho_p_dist(active).log_prob(latent.rho_p_latent)
        if self.scheduling_enabled and self.scheduling_head != "fixed_top_m":
            total = total + self.schedule.log_prob(latent.schedule_raw)
        return total

    def entropy(self, active: torch.Tensor) -> torch.Tensor:
        """``(B,)`` joint entropy given the stored ``active_mask``."""
        active = active.to(torch.bool)
        ent = self._zero_latent(active, 1).squeeze(-1)
        if self.learn_power:
            if self.common_gate_enabled:
                prob = torch.sigmoid(self.common_gate.logits).reshape(-1)
                on = torch.ones_like(self.common_gate.logits)
                off = torch.zeros_like(self.common_gate.logits)
                ent = ent + prob * self._power_dist(active, on).entropy()
                ent = ent + (1-prob) * self._power_dist(active, off).entropy()
            else:
                ent = ent + self._power_dist(active).entropy()
        if self.learn_split:
            split_ent = self._rate_dist(active).entropy()
            if self.learn_rho:
                split_ent = split_ent + self._rho_r_dist(active).entropy()
            if self.common_gate_enabled:
                split_ent = torch.sigmoid(self.common_gate.logits).reshape(-1) * split_ent
            ent = ent + split_ent
        if self.common_gate_enabled:
            ent = ent + self.common_gate.entropy()
        if self.has_power_scale:
            ent = ent + self._rho_p_dist(active).entropy()
        if self.scheduling_enabled and self.scheduling_head != "fixed_top_m":
            ent = ent + self.schedule.entropy()
        return ent

    def head_entropies(self, active: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-head entropies ``(B,)`` each, for the collapse diagnostics.

        The joint entropy bonus can stay high while one head collapses, so the
        breakdown is logged separately during training.
        """
        active = active.to(torch.bool)
        out = {
            "power": self._power_dist(active).entropy() if self.learn_power else self._zero_latent(active, 1).squeeze(-1),
            "rate": self._rate_dist(active).entropy() if self.learn_split else self._zero_latent(active, 1).squeeze(-1),
            "rho_r": self._rho_r_dist(active).entropy() if self.learn_rho else self._zero_latent(active, 1).squeeze(-1),
        }
        if self.common_gate_enabled:
            prob = torch.sigmoid(self.common_gate.logits).reshape(-1)
            if self.learn_power:
                on = torch.ones_like(self.common_gate.logits)
                off = torch.zeros_like(self.common_gate.logits)
                out["power"] = prob * self._power_dist(active, on).entropy()
                out["power"] += (1-prob) * self._power_dist(active, off).entropy()
            out["rate"] = out["rate"] * prob
            out["rho_r"] = out["rho_r"] * prob
            out["common_gate"] = self.common_gate.entropy()
        if self.scheduling_enabled and self.scheduling_head != "fixed_top_m":
            out["schedule"] = self.schedule.entropy()
        if self.direct_fraction:
            out.pop("rho_r", None)
        return out

    # -- transform to the environment action ------------------------------
    def to_env_action(self, latent: LatentAction) -> dict[str, torch.Tensor]:
        """Deterministic latent -> feasible environment action.

        Returns tensors ``schedule (B,K) float``, ``power_fractions (B,K+1)``,
        ``rate_fractions (B,K)``, ``common_payload_scale (B,1)`` and optionally
        ``power_scale (B,1)``.
        """
        if self.direct_fraction:
            # The endpoint-capable transform makes exact SDMA/fully-common
            # boundaries reachable without a separate discrete gate.
            private = masked_softmax(latent.power_latent[:, 1:], latent.active_mask)
            if self.force_common_off:
                alpha = torch.zeros_like(latent.power_latent[:, :1])
                phi = torch.zeros_like(latent.rate_latent)
            else:
                alpha = hard_sigmoid_torch(latent.power_latent[:, :1])
                phi = hard_sigmoid_torch(latent.rate_latent) * latent.active_mask
            return {
                "schedule": latent.active_mask.to(latent.power_latent.dtype),
                "common_power_fraction": alpha,
                "private_power_fractions": private,
                "common_payload_fractions": phi,
                # Internal aliases retained for differentiable objective code.
                "power_fractions": torch.cat([alpha, (1-alpha)*private], dim=-1),
                "rate_fractions": phi,
            }
        out = latent_to_feasible_action(
            power_latent=latent.power_latent,
            rate_latent=latent.rate_latent,
            rho_r_latent=latent.rho_r_latent,
            active_mask=latent.active_mask,
            scalar_kind=self.scalar_kind,
            rho_p_latent=latent.rho_p_latent if self.has_power_scale else None,
        )
        active = latent.active_mask.to(latent.power_latent.dtype)
        equal = active / active.sum(-1, keepdim=True).clamp_min(1)
        if not self.learn_power:
            common = torch.full_like(equal[:, :1], self.fixed_common_power_fraction)
            out["power_fractions"] = torch.cat([common, (1 - common) * equal], dim=-1)
        if not self.learn_split:
            out["rate_fractions"] = equal
            out["common_payload_scale"] = torch.full_like(equal[:, :1], self.fixed_common_payload_scale)
        if self.force_common_off:
            private = out["power_fractions"][:, 1:]
            private = private / private.sum(-1, keepdim=True).clamp_min(1e-30)
            out["power_fractions"] = torch.cat([torch.zeros_like(private[:, :1]), private], dim=-1)
            out["common_payload_scale"] = torch.zeros_like(out["common_payload_scale"])
        if self.common_gate_enabled:
            if latent.common_gate_raw is None:
                raise ValueError("stored action is missing common_gate_raw")
            gate = latent.common_gate_raw > 0.5
            private = out["power_fractions"][:, 1:]
            private = private / private.sum(-1, keepdim=True).clamp_min(1e-30)
            off_power = torch.cat([torch.zeros_like(private[:, :1]), private], dim=-1)
            out["power_fractions"] = torch.where(gate, out["power_fractions"], off_power)
            out["common_payload_scale"] = torch.where(
                gate, out["common_payload_scale"], torch.zeros_like(out["common_payload_scale"])
            )
        return out


# ---------------------------------------------------------------------------
# subset-categorical scheduling head
# ---------------------------------------------------------------------------
def enumerate_subsets(num_users: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """All non-empty subsets of ``K`` users as a ``(2^K - 1, K)`` boolean table.

    Row ``i`` is the binary expansion of ``i + 1``, so index 0 is ``{user 0}``
    and the last index is "everyone".
    """
    if num_users < 1:
        raise ValueError("num_users must be >= 1")
    codes = torch.arange(1, 2**num_users, device=device)
    bits = torch.arange(num_users, device=device)
    return ((codes[:, None] >> bits[None, :]) & 1).to(torch.bool)


class SubsetCategorical:
    """Categorical distribution over the non-empty scheduling subsets.

    Unlike :class:`BernoulliSchedule` this needs no feasibility repair: every
    category is a valid schedule, the sampled category is exactly what the
    environment executes, and it is exactly what ``log_prob`` scores.  That
    removes the all-zero attractor documented above.

    Args:
        logits: ``(B, 2^K - 1)`` category logits.
        num_users: K (used to decode a category into a mask).
    """

    def __init__(self, logits: torch.Tensor, num_users: int) -> None:
        if logits.dim() != 2:
            raise ValueError("logits must have shape (B, 2^K - 1)")
        expected = 2**num_users - 1
        if logits.shape[-1] != expected:
            raise ValueError(f"expected {expected} logits for K={num_users}, got {logits.shape[-1]}")
        self.logits = logits
        self.num_users = num_users
        self.table = enumerate_subsets(num_users, logits.device)
        self._dist = td.Categorical(logits=logits)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw category indices as a ``(B, 1)`` float tensor."""
        probs = self._dist.probs
        u = torch.rand(probs.shape[0], 1, generator=generator, device=probs.device,
                       dtype=probs.dtype)
        index = (probs.cumsum(dim=-1) < u).sum(dim=-1)
        index = index.clamp(max=probs.shape[-1] - 1)
        return index.to(probs.dtype).unsqueeze(-1)

    def mode(self) -> torch.Tensor:
        return self.logits.argmax(dim=-1).to(self.logits.dtype).unsqueeze(-1)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """``(B,)`` log-probability of the stored category index."""
        return self._dist.log_prob(value.reshape(-1).long())

    def entropy(self) -> torch.Tensor:
        return self._dist.entropy()

    def decode(self, value: torch.Tensor) -> torch.Tensor:
        """Category index ``(B, 1)`` -> boolean schedule mask ``(B, K)``."""
        return self.table[value.reshape(-1).long()]


# ---------------------------------------------------------------------------
# curriculum scheduling heads
# ---------------------------------------------------------------------------
def enumerate_sized_subsets(
    num_users: int, sizes: tuple[int, ...], device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Subsets of the given sizes as a ``(N, K)`` boolean table, ordered by size.

    Enumeration is combinatorial, so this is intended for small ``K``
    (``K <= 12`` for a full enumeration; ``sizes=(m,)`` stays small much longer).
    """
    import itertools as _it

    rows = []
    for size in sorted(sizes):
        if not 1 <= size <= num_users:
            raise ValueError(f"subset size {size} out of range for K={num_users}")
        for combo in _it.combinations(range(num_users), size):
            mask = torch.zeros(num_users, dtype=torch.bool, device=device)
            mask[list(combo)] = True
            rows.append(mask)
    if not rows:
        raise ValueError("sizes must be non-empty")
    return torch.stack(rows)


class TopMCategorical:
    """Categorical over the ``C(K, m)`` subsets of exactly ``m`` active users.

    Stage-2 curriculum head: the schedule size is fixed, so the policy only has
    to learn *which* users to serve.  Every outcome is feasible, so there is no
    repair rule.
    """

    def __init__(self, logits: torch.Tensor, num_users: int, m: int) -> None:
        self.num_users = num_users
        self.m = int(m)
        self.table = enumerate_sized_subsets(num_users, (self.m,), logits.device)
        if logits.dim() != 2 or logits.shape[-1] != self.table.shape[0]:
            raise ValueError(
                f"expected {self.table.shape[0]} logits for K={num_users}, m={m}, "
                f"got shape {tuple(logits.shape)}"
            )
        self.logits = logits
        self._dist = td.Categorical(logits=logits)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        probs = self._dist.probs
        u = torch.rand(probs.shape[0], 1, generator=generator, device=probs.device,
                       dtype=probs.dtype)
        index = (probs.cumsum(dim=-1) < u).sum(dim=-1).clamp(max=probs.shape[-1] - 1)
        return index.to(probs.dtype).unsqueeze(-1)

    def mode(self) -> torch.Tensor:
        return self.logits.argmax(dim=-1).to(self.logits.dtype).unsqueeze(-1)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self._dist.log_prob(value.reshape(-1).long())

    def entropy(self) -> torch.Tensor:
        return self._dist.entropy()

    def decode(self, value: torch.Tensor) -> torch.Tensor:
        return self.table[value.reshape(-1).long()]


class CountThenSubset:
    """Hierarchical head: choose the active-user count, then the subset.

    ``log p(A|s) = log p(m|s) + log p(A|m, s)`` with ``m`` drawn from a
    categorical over ``sizes`` and the subset drawn from a categorical over the
    ``C(K, m)`` subsets of that size.  A single flat index into the
    size-ordered subset table is stored, and ``m`` is recovered from it, so the
    rollout buffer needs one number per step.

    Args:
        count_logits: ``(B, len(sizes))``.
        subset_logits: ``(B, sum_m C(K, m))`` - one block per size, in the same
            order as ``sorted(sizes)``.
        num_users: K.
        sizes: allowed active-user counts, e.g. ``(1, 2)``.
    """

    def __init__(
        self,
        count_logits: torch.Tensor,
        subset_logits: torch.Tensor,
        num_users: int,
        sizes: tuple[int, ...],
    ) -> None:
        self.num_users = num_users
        self.sizes = tuple(sorted(sizes))
        self.table = enumerate_sized_subsets(num_users, self.sizes, subset_logits.device)
        counts = self.table.sum(dim=-1)
        self.block_sizes = [int((counts == size).sum()) for size in self.sizes]
        if count_logits.shape[-1] != len(self.sizes):
            raise ValueError(f"count head must emit {len(self.sizes)} logits")
        if subset_logits.shape[-1] != self.table.shape[0]:
            raise ValueError(f"subset head must emit {self.table.shape[0]} logits")
        self.count_logits = count_logits
        self.subset_logits = subset_logits
        self._count_dist = td.Categorical(logits=count_logits)
        offsets = np.cumsum([0] + self.block_sizes)
        self._offsets = [int(o) for o in offsets]
        #: flat index -> which size block it belongs to
        self.size_of_index = torch.zeros(self.table.shape[0], dtype=torch.long,
                                         device=subset_logits.device)
        for block, size in enumerate(self.sizes):
            self.size_of_index[self._offsets[block] : self._offsets[block + 1]] = block

    def _block_log_probs(self, block: int) -> torch.Tensor:
        lo, hi = self._offsets[block], self._offsets[block + 1]
        return torch.log_softmax(self.subset_logits[:, lo:hi], dim=-1)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw ``m`` then a subset of that size; returns the flat index ``(B, 1)``."""
        probs = self._count_dist.probs
        u = torch.rand(probs.shape[0], 1, generator=generator, device=probs.device,
                       dtype=probs.dtype)
        block = (probs.cumsum(dim=-1) < u).sum(dim=-1).clamp(max=probs.shape[-1] - 1)
        flat = torch.zeros(probs.shape[0], dtype=torch.long, device=probs.device)
        for b in range(len(self.sizes)):
            rows = block == b
            if not bool(rows.any()):
                continue
            block_probs = torch.softmax(
                self.subset_logits[rows, self._offsets[b] : self._offsets[b + 1]], dim=-1
            )
            v = torch.rand(int(rows.sum()), 1, generator=generator,
                           device=probs.device, dtype=probs.dtype)
            within = (block_probs.cumsum(dim=-1) < v).sum(dim=-1).clamp(
                max=block_probs.shape[-1] - 1
            )
            flat[rows] = within + self._offsets[b]
        return flat.to(probs.dtype).unsqueeze(-1)

    def mode(self) -> torch.Tensor:
        block = self.count_logits.argmax(dim=-1)
        flat = torch.zeros(block.shape[0], dtype=torch.long, device=block.device)
        for b in range(len(self.sizes)):
            rows = block == b
            if not bool(rows.any()):
                continue
            lo, hi = self._offsets[b], self._offsets[b + 1]
            flat[rows] = self.subset_logits[rows, lo:hi].argmax(dim=-1) + lo
        return flat.to(self.count_logits.dtype).unsqueeze(-1)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """``log p(m) + log p(A|m)`` for a stored flat index."""
        flat = value.reshape(-1).long()
        block = self.size_of_index[flat]
        total = self._count_dist.log_prob(block)
        for b in range(len(self.sizes)):
            rows = block == b
            if not bool(rows.any()):
                continue
            within = flat[rows] - self._offsets[b]
            total = total.clone()
            total[rows] = total[rows] + self._block_log_probs(b)[rows, within]
        return total

    def entropy(self) -> torch.Tensor:
        """Exact entropy of the hierarchical distribution."""
        ent = self._count_dist.entropy()
        probs = self._count_dist.probs
        for b in range(len(self.sizes)):
            log_p = self._block_log_probs(b)
            block_ent = -(log_p.exp() * log_p).sum(dim=-1)
            ent = ent + probs[:, b] * block_ent
        return ent

    def decode(self, value: torch.Tensor) -> torch.Tensor:
        return self.table[value.reshape(-1).long()]
