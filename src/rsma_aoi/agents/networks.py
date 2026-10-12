"""Actor-critic network: shared encoder + hybrid policy heads + value head."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from rsma_aoi.agents.distributions import HybridPolicyDistribution
from rsma_aoi.utils.action_mapping import hard_sigmoid_inverse

_ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}


def _schedule_logit_dim(
    head: str, num_users: int, schedule_size: int | None, schedule_sizes: tuple[int, ...] | None
) -> int:
    """Width of the scheduling head's output for each parameterisation."""
    import math

    if head in ("bernoulli", "fixed_top_m"):
        return num_users
    if head == "subset_categorical":
        return 2**num_users - 1
    if head == "topm_categorical":
        if schedule_size is None:
            raise ValueError("topm_categorical requires schedule_size")
        return math.comb(num_users, schedule_size)
    if head == "count_then_subset":
        if not schedule_sizes:
            raise ValueError("count_then_subset requires schedule_sizes")
        return len(schedule_sizes) + sum(math.comb(num_users, m) for m in schedule_sizes)
    raise ValueError(f"unknown scheduling head: {head!r}")


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal initialisation (the usual PPO default)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def make_mlp(in_dim: int, hidden: list[int], activation: str) -> tuple[nn.Sequential, int]:
    """Build an MLP trunk; returns the module and its output width."""
    act = _ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    last = in_dim
    for width in hidden:
        layers.append(layer_init(nn.Linear(last, width)))
        layers.append(act())
        last = width
    return nn.Sequential(*layers), last


class HybridActorCritic(nn.Module):
    """Shared-encoder actor-critic for the RSMA/AoI hybrid action space.

    Heads:
        ``schedule``: ``K`` Bernoulli logits, or ``2^K - 1`` categorical logits
            when ``scheduling_head='subset_categorical'``.
        ``power``: ``K+1`` latent means (+ a state-independent log-std vector).
        ``rate``: ``K`` latent means for per-user payload fractions ``phi_k``
            in the canonical direct mode (+ log-std vector).
        ``rho_r``: legacy-only scalar head for ``simplex_scaled`` checkpoints.
        ``rho_p``: 2 raw parameters, only when ``power_scale_action`` is on.
        ``value``: scalar state value.
    """

    def __init__(
        self,
        obs_dim: int,
        num_users: int,
        hidden_sizes: list[int],
        activation: str = "tanh",
        log_std_init: float = -0.5,
        scalar_dist: str = "beta",
        scheduling_enabled: bool = True,
        power_scale_action: bool = False,
        scheduling_head: str = "bernoulli",
        conditional_allocation: bool = False,
        schedule_size: int | None = None,
        schedule_sizes: tuple[int, ...] | None = None,
        count_prior: list[float] | None = None,
        allocation_control: str = "joint",
        fixed_common_power_fraction: float = 0.25,
        fixed_common_payload_scale: float = 0.3,
        initial_common_power_fraction: float | None = None,
        initial_common_payload_scale: float | None = None,
        initial_scalar_concentration: float = 20.0,
        common_gate_enabled: bool = False,
        initial_common_on_probability: float = 0.5,
        rate_split_mode: str = "simplex_scaled",
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.rate_split_mode = rate_split_mode
        self.direct_fraction = rate_split_mode == "per_user_fraction"
        if self.direct_fraction and (
            common_gate_enabled
            or allocation_control not in ("joint", "sdma_power")
            or power_scale_action
        ):
            raise ValueError(
                "per_user_fraction requires joint or sdma_power allocation, "
                "no gate, and fixed total power"
            )
        self.allocation_control = allocation_control
        self.fixed_common_power_fraction = fixed_common_power_fraction
        self.fixed_common_payload_scale = fixed_common_payload_scale
        self.common_gate_enabled = common_gate_enabled
        self.scalar_dist = scalar_dist
        self.scheduling_enabled = scheduling_enabled
        self.power_scale_action = power_scale_action
        self.scheduling_head = scheduling_head
        self.schedule_size = schedule_size
        self.schedule_sizes = tuple(schedule_sizes) if schedule_sizes else None
        #: width of the scheduling *logits* produced by the head
        self.schedule_dim = _schedule_logit_dim(
            scheduling_head, num_users, schedule_size, self.schedule_sizes
        )
        #: width of the scheduling *variable* stored in the rollout buffer
        #: (K Bernoulli samples, or a single flat index)
        self.schedule_action_dim = (
            num_users if scheduling_head in ("bernoulli", "fixed_top_m") else 1
        )
        #: When True the continuous heads additionally receive the sampled
        #: schedule, giving the autoregressive factorisation
        #: ``pi(a|s) = pi(A|s) * pi(alloc|s, A)``.  Without it the allocation
        #: logits are identical for every subset, so an exploratory schedule is
        #: scored with an allocation tuned for the modal one (see the README).
        self.conditional_allocation = bool(conditional_allocation)

        self.encoder, feat = make_mlp(obs_dim, hidden_sizes, activation)
        self.critic_trunk, cfeat = make_mlp(obs_dim, hidden_sizes, activation)

        alloc_in = feat + (num_users if self.conditional_allocation else 0)
        self.schedule_head = layer_init(nn.Linear(feat, self.schedule_dim), std=0.01)
        self.power_head = layer_init(nn.Linear(alloc_in, num_users + 1), std=0.01)
        self.rate_head = layer_init(nn.Linear(alloc_in, num_users), std=0.01)
        self.rho_r_head = None if self.direct_fraction else layer_init(nn.Linear(alloc_in, 2), std=0.01)
        self.common_gate_head = (
            layer_init(
                nn.Linear(alloc_in, 1), std=0.01,
                bias_const=float(np.log(initial_common_on_probability / (1-initial_common_on_probability))),
            ) if common_gate_enabled else None
        )
        self.rho_p_head = (
            layer_init(nn.Linear(alloc_in, 2), std=0.01) if power_scale_action else None
        )
        self.value_head = layer_init(nn.Linear(cfeat, 1), std=1.0)

        self.power_log_std = nn.Parameter(torch.full((num_users + 1,), float(log_std_init)))
        self.rate_log_std = nn.Parameter(torch.full((num_users,), float(log_std_init)))

        # A symmetric zero bias starts at f_c=1/(K+1), rho=0.5.  In a short-packet
        # problem that modal coding rate can be virtually undecodable.  These
        # optional biases change only the initial policy, never the action set.
        with torch.no_grad():
            if initial_common_power_fraction is not None:
                private = (1.0 - initial_common_power_fraction) / num_users
                target = torch.tensor(
                    [initial_common_power_fraction] + [private] * num_users,
                    dtype=self.power_head.bias.dtype, device=self.power_head.bias.device,
                )
                self.power_head.bias.copy_(target.log())
                if self.direct_fraction:
                    self.power_head.bias[0] = hard_sigmoid_inverse(initial_common_power_fraction)
            if initial_common_payload_scale is not None:
                target = float(initial_common_payload_scale)
                if self.direct_fraction:
                    self.rate_head.bias.fill_(hard_sigmoid_inverse(target))
                elif scalar_dist == "beta":
                    # Beta mode = u/(u+v), where u/v are the positive softplus
                    # increments above one.  Set u+v to the requested concentration.
                    masses = torch.tensor(
                        [initial_scalar_concentration * target,
                         initial_scalar_concentration * (1.0 - target)],
                        dtype=self.rho_r_head.bias.dtype, device=self.rho_r_head.bias.device,
                    )
                    # Stable inverse-softplus: log(exp(x)-1) = x+log(1-exp(-x)).
                    self.rho_r_head.bias.copy_(masses + torch.log(-torch.expm1(-masses)))
                elif scalar_dist == "squashed_gaussian":
                    self.rho_r_head.bias[0] = np.log(target / (1.0 - target))
                    self.rho_r_head.bias[1] = float(log_std_init)
                else:  # configuration validation normally catches this first
                    raise ValueError(f"unknown scalar_dist: {scalar_dist}")

        if scheduling_head == "count_then_subset" and count_prior:
            # bias the count head towards the requested prior; it stays learnable
            prior = torch.as_tensor(count_prior, dtype=torch.float32)
            if prior.numel() != len(self.schedule_sizes or ()):
                raise ValueError("count_prior must have one entry per schedule size")
            with torch.no_grad():
                self.schedule_head.bias[: prior.numel()] = torch.log(
                    prior.clamp_min(1e-6) / prior.clamp_min(1e-6).sum()
                )

    # ------------------------------------------------------------------
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """``(B,)`` state-value estimate."""
        return self.value_head(self.critic_trunk(obs)).squeeze(-1)

    def value_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """Return values from an already encoded critic feature tensor."""
        return self.value_head(features).squeeze(-1)

    def _allocation_params(self, feat: torch.Tensor, active: torch.Tensor) -> dict:
        """Continuous head parameters for a given (already sampled) schedule."""
        joint = torch.cat([feat, active.to(feat.dtype)], dim=-1)
        return {
            "power_mean": self.power_head(joint),
            "power_log_std": self.power_log_std,
            "rate_mean": self.rate_head(joint),
            "rate_log_std": self.rate_log_std,
            "rho_r_raw": None if self.rho_r_head is None else self.rho_r_head(joint),
            "common_gate_logits": None if self.common_gate_head is None else self.common_gate_head(joint),
            "rho_p_raw": None if self.rho_p_head is None else self.rho_p_head(joint),
        }

    def distribution(
        self, obs: torch.Tensor, forced_mask: torch.Tensor | None = None
    ) -> HybridPolicyDistribution:
        """Build the joint action distribution for a batch of observations.

        With ``conditional_allocation`` the continuous heads are evaluated
        lazily, once the scheduling mask is known, so the policy factorises as
        ``pi(a|s) = pi(A|s) * pi(allocation|s, A)``.

        ``forced_mask`` supplies the schedule externally (the ``fixed_top_m``
        curriculum stage), in which case the scheduling head contributes neither
        log-probability nor entropy.
        """
        return self.distribution_from_features(self.encoder(obs), forced_mask=forced_mask)

    def distribution_from_features(
        self, feat: torch.Tensor, forced_mask: torch.Tensor | None = None
    ) -> HybridPolicyDistribution:
        """Build the policy distribution from precomputed actor features.

        This split keeps the ordinary feed-forward policy unchanged while
        allowing recurrent policies to insert memory between the encoder and
        the existing, thoroughly tested hybrid action heads.
        """
        continuous_fn = None
        if self.conditional_allocation:
            continuous_fn = lambda active: self._allocation_params(feat, active)  # noqa: E731
        return HybridPolicyDistribution(
            schedule_logits=self.schedule_head(feat),
            power_mean=self.power_head(feat) if not self.conditional_allocation else None,
            power_log_std=self.power_log_std,
            rate_mean=self.rate_head(feat) if not self.conditional_allocation else None,
            rate_log_std=self.rate_log_std,
            rho_r_raw=self.rho_r_head(feat) if self.rho_r_head is not None and not self.conditional_allocation else None,
            common_gate_logits=(
                None if self.common_gate_head is None or self.conditional_allocation
                else self.common_gate_head(feat)
            ),
            rho_p_raw=(
                None
                if self.rho_p_head is None or self.conditional_allocation
                else self.rho_p_head(feat)
            ),
            scalar_dist=self.scalar_dist,
            scheduling_enabled=self.scheduling_enabled,
            scheduling_head=self.scheduling_head,
            num_users=self.num_users,
            continuous_fn=continuous_fn,
            has_power_scale=self.rho_p_head is not None,
            schedule_size=self.schedule_size,
            schedule_sizes=self.schedule_sizes,
            forced_mask=forced_mask,
            allocation_control=self.allocation_control,
            fixed_common_power_fraction=self.fixed_common_power_fraction,
            fixed_common_payload_scale=self.fixed_common_payload_scale,
            common_gate_enabled=self.common_gate_enabled,
            rate_split_mode=self.rate_split_mode,
        )

    def forward(self, obs: torch.Tensor) -> tuple[HybridPolicyDistribution, torch.Tensor]:
        return self.distribution(obs), self.value(obs)


class RecurrentHybridActorCritic(HybridActorCritic):
    """Hybrid actor-critic with separate GRU memories for actor and critic.

    The GRUs are residual adapters on top of the ordinary MLP features.  Their
    output projections start at zero, so loading a feed-forward warm start
    yields *exactly* the same initial policy.  PPO can then learn temporal
    corrections without discarding the objective-pretrained allocation map.
    """

    def __init__(self, *args, recurrent_hidden_size: int = 128, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        actor_width = self.schedule_head.in_features
        critic_width = self.value_head.in_features
        self.recurrent_hidden_size = int(recurrent_hidden_size)
        self.actor_gru = nn.GRUCell(actor_width, self.recurrent_hidden_size)
        self.critic_gru = nn.GRUCell(critic_width, self.recurrent_hidden_size)
        self.actor_recurrent_proj = nn.Linear(self.recurrent_hidden_size, actor_width)
        self.critic_recurrent_proj = nn.Linear(self.recurrent_hidden_size, critic_width)
        nn.init.zeros_(self.actor_recurrent_proj.weight)
        nn.init.zeros_(self.actor_recurrent_proj.bias)
        nn.init.zeros_(self.critic_recurrent_proj.weight)
        nn.init.zeros_(self.critic_recurrent_proj.bias)

    def initial_state(
        self, batch_size: int, device: torch.device | str | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = device or next(self.parameters()).device
        shape = (int(batch_size), self.recurrent_hidden_size)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

    @staticmethod
    def _reset_hidden(hidden: torch.Tensor, episode_start: torch.Tensor | None) -> torch.Tensor:
        if episode_start is None:
            return hidden
        keep = (~episode_start.to(dtype=torch.bool)).to(hidden.dtype).unsqueeze(-1)
        return hidden * keep

    def recurrent_step(
        self,
        obs: torch.Tensor,
        actor_hidden: torch.Tensor,
        critic_hidden: torch.Tensor,
        episode_start: torch.Tensor | None = None,
        forced_mask: torch.Tensor | None = None,
    ) -> tuple[HybridPolicyDistribution, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process one vectorised time step and return updated memories."""
        actor_hidden = self._reset_hidden(actor_hidden, episode_start)
        critic_hidden = self._reset_hidden(critic_hidden, episode_start)
        actor_base = self.encoder(obs)
        critic_base = self.critic_trunk(obs)
        next_actor = self.actor_gru(actor_base, actor_hidden)
        next_critic = self.critic_gru(critic_base, critic_hidden)
        actor_features = actor_base + self.actor_recurrent_proj(next_actor)
        critic_features = critic_base + self.critic_recurrent_proj(next_critic)
        dist = self.distribution_from_features(actor_features, forced_mask=forced_mask)
        value = self.value_from_features(critic_features)
        return dist, value, next_actor, next_critic

    def recurrent_value_step(
        self,
        obs: torch.Tensor,
        critic_hidden: torch.Tensor,
        episode_start: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a critic time step without advancing actor memory."""
        critic_hidden = self._reset_hidden(critic_hidden, episode_start)
        base = self.critic_trunk(obs)
        next_hidden = self.critic_gru(base, critic_hidden)
        features = base + self.critic_recurrent_proj(next_hidden)
        return self.value_from_features(features), next_hidden


class DeterministicActor(nn.Module):
    """TD3 actor: observation -> bounded latent action.

    The output is ``latent_scale * tanh(.)``, so the latent lives in a compact
    box; it is then mapped to a feasible environment action by
    :func:`rsma_aoi.agents.distributions.latent_to_feasible_action` - the very
    same transform the PPO policy uses.

    Latent layout (width ``2K + 2`` with the optional power scalar):
        ``[power logits (K+1) | rate logits (K) | rho_r | (rho_P)]``.
    """

    def __init__(
        self,
        obs_dim: int,
        num_users: int,
        hidden_sizes: list[int],
        activation: str = "relu",
        latent_scale: float = 3.0,
        power_scale_action: bool = False,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.power_scale_action = power_scale_action
        self.latent_scale = float(latent_scale)
        self.action_dim = (num_users + 1) + num_users + 1 + (1 if power_scale_action else 0)
        self.trunk, feat = make_mlp(obs_dim, hidden_sizes, activation)
        self.head = layer_init(nn.Linear(feat, self.action_dim), std=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """``(B, obs_dim)`` -> ``(B, action_dim)`` in ``[-scale, scale]``."""
        return self.latent_scale * torch.tanh(self.head(self.trunk(obs)))


class TwinCritic(nn.Module):
    """TD3 twin Q-networks over ``(observation, latent action)``."""

    def __init__(
        self, obs_dim: int, action_dim: int, hidden_sizes: list[int], activation: str = "relu"
    ) -> None:
        super().__init__()
        self.q1_trunk, f1 = make_mlp(obs_dim + action_dim, hidden_sizes, activation)
        self.q1_head = layer_init(nn.Linear(f1, 1), std=1.0)
        self.q2_trunk, f2 = make_mlp(obs_dim + action_dim, hidden_sizes, activation)
        self.q2_head = layer_init(nn.Linear(f2, 1), std=1.0)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both Q estimates, each of shape ``(B,)``."""
        joint = torch.cat([obs, action], dim=-1)
        return (
            self.q1_head(self.q1_trunk(joint)).squeeze(-1),
            self.q2_head(self.q2_trunk(joint)).squeeze(-1),
        )

    def q1(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Only the first critic (used for the actor loss)."""
        return self.q1_head(self.q1_trunk(torch.cat([obs, action], dim=-1))).squeeze(-1)
