"""Typed configuration objects with explicit validation.

Configs are plain dataclasses loaded from YAML.  Unknown keys raise, so a typo
in a config file fails loudly instead of silently using a default.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")

# ---------------------------------------------------------------------------
# allowed literal values
# ---------------------------------------------------------------------------
PRIVATE_BEAMS = ("mrt", "zf", "rzf")
COMMON_BEAMS = ("mrt_sum", "principal", "maxmin")
SMALL_SCALE_MODELS = ("rayleigh", "rician_ula")
MULTIPLE_ACCESS = ("rsma", "sdma", "oma", "noma")
LINK_MODELS = ("fbl", "shannon", "fixed")
RATE_SPLIT_MODES = ("simplex_scaled", "per_user_fraction")
COMMON_PAYLOAD_MAPPINGS = ("direct", "fbl_normalized")
ACK_MODES = ("sampled", "expected")
REWARD_MODES = ("sampled_aoi", "expected_aoi")
CSIT_ERROR_MODES = ("relative", "absolute", "normalized")
PATHLOSS_MODELS = ("homogeneous", "explicit_near_far", "distance_pathloss")
BETA_NORMALIZATIONS = ("none", "mean_one", "cell_edge_reference")
AOI_WEIGHT_MODES = ("explicit", "equal", "inverse_snr", "class_priority")
ALLOCATOR_MODES = ("none", "equal", "equal_rsma", "grid", "cem_fast", "cem_full",
                   "cem_heavy", "random_shooting", "cached")
OBS_CHANNEL_MODES = ("complex_flat", "gain_features", "beam_sinr_features", "none")
SCALAR_DISTS = ("beta", "squashed_gaussian")
SCHEDULING_HEADS = (
    "bernoulli", "subset_categorical", "topm_categorical", "count_then_subset",
    "fixed_top_m",
)
AOI_INIT = ("one", "random")


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or inconsistent."""


def _from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Build dataclass ``cls`` from ``data``, rejecting unknown keys."""
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"unknown keys for {cls.__name__}: {sorted(unknown)} (known: {sorted(known)})"
        )
    return cls(**data)  # type: ignore[call-arg]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict (empty file -> empty dict)."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must contain a mapping at top level")
    return data


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    """Return ``data[name]`` if present, else ``data`` itself (flat layout)."""
    if name in data and isinstance(data[name], dict):
        return dict(data[name])
    return {k: v for k, v in data.items() if k not in ("env", "ppo", "td3", "experiment", "sweep")}


# ---------------------------------------------------------------------------
# environment config
# ---------------------------------------------------------------------------
@dataclass
class EnvConfig:
    """Configuration of :class:`rsma_aoi.envs.aoi_env.AoIRSMAEnv`.

    Attributes are grouped as topology / channel / PHY / AoI / action / reward /
    observation.  See ``configs/env_default.yaml`` for documented defaults.
    """

    # --- topology -------------------------------------------------------
    num_users: int = 2
    num_antennas: int = 4

    # --- channel --------------------------------------------------------
    alpha: float = 0.0
    """Gauss-Markov temporal correlation; 0 = i.i.d. block fading."""
    small_scale_model: str = "rayleigh"
    """'rayleigh' or a mean-preserving directional 'rician_ula' channel."""
    rician_k_db: float = 8.0
    """Rician LoS-to-scatter power ratio in dB (rician_ula only)."""
    user_angles_deg: list[float] | None = None
    """Per-user physical ULA departure angles; required by rician_ula."""
    user_angle_sets_deg: list[list[float]] | None = None
    """Optional finite set of per-user ULA angle vectors.

    One vector is sampled uniformly at every environment reset and held fixed
    for the episode. ``user_angles_deg`` remains the deterministic default.
    """
    antenna_spacing_wavelength: float = 0.5
    """ULA element spacing divided by carrier wavelength."""
    path_gains_db: list[float] | None = None
    """Per-user large-scale gain in dB (None -> 0 dB for every user).
    Used directly by ``pathloss_model='explicit_near_far'``."""
    csit_error_std: float = 0.0
    """sigma_e of the CSIT error model; 0 = perfect CSIT."""
    csit_error_mode: str = "relative"
    """'relative' (error scales with beta_k, the historical model),
    'absolute' (common error power - inflates weak users' estimates), or
    'normalized' (absolute error, rescaled to keep E|h_hat_k|^2 = beta_k)."""

    # --- large-scale fading / geometry -----------------------------------
    pathloss_model: str = "homogeneous"
    user_distances: list[float] | None = None
    cell_radius: float = 500.0
    min_distance: float = 35.0
    pathloss_exponent: float = 3.5
    reference_distance: float = 1.0
    reference_gain_db: float = 0.0
    shadowing_std_db: float = 0.0
    beta_normalization: str = "mean_one"
    geometry_seed: int = 0
    user_classes: list[str] | None = None
    """Per-user class label ('near'/'middle'/'far') used for per-class metrics."""
    aoi_weight_mode: str = "explicit"
    """How ``w_k`` is formed: 'explicit' (use aoi_weights), 'equal',
    'inverse_snr' (w ∝ 1/beta_k, normalised to mean 1) or 'class_priority'."""
    class_priority: dict[str, float] | None = None

    # --- PHY ------------------------------------------------------------
    noise_power: float = 1.0
    snr_db: float | None = 10.0
    """Transmit SNR P_max / sigma_n^2 in dB.  Overrides ``p_max`` when set."""
    snr_db_values: list[float] | None = None
    """Optional finite set of episode-level transmit-SNR values in dB.

    One value is sampled uniformly at reset and held fixed for the episode.
    ``snr_db`` remains the deterministic default and defines fixed-environment
    behaviour when this field is ``None``.
    """
    p_max: float | None = None
    blocklength: int = 128
    blocklength_values: list[int] | None = None
    """Optional episode-level blocklengths, sampled uniformly at reset."""
    payload_bits: float | list[float] = 256.0
    private_beam: str = "rzf"
    common_beam: str = "principal"
    common_beam_iters: int = 30
    """Iterations for the diagnostic inverse-gain reweighted common beam."""
    common_beam_damping: float = 0.3
    """Weight-update damping for ``common_beam='maxmin'``."""
    rzf_regularization: float | None = None
    """RZF loading factor; None -> K * sigma_n^2 / P_max."""
    multiple_access: str = "rsma"
    rate_split_mode: str = "simplex_scaled"
    """Payload-split parameterisation.

    ``per_user_fraction`` is the thesis default selected by every current
    ``*_direct.yaml`` config.  The dataclass fallback stays ``simplex_scaled``
    so archived scripts that constructed ``EnvConfig()`` without recording the
    parameter keep their original meaning.
    """
    common_payload_mapping: str = "direct"
    """Interpretation of the PPO common-payload scalar. ``direct`` uses it as
    the feasible payload fraction. ``fbl_normalized`` uses it as the fraction
    of the estimated-CSIT FBL-decodable common-payload budget."""
    common_target_bler: float = 0.1
    """Worst-user common-stream BLER target for ``fbl_normalized`` mapping."""
    link_model: str = "fbl"
    fixed_success_prob: float = 0.8
    """Success probability used when ``link_model == 'fixed'`` (M0 debug)."""

    # --- AoI ------------------------------------------------------------
    aoi_max: int = 64
    aoi_cap_enabled: bool = True
    """False keeps physical age uncapped; aoi_max remains the observation scale."""
    aoi_init: str = "one"
    aoi_weights: list[float] | None = None

    # --- actions --------------------------------------------------------
    scheduling_enabled: bool = True
    power_scale_action: bool = False
    """If True the agent also controls total radiated power rho_P in (0, 1]."""

    # --- hierarchical control -------------------------------------------
    allocator_mode: str = "none"
    """Low-level allocator. ``'none'`` keeps the flat action space (the agent
    outputs the allocation itself).  Any other value makes the environment
    hierarchical: the agent's schedule is kept, and the continuous allocation is
    replaced by the named optimiser's solution for that subset."""
    allocator_common_gate: bool = False
    """When hierarchical, threshold the agent's ``common_payload_scale`` at 0.5
    to decide whether the allocator may use the common stream at all."""
    allocator_cache_tol: float = 0.15
    """Nearest-neighbour radius for ``allocator_mode='cached'`` (approximate
    reuse: a hit returns the allocation of a *similar* state, so the hit rate and
    bias are reported in ``info``)."""
    allocator_seed: int = 0

    # --- transmission / feedback ----------------------------------------
    ack_mode: str = "sampled"

    # --- reward ---------------------------------------------------------
    reward_scale: float | None = None
    """Multiplier applied to the raw reward; None -> 1 / (num_users*aoi_max)."""
    reward_mode: str = "sampled_aoi"
    """Use realised next AoI or its conditional expectation as the reward cost."""
    lambda_power: float = 0.0
    lambda_tail: float = 0.0
    lambda_violation: float = 10.0
    aoi_tail_threshold: int = 10

    # --- episode / observation ------------------------------------------
    episode_length: int = 200
    obs_channel_mode: str = "complex_flat"
    obs_include_prev_action: bool = True
    obs_include_snr: bool = False
    """Append the current episode SNR divided by 20 dB to the observation."""
    obs_include_blocklength: bool = False
    """Append current blocklength divided by the configured default."""

    # --- numerics -------------------------------------------------------
    feasibility_tol: float = 1e-5

    def __post_init__(self) -> None:
        self.validate()

    # -- derived quantities ---------------------------------------------
    @property
    def total_power(self) -> float:
        """P_max in linear units."""
        if self.snr_db is not None:
            return float(self.noise_power) * float(10.0 ** (self.snr_db / 10.0))
        assert self.p_max is not None  # guaranteed by validate()
        return float(self.p_max)

    @property
    def payload_vector(self) -> list[float]:
        """Per-user payload ``D_k`` in bits, length ``num_users``."""
        if isinstance(self.payload_bits, (int, float)):
            return [float(self.payload_bits)] * self.num_users
        return [float(x) for x in self.payload_bits]

    @property
    def weight_vector(self) -> list[float]:
        """Per-user AoI weights ``w_k`` under :attr:`aoi_weight_mode`.

        The same weights are used by every algorithm in a comparison; the
        fairness auditor verifies this.
        """
        if self.aoi_weight_mode == "explicit":
            if self.aoi_weights is None:
                return [1.0] * self.num_users
            return [float(x) for x in self.aoi_weights]
        from rsma_aoi.envs.geometry import aoi_weights_for

        weights = aoi_weights_for(
            self.aoi_weight_mode, self.geometry, self.aoi_weights,
            self.class_priority, self.class_vector,
        )
        return [float(x) for x in weights]

    @property
    def effective_reward_scale(self) -> float:
        if self.reward_scale is not None:
            return float(self.reward_scale)
        return 1.0 / float(self.num_users * self.aoi_max)

    @property
    def geometry(self):
        """Resolved large-scale geometry (cached per config instance)."""
        cached = getattr(self, "_geometry_cache", None)
        if cached is not None:
            return cached
        from rsma_aoi.envs.geometry import GeometrySpec, build_geometry

        spec = GeometrySpec(
            num_users=self.num_users, model=self.pathloss_model,
            path_gains_db=self.path_gains_db, distances=self.user_distances,
            cell_radius=self.cell_radius, min_distance=self.min_distance,
            pathloss_exponent=self.pathloss_exponent,
            reference_distance=self.reference_distance,
            reference_gain_db=self.reference_gain_db,
            shadowing_std_db=self.shadowing_std_db,
            normalization=self.beta_normalization, geometry_seed=self.geometry_seed,
            user_classes=self.user_classes,
        )
        geometry = build_geometry(spec)
        object.__setattr__(self, "_geometry_cache", geometry)
        return geometry

    @property
    def path_gain_linear(self) -> list[float]:
        """Per-user large-scale gains ``beta_k`` after normalisation."""
        if self.pathloss_model == "homogeneous" and self.path_gains_db is None:
            return [1.0] * self.num_users
        if self.pathloss_model == "homogeneous":
            # backwards compatibility: path_gains_db without a model set
            return [float(10.0 ** (g / 10.0)) for g in self.path_gains_db]
        return [float(b) for b in self.geometry.beta]

    @property
    def class_vector(self) -> list[str]:
        """Per-user class label, defaulting to a single 'all' class."""
        if self.user_classes is not None:
            return list(self.user_classes)
        return ["all"] * self.num_users

    # -- validation ------------------------------------------------------
    def validate(self) -> None:
        if self.num_users < 1:
            raise ConfigError("num_users must be >= 1")
        if self.num_antennas < 1:
            raise ConfigError("num_antennas must be >= 1")
        if not 0.0 <= self.alpha <= 1.0:
            raise ConfigError("alpha must lie in [0, 1]")
        if self.csit_error_std < 0.0 or self.csit_error_std > 1.0:
            raise ConfigError("csit_error_std must lie in [0, 1]")
        if self.noise_power <= 0.0:
            raise ConfigError("noise_power must be > 0")
        if self.snr_db is None and self.p_max is None:
            raise ConfigError("either snr_db or p_max must be given")
        if self.snr_db_values is not None:
            if not self.snr_db_values:
                raise ConfigError("snr_db_values must not be empty")
            if self.snr_db is None:
                raise ConfigError("snr_db_values requires a non-null snr_db default")
            if any(not math.isfinite(float(value)) for value in self.snr_db_values):
                raise ConfigError("snr_db_values must contain only finite values")
            if len(set(float(value) for value in self.snr_db_values)) != len(self.snr_db_values):
                raise ConfigError("snr_db_values must not contain duplicates")
            if not self.obs_include_snr:
                raise ConfigError("randomized snr_db_values requires obs_include_snr=true")
        if self.p_max is not None and self.p_max <= 0.0:
            raise ConfigError("p_max must be > 0")
        if self.blocklength < 1:
            raise ConfigError("blocklength must be >= 1")
        if self.blocklength_values is not None:
            if not self.blocklength_values:
                raise ConfigError("blocklength_values must not be empty")
            if any(int(value) < 1 or float(value) != int(value) for value in self.blocklength_values):
                raise ConfigError("blocklength_values must contain positive integers")
            if len(set(int(value) for value in self.blocklength_values)) != len(self.blocklength_values):
                raise ConfigError("blocklength_values must not contain duplicates")
            if not self.obs_include_blocklength:
                raise ConfigError(
                    "randomized blocklength_values requires obs_include_blocklength=true"
                )
        for name, allowed in (
            ("private_beam", PRIVATE_BEAMS),
            ("common_beam", COMMON_BEAMS),
            ("multiple_access", MULTIPLE_ACCESS),
            ("rate_split_mode", RATE_SPLIT_MODES),
            ("common_payload_mapping", COMMON_PAYLOAD_MAPPINGS),
            ("link_model", LINK_MODELS),
            ("ack_mode", ACK_MODES),
            ("reward_mode", REWARD_MODES),
            ("obs_channel_mode", OBS_CHANNEL_MODES),
            ("aoi_init", AOI_INIT),
            ("allocator_mode", ALLOCATOR_MODES),
            ("csit_error_mode", CSIT_ERROR_MODES),
            ("pathloss_model", PATHLOSS_MODELS),
            ("beta_normalization", BETA_NORMALIZATIONS),
            ("aoi_weight_mode", AOI_WEIGHT_MODES),
            ("small_scale_model", SMALL_SCALE_MODELS),
        ):
            value = getattr(self, name)
            if value not in allowed:
                raise ConfigError(f"{name}={value!r} must be one of {allowed}")
        # must precede any weight_vector access: resolving weights needs these
        if self.aoi_weight_mode == "class_priority" and not self.class_priority:
            raise ConfigError("aoi_weight_mode='class_priority' requires class_priority")
        if self.user_classes is not None and len(self.user_classes) != self.num_users:
            raise ConfigError("user_classes length must equal num_users")
        if not math.isfinite(float(self.rician_k_db)) or not -100.0 <= float(self.rician_k_db) <= 100.0:
            raise ConfigError("rician_k_db must be finite and lie in [-100, 100]")
        if not 0.0 < self.antenna_spacing_wavelength <= 0.5:
            raise ConfigError("antenna_spacing_wavelength must lie in (0, 0.5]")
        if self.small_scale_model == "rician_ula":
            if self.user_angles_deg is None:
                raise ConfigError("rician_ula requires explicit user_angles_deg")
            if len(self.user_angles_deg) != self.num_users:
                raise ConfigError("user_angles_deg length must equal num_users")
            if any(not -90.0 <= float(angle) <= 90.0 for angle in self.user_angles_deg):
                raise ConfigError("user_angles_deg values must lie in [-90, 90]")
            if self.csit_error_mode != "relative":
                raise ConfigError(
                    "rician_ula currently supports only mean-preserving relative CSIT"
                )
            if self.user_angle_sets_deg is not None:
                if not self.user_angle_sets_deg:
                    raise ConfigError("user_angle_sets_deg must not be empty")
                for angle_set in self.user_angle_sets_deg:
                    if len(angle_set) != self.num_users:
                        raise ConfigError(
                            "each user_angle_sets_deg entry must have length num_users"
                        )
                    if any(
                        not math.isfinite(float(angle))
                        or not -90.0 <= float(angle) <= 90.0
                        for angle in angle_set
                    ):
                        raise ConfigError(
                            "user_angle_sets_deg values must be finite and lie in [-90, 90]"
                        )
        elif self.user_angle_sets_deg is not None:
            raise ConfigError("user_angle_sets_deg is supported only by rician_ula")
        if self.rzf_regularization is not None and self.rzf_regularization < 0.0:
            raise ConfigError("rzf_regularization must be >= 0")
        if self.common_beam_iters < 0:
            raise ConfigError("common_beam_iters must be >= 0")
        if not 0.0 < self.common_beam_damping <= 1.0:
            raise ConfigError("common_beam_damping must lie in (0, 1]")
        if not 0.0 <= self.fixed_success_prob <= 1.0:
            raise ConfigError("fixed_success_prob must lie in [0, 1]")
        if not 0.0 < self.common_target_bler < 1.0:
            raise ConfigError("common_target_bler must lie strictly inside (0, 1)")
        if self.rate_split_mode == "per_user_fraction" and (self.allocator_mode != "none" or self.power_scale_action):
            raise ConfigError("per_user_fraction currently requires direct allocation with fixed total power")
        if self.common_payload_mapping == "fbl_normalized":
            if self.link_model != "fbl" or self.rate_split_mode != "simplex_scaled":
                raise ConfigError(
                    "fbl_normalized common payload requires FBL simplex_scaled RSMA"
                )
            if self.multiple_access != "rsma" or self.allocator_mode != "none":
                raise ConfigError(
                    "fbl_normalized common payload requires direct RSMA allocation"
                )
        if self.aoi_max < 1:
            raise ConfigError("aoi_max must be >= 1")
        if self.aoi_tail_threshold < 1:
            raise ConfigError("aoi_tail_threshold must be >= 1")
        if self.episode_length < 1:
            raise ConfigError("episode_length must be >= 1")
        if len(self.payload_vector) != self.num_users:
            raise ConfigError("payload_bits list length must equal num_users")
        if any(d <= 0 for d in self.payload_vector):
            raise ConfigError("payload_bits must be > 0")
        if self.aoi_weight_mode == "explicit" and len(self.weight_vector) != self.num_users:
            raise ConfigError("aoi_weights length must equal num_users")
        if any(w < 0 for w in self.weight_vector):
            raise ConfigError("aoi_weights must be >= 0")
        if self.path_gains_db is not None and len(self.path_gains_db) != self.num_users:
            raise ConfigError("path_gains_db length must equal num_users")
        if self.user_classes is not None and len(self.user_classes) != self.num_users:
            raise ConfigError("user_classes length must equal num_users")
        if self.user_distances is not None and len(self.user_distances) != self.num_users:
            raise ConfigError("user_distances length must equal num_users")
        if self.pathloss_model == "explicit_near_far" and self.path_gains_db is None:
            raise ConfigError("pathloss_model='explicit_near_far' requires path_gains_db")
        for name in ("cell_radius", "min_distance", "pathloss_exponent",
                     "reference_distance"):
            if getattr(self, name) <= 0.0:
                raise ConfigError(f"{name} must be > 0")
        if self.min_distance >= self.cell_radius:
            raise ConfigError("min_distance must be < cell_radius")
        if self.shadowing_std_db < 0.0:
            raise ConfigError("shadowing_std_db must be >= 0")
        for name in ("lambda_power", "lambda_tail", "lambda_violation"):
            if getattr(self, name) < 0.0:
                raise ConfigError(f"{name} must be >= 0")
        if self.feasibility_tol <= 0.0:
            raise ConfigError("feasibility_tol must be > 0")
        if self.allocator_cache_tol <= 0.0:
            raise ConfigError("allocator_cache_tol must be > 0")
        if self.allocator_mode != "none" and self.multiple_access == "noma":
            raise ConfigError("the hierarchical allocator does not support NOMA")

    # -- (de)serialisation ------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnvConfig":
        return _from_dict(cls, _section(data, "env"))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EnvConfig":
        return cls.from_dict(load_yaml(path))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# PPO config
# ---------------------------------------------------------------------------
@dataclass
class PPOConfig:
    """Hyper-parameters of :class:`rsma_aoi.agents.ppo.PPOTrainer`."""

    total_timesteps: int = 400_000
    num_envs: int = 8
    rollout_steps: int = 256
    """Steps collected *per environment* per update."""
    update_epochs: int = 10
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_value_loss: bool = True
    value_coef: float = 0.5
    entropy_coef: float = 0.005
    max_grad_norm: float = 0.5
    learning_rate: float = 3.0e-4
    anneal_lr: bool = True
    target_kl: float | None = 0.03
    normalize_advantage: bool = True
    normalize_observation: bool = True
    freeze_observation_stats: bool = False
    """Keep restored observation-normalisation statistics fixed during PPO.
    This is useful for warm-started actors whose weights were fitted under a
    particular input scale."""
    normalize_reward: bool = False

    # network
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])
    activation: str = "tanh"
    recurrent_enabled: bool = False
    """Use sequence-preserving GRU-PPO instead of the feed-forward policy."""
    recurrent_hidden_size: int = 128
    """Width of the separate actor and critic GRU memories."""
    recurrent_seq_len: int = 32
    """Truncated backpropagation length used by recurrent PPO updates."""
    log_std_init: float = -0.5
    scalar_dist: str = "beta"
    """Distribution for scalar (0,1) actions: 'beta' or 'squashed_gaussian'."""
    conditional_allocation: bool = False
    """Condition the power/rate heads on the sampled schedule, giving
    pi(a|s) = pi(A|s) pi(alloc|s, A).  Without it the allocation logits are the
    same for every subset (see the README's coupled-local-optimum finding)."""
    allocation_control: str = "joint"
    """joint | power_only | split_only | sdma_power; fixed components are excluded from PPO scores.

    In direct per-user-fraction mode, ``sdma_power`` learns only the private
    power simplex and fixes common power and all common payload fractions to 0.
    """
    fixed_common_power_fraction: float = 0.25
    fixed_common_payload_scale: float = 0.3
    initial_common_power_fraction: float | None = None
    """Optional initial modal common-power share; None keeps symmetric logits."""
    initial_common_fraction: float | None = None
    """Initial modal per-user common-payload fraction ``phi_k`` (direct mode)."""
    initial_common_payload_scale: float | None = None
    """Legacy alias for ``initial_common_fraction`` or the old ``rho_r`` mode."""
    initial_scalar_concentration: float = 20.0
    """Beta pseudo-count controlling spread around initial_common_payload_scale."""
    common_gate_enabled: bool = False
    """Learn an exact common-off/on Bernoulli gate in addition to continuous allocation."""
    initial_common_on_probability: float = 0.5
    scheduling_head: str = "bernoulli"
    """Scheduling parameterisation / curriculum stage:
    'bernoulli' (K independent decisions + an all-zero repair),
    'subset_categorical' (one categorical over the 2^K - 1 non-empty subsets),
    'topm_categorical' (categorical over the C(K, m) subsets of size m),
    'count_then_subset' (p(m|s) then p(A|m,s)), or
    'fixed_top_m' (a deterministic max-AoI-first schedule; the head is frozen).
    Enumeration is combinatorial, so the categorical variants are intended for
    small K - see the README."""
    schedule_size: int | None = None
    """m for 'topm_categorical'; None resolves to min(K, Nt)."""
    schedule_sizes: list[int] | None = None
    """Allowed active-user counts for 'count_then_subset'; None -> [1..min(K, Nt)]."""
    count_prior: list[float] | None = None
    """Initial (learnable) prior over the counts in 'count_then_subset'."""

    # --- warm start -----------------------------------------------------
    pretrained_actor: str | None = None
    """Path to a behaviour-cloning checkpoint to initialise the actor from."""
    warm_start_log_std: float | None = None
    """Optional latent-action log standard deviation applied after loading a
    pretrained actor.  This lets PPO begin with narrow exploration around a
    deterministic warm solution instead of inheriting the generic policy
    default.  None preserves the checkpoint/default value."""
    reset_critic: bool = True
    """The cloned actor has no value function, so the critic starts fresh."""
    freeze_actor_steps: int = 0
    """Environment steps during which every actor parameter is frozen so a
    freshly reset critic can fit the warm policy before policy updates begin."""
    freeze_scheduling_steps: int = 0
    """Environment steps during which the scheduling head is frozen."""
    freeze_allocation_steps: int = 0
    """Environment steps during which the allocation heads are frozen."""
    actor_lr: float | None = None
    """Actor learning rate; None -> ``learning_rate``."""
    critic_lr: float | None = None
    """Critic learning rate; None -> ``learning_rate``."""
    reference_kl_coef: float = 0.0
    """Penalty pulling the policy towards the pretrained reference policy."""
    reference_kl_anneal_steps: int = 0
    """Linearly anneal ``reference_kl_coef`` to 0 over this many steps (0 = never)."""
    probe_dataset: str | None = None
    """Oracle dataset directory used for the fixed forgetting probe."""
    probe_size: int = 512
    probe_interval: int = 5
    """Updates between probe evaluations (0 disables)."""

    # bookkeeping
    device: str = "cpu"
    log_interval: int = 1
    checkpoint_interval: int = 20
    eval_interval: int = 0
    eval_episodes: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        self.validate()

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_steps

    @property
    def effective_initial_common_fraction(self) -> float | None:
        """Resolve the canonical direct-mode initialisation and its legacy alias."""
        return (
            self.initial_common_fraction
            if self.initial_common_fraction is not None
            else self.initial_common_payload_scale
        )

    def validate(self) -> None:
        for name in ("total_timesteps", "num_envs", "rollout_steps", "update_epochs",
                     "minibatch_size"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be >= 1")
        if self.batch_size % self.minibatch_size != 0:
            raise ConfigError(
                f"batch_size ({self.batch_size}) must be divisible by minibatch_size "
                f"({self.minibatch_size})"
            )
        if self.recurrent_hidden_size < 1 or self.recurrent_seq_len < 1:
            raise ConfigError("recurrent_hidden_size and recurrent_seq_len must be >= 1")
        if self.recurrent_enabled:
            if self.rollout_steps % self.recurrent_seq_len != 0:
                raise ConfigError(
                    "rollout_steps must be divisible by recurrent_seq_len for GRU-PPO"
                )
            if self.minibatch_size % self.recurrent_seq_len != 0:
                raise ConfigError(
                    "minibatch_size must be divisible by recurrent_seq_len for GRU-PPO"
                )
        if not 0.0 < self.gamma <= 1.0:
            raise ConfigError("gamma must lie in (0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ConfigError("gae_lambda must lie in [0, 1]")
        if self.clip_coef <= 0.0:
            raise ConfigError("clip_coef must be > 0")
        if self.learning_rate <= 0.0:
            raise ConfigError("learning_rate must be > 0")
        if self.max_grad_norm <= 0.0:
            raise ConfigError("max_grad_norm must be > 0")
        if self.target_kl is not None and self.target_kl <= 0.0:
            raise ConfigError("target_kl must be > 0 or null")
        if self.scalar_dist not in SCALAR_DISTS:
            raise ConfigError(f"scalar_dist must be one of {SCALAR_DISTS}")
        if self.allocation_control not in ("joint", "power_only", "split_only", "sdma_power"):
            raise ConfigError("allocation_control must be joint, power_only, split_only, or sdma_power")
        for name in ("fixed_common_power_fraction", "fixed_common_payload_scale"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ConfigError(f"{name} must lie in [0, 1]")
        if self.initial_common_fraction is not None and self.initial_common_payload_scale is not None:
            raise ConfigError(
                "set only initial_common_fraction; initial_common_payload_scale is a legacy alias"
            )
        for name in (
            "initial_common_power_fraction",
            "initial_common_fraction",
            "initial_common_payload_scale",
        ):
            value = getattr(self, name)
            if value is not None and not 0.0 < value < 1.0:
                raise ConfigError(f"{name} must lie strictly inside (0, 1) or be null")
        if self.initial_scalar_concentration <= 0.0:
            raise ConfigError("initial_scalar_concentration must be > 0")
        if not 0.0 < self.initial_common_on_probability < 1.0:
            raise ConfigError("initial_common_on_probability must lie strictly inside (0, 1)")
        if self.scheduling_head not in SCHEDULING_HEADS:
            raise ConfigError(f"scheduling_head must be one of {SCHEDULING_HEADS}")
        if self.schedule_size is not None and self.schedule_size < 1:
            raise ConfigError("schedule_size must be >= 1")
        if self.schedule_sizes is not None:
            if not self.schedule_sizes or any(m < 1 for m in self.schedule_sizes):
                raise ConfigError("schedule_sizes must be a non-empty list of positive ints")
            if len(set(self.schedule_sizes)) != len(self.schedule_sizes):
                raise ConfigError("schedule_sizes must not contain duplicates")
        for name in ("actor_lr", "critic_lr"):
            value = getattr(self, name)
            if value is not None and value <= 0.0:
                raise ConfigError(f"{name} must be > 0 or null")
        if self.warm_start_log_std is not None:
            if not math.isfinite(self.warm_start_log_std):
                raise ConfigError("warm_start_log_std must be finite or null")
            if not -10.0 <= self.warm_start_log_std <= 2.0:
                raise ConfigError("warm_start_log_std must lie in [-10, 2] or be null")
        for name in ("freeze_actor_steps", "freeze_scheduling_steps", "freeze_allocation_steps",
                     "reference_kl_anneal_steps", "probe_size", "probe_interval"):
            if getattr(self, name) < 0:
                raise ConfigError(f"{name} must be >= 0")
        if self.reference_kl_coef < 0.0:
            raise ConfigError("reference_kl_coef must be >= 0")
        if self.count_prior is not None:
            if any(v < 0 for v in self.count_prior) or sum(self.count_prior) <= 0:
                raise ConfigError("count_prior must be non-negative and sum to > 0")
        if self.activation not in ("tanh", "relu", "elu"):
            raise ConfigError("activation must be one of ('tanh', 'relu', 'elu')")
        if not self.hidden_sizes:
            raise ConfigError("hidden_sizes must be non-empty")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PPOConfig":
        return _from_dict(cls, _section(data, "ppo"))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PPOConfig":
        return cls.from_dict(load_yaml(path))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def dump_config_snapshot(path: str | Path, **sections: Any) -> None:
    """Write a YAML snapshot of the configs actually used by a run."""
    payload = {
        key: (value.to_dict() if hasattr(value, "to_dict") else value)
        for key, value in sections.items()
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)


def load_run_configs(
    env_config: str | Path, agent_config: str | Path | None = None
) -> tuple[EnvConfig, PPOConfig]:
    """Load the environment and PPO configs used by a training run.

    Precedence for PPO keys: defaults < ``agent_config`` file < a ``ppo:``
    section inside ``env_config`` (so a scenario file such as
    ``configs/m0_debug.yaml`` can be self-contained).
    """
    env_data = load_yaml(env_config)
    env_cfg = EnvConfig.from_dict(env_data)

    ppo_data: dict[str, Any] = {}
    if agent_config is not None:
        ppo_data.update(_section(load_yaml(agent_config), "ppo"))
    if isinstance(env_data.get("ppo"), dict):
        ppo_data.update(env_data["ppo"])
    ppo_cfg = _from_dict(PPOConfig, ppo_data)
    return env_cfg, ppo_cfg


def _coerce(value: str, field_type: Any) -> Any:
    """Convert a ``KEY=VALUE`` string to the dataclass field's type."""
    text = value.strip()
    lowered = text.lower()
    if lowered in ("null", "none"):
        return None
    origin = str(field_type)
    if "bool" in origin and "|" not in origin.replace("bool", "", 1):
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        raise ConfigError(f"cannot parse {value!r} as a boolean")
    if lowered in ("true", "false"):
        return lowered == "true"
    if text.startswith("[") and text.endswith("]"):
        return yaml.safe_load(text)
    if "int" in origin and "float" not in origin:
        return int(float(text))
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if "int" in origin and number.is_integer() else number


def apply_overrides_from_strings(config: T, overrides: list[str]) -> T:
    """Return a copy of ``config`` with ``KEY=VALUE`` command-line overrides applied.

    Values are coerced using the dataclass field annotations, so
    ``--set anneal_lr=false entropy_coef=0.02 hidden_sizes=[256,256]`` works.
    The result is revalidated, so an invalid override fails immediately.
    """
    known = {f.name: f.type for f in fields(config)}  # type: ignore[arg-type]
    updates: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override {item!r} must have the form KEY=VALUE")
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in known:
            raise ConfigError(f"unknown config field {key!r}; known: {sorted(known)}")
        updates[key] = _coerce(raw, known[key])
    new = dataclasses.replace(config, **updates)  # type: ignore[type-var]
    new.validate()  # type: ignore[attr-defined]
    return new


@dataclass
class TD3Config:
    """Hyper-parameters of :class:`rsma_aoi.agents.td3.TD3Trainer`.

    Defaults follow Fujimoto et al. (2018) adapted to this environment's short
    episodes and vectorised data collection.
    """

    total_timesteps: int = 400_000
    num_envs: int = 8
    buffer_size: int = 300_000
    batch_size: int = 256
    learning_starts: int = 10_000
    """Uniform random latent actions are taken until this many steps elapse."""
    gradient_steps_per_iter: int = 1
    """Gradient updates per *vectorised* environment step."""
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    exploration_noise: float = 0.15
    target_noise: float = 0.2
    noise_clip: float = 0.5
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    max_grad_norm: float = 10.0
    latent_scale: float = 3.0
    """Half-width of the tanh-bounded latent box the actor emits."""
    normalize_observation: bool = True
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    device: str = "cpu"
    log_interval: int = 20
    checkpoint_interval: int = 100
    seed: int = 0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in ("total_timesteps", "num_envs", "buffer_size", "batch_size",
                     "policy_delay", "gradient_steps_per_iter"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be >= 1")
        if self.learning_starts < 0:
            raise ConfigError("learning_starts must be >= 0")
        if not 0.0 < self.gamma <= 1.0:
            raise ConfigError("gamma must lie in (0, 1]")
        if not 0.0 < self.tau <= 1.0:
            raise ConfigError("tau must lie in (0, 1]")
        for name in ("exploration_noise", "target_noise", "noise_clip"):
            if getattr(self, name) < 0.0:
                raise ConfigError(f"{name} must be >= 0")
        for name in ("actor_lr", "critic_lr", "max_grad_norm", "latent_scale"):
            if getattr(self, name) <= 0.0:
                raise ConfigError(f"{name} must be > 0")
        if self.activation not in ("tanh", "relu", "elu"):
            raise ConfigError("activation must be one of ('tanh', 'relu', 'elu')")
        if not self.hidden_sizes:
            raise ConfigError("hidden_sizes must be non-empty")
        if self.batch_size > self.buffer_size:
            raise ConfigError("batch_size must not exceed buffer_size")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TD3Config":
        return _from_dict(cls, _section(data, "td3"))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TD3Config":
        return cls.from_dict(load_yaml(path))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_td3_configs(
    env_config: str | Path, agent_config: str | Path | None = None
) -> tuple[EnvConfig, TD3Config]:
    """Load the environment and TD3 configs used by a training run."""
    env_data = load_yaml(env_config)
    env_cfg = EnvConfig.from_dict(env_data)
    td3_data: dict[str, Any] = {}
    if agent_config is not None:
        td3_data.update(_section(load_yaml(agent_config), "td3"))
    if isinstance(env_data.get("td3"), dict):
        td3_data.update(env_data["td3"])
    return env_cfg, _from_dict(TD3Config, td3_data)
