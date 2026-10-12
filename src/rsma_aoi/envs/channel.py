"""Rayleigh/Rician block-fading MU-MISO channel with temporal correlation.

Model (all quantities complex, shapes ``(K, Nt)``)
-------------------------------------------------
The default small-scale model is Rayleigh, i.e. every entry of ``h_k`` is
``CN(0, beta_k)``.  ``rician_ula`` adds a deterministic ULA LoS mean while
preserving the same total per-entry power ``beta_k``::

    m_k = sqrt(beta_k * kappa/(kappa+1)) * a(phi_k)
    h_k = m_k + g_k,  g_k ~ CN(0, beta_k/(kappa+1) I)

The steering vector has unit-magnitude entries, not unit vector norm, so both
models satisfy ``E[||h_k||^2] = Nt * beta_k``.

Temporal evolution is first-order Gauss-Markov::

    h_k(t+1) = m_k + alpha * (h_k(t) - m_k) + sqrt(1-alpha^2) z_k(t)

where ``z_k`` has the model's scatter covariance. ``alpha = 0`` gives
independent blocks and ``alpha = 1`` freezes the scattering realization.

Imperfect CSIT
--------------
With a heterogeneous geometry the *same* nominal ``sigma_e`` can mean very
different things for a near and a far user, so three error models are provided
(``csit_error_mode``).  Write ``beta_ref = mean_k beta_k``.

``relative`` (default)
    ``h_hat_k = sqrt(1 - sigma_e^2) h_k + sigma_e e_k``, ``e_k ~ CN(0, beta_k I)``.
    The error scales with the user's own gain, so *every* user has the same
    relative CSIT quality and ``E[|h_hat_k|^2] = beta_k`` exactly.  This is the
    model every earlier experiment used, and it reduces to the homogeneous case
    when all ``beta_k`` are equal.

``absolute``
    ``e_k ~ CN(0, beta_ref I)`` - the estimation error has the same *absolute*
    power for everyone, which is what a noise-limited uplink-pilot estimator
    produces.  Far users are then relatively much worse estimated.  Note the
    artefact this creates: ``E[|h_hat_k|^2] = (1 - sigma_e^2) beta_k +
    sigma_e^2 beta_ref``, which **exceeds** ``beta_k`` for weak users, i.e. the
    transmitter believes a far user is stronger than it is.  This mode is
    provided so that artefact can be studied deliberately, not by accident.

``normalized``
    Absolute error power, then each row is rescaled so ``E[|h_hat_k|^2] =
    beta_k`` again.  This keeps the realistic property that far users are
    relatively worse estimated while removing the power-inflation artefact, and
    is the recommended setting for heterogeneous studies.

For ``rician_ula``, relative CSIT error is applied around the LoS mean with the
scatter covariance, preserving both the Rician mean and marginal covariance.
The other CSIT modes are deliberately rejected for Rician until a physical
estimator is specified. In every mode the transmitter designs beamformers from
``h_hat`` while SINRs use the true ``h``; ``sigma_e=0`` gives exact CSIT.

ASSUMPTION: ``h`` and ``h_hat`` are jointly Gaussian and the error ``e`` is
independent of ``h``; no channel-estimation pilot overhead is deducted from the
blocklength.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rsma_aoi.utils.config import EnvConfig


#: Supported CSIT error models (see the module docstring).
CSIT_ERROR_MODES = ("relative", "absolute", "normalized")


def _cn(rng: np.random.Generator, shape: tuple[int, ...], var: np.ndarray | float) -> np.ndarray:
    """Draw circularly-symmetric complex Gaussian samples with variance ``var``.

    Args:
        rng: explicit generator (no global state is used).
        shape: output shape.
        var: per-entry variance, broadcastable to ``shape``.

    Returns:
        Complex array of shape ``shape`` with ``E[|x|^2] = var``.
    """
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    return np.sqrt(np.asarray(var) / 2.0) * (real + 1j * imag)


@dataclass
class ChannelModel:
    """Stateless channel sampler driven by an explicit generator.

    Attributes:
        num_users: K.
        num_antennas: Nt.
        alpha: Gauss-Markov correlation coefficient in [0, 1].
        csit_error_std: sigma_e in [0, 1].
        path_gain: shape ``(K,)`` linear large-scale gains beta_k.
    """

    num_users: int
    num_antennas: int
    alpha: float
    csit_error_std: float
    path_gain: np.ndarray
    csit_error_mode: str = "relative"
    small_scale_model: str = "rayleigh"
    rician_k_db: float = 8.0
    user_angles_deg: np.ndarray | None = None
    user_angle_sets_deg: np.ndarray | None = None
    antenna_spacing_wavelength: float = 0.5

    @classmethod
    def from_config(cls, cfg: EnvConfig) -> "ChannelModel":
        return cls(
            num_users=cfg.num_users,
            num_antennas=cfg.num_antennas,
            alpha=float(cfg.alpha),
            csit_error_std=float(cfg.csit_error_std),
            path_gain=np.asarray(cfg.path_gain_linear, dtype=np.float64),
            csit_error_mode=cfg.csit_error_mode,
            small_scale_model=cfg.small_scale_model,
            rician_k_db=float(cfg.rician_k_db),
            user_angles_deg=(
                None
                if cfg.user_angles_deg is None
                else np.asarray(cfg.user_angles_deg, dtype=np.float64)
            ),
            user_angle_sets_deg=(
                None
                if cfg.user_angle_sets_deg is None
                else np.asarray(cfg.user_angle_sets_deg, dtype=np.float64)
            ),
            antenna_spacing_wavelength=float(cfg.antenna_spacing_wavelength),
        )

    def __post_init__(self) -> None:
        self.path_gain = np.asarray(self.path_gain, dtype=np.float64).reshape(-1)
        if self.path_gain.shape[0] != self.num_users:
            raise ValueError("path_gain must have length num_users")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if not 0.0 <= self.csit_error_std <= 1.0:
            raise ValueError("csit_error_std must lie in [0, 1]")
        if self.csit_error_mode not in CSIT_ERROR_MODES:
            raise ValueError(f"csit_error_mode must be one of {CSIT_ERROR_MODES}")
        if self.small_scale_model not in ("rayleigh", "rician_ula"):
            raise ValueError("small_scale_model must be 'rayleigh' or 'rician_ula'")
        if not np.isfinite(self.rician_k_db) or not -100.0 <= self.rician_k_db <= 100.0:
            raise ValueError("rician_k_db must lie in [-100, 100]")
        if not 0.0 < self.antenna_spacing_wavelength <= 0.5:
            raise ValueError("antenna_spacing_wavelength must lie in (0, 0.5]")
        if self.small_scale_model == "rician_ula":
            if self.user_angles_deg is None:
                raise ValueError("rician_ula requires explicit user_angles_deg")
            self.user_angles_deg = np.asarray(self.user_angles_deg, dtype=np.float64).reshape(-1)
            if self.user_angles_deg.shape[0] != self.num_users:
                raise ValueError("user_angles_deg must have length num_users")
            if np.any(~np.isfinite(self.user_angles_deg)) or np.any(
                (self.user_angles_deg < -90.0) | (self.user_angles_deg > 90.0)
            ):
                raise ValueError("user_angles_deg values must lie in [-90, 90]")
            if self.csit_error_mode != "relative":
                raise ValueError("rician_ula supports only mean-preserving relative CSIT")
            if self.user_angle_sets_deg is not None:
                self.user_angle_sets_deg = np.asarray(
                    self.user_angle_sets_deg, dtype=np.float64
                )
                if (
                    self.user_angle_sets_deg.ndim != 2
                    or self.user_angle_sets_deg.shape[1] != self.num_users
                    or self.user_angle_sets_deg.shape[0] < 1
                ):
                    raise ValueError(
                        "user_angle_sets_deg must have shape (num_sets, num_users)"
                    )
                if np.any(~np.isfinite(self.user_angle_sets_deg)) or np.any(
                    (self.user_angle_sets_deg < -90.0)
                    | (self.user_angle_sets_deg > 90.0)
                ):
                    raise ValueError(
                        "user_angle_sets_deg values must lie in [-90, 90]"
                    )
        elif self.user_angle_sets_deg is not None:
            raise ValueError("user_angle_sets_deg is supported only by rician_ula")

    def resample_episode_geometry(self, rng: np.random.Generator) -> int | None:
        """Select an episode-level Rician angle vector when configured.

        Fixed-geometry models return ``None`` without consuming RNG state, so
        historical seeded traces remain unchanged.
        """
        if self.user_angle_sets_deg is None:
            return None
        index = int(rng.integers(self.user_angle_sets_deg.shape[0]))
        self.user_angles_deg = self.user_angle_sets_deg[index].copy()
        return index

    @property
    def shape(self) -> tuple[int, int]:
        return (self.num_users, self.num_antennas)

    @property
    def _var_col(self) -> np.ndarray:
        """Per-user scattering variance, broadcast over antennas."""
        return self.scatter_variance[:, None]

    @property
    def rician_k_linear(self) -> float:
        """Linear Rician K-factor (unused by the Rayleigh path)."""
        return float(10.0 ** (self.rician_k_db / 10.0))

    @property
    def scatter_variance(self) -> np.ndarray:
        """Per-entry variance of the zero-mean time-varying component."""
        if self.small_scale_model == "rayleigh":
            return self.path_gain.copy()
        return self.path_gain / (self.rician_k_linear + 1.0)

    @property
    def mean(self) -> np.ndarray:
        """Deterministic channel mean, shape ``(K, Nt)``."""
        if self.small_scale_model == "rayleigh":
            return np.zeros(self.shape, dtype=np.complex128)
        assert self.user_angles_deg is not None
        angles = np.deg2rad(self.user_angles_deg)
        elements = np.arange(self.num_antennas, dtype=np.float64)
        steering = np.exp(
            1j
            * 2.0
            * np.pi
            * self.antenna_spacing_wavelength
            * np.sin(angles)[:, None]
            * elements[None, :]
        )
        los_power = self.path_gain * self.rician_k_linear / (self.rician_k_linear + 1.0)
        return np.sqrt(los_power)[:, None] * steering

    def sample_initial(self, rng: np.random.Generator) -> np.ndarray:
        """Draw ``H(0)`` from the stationary distribution. Shape ``(K, Nt)``."""
        # Preserve the historical arithmetic path and seeded samples exactly.
        if self.small_scale_model == "rayleigh":
            return _cn(rng, self.shape, self._var_col)
        return self.mean + _cn(rng, self.shape, self._var_col)

    def evolve(self, h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Advance the channel by one slot. ``h`` and output are ``(K, Nt)``."""
        if h.shape != self.shape:
            raise ValueError(f"expected channel of shape {self.shape}, got {h.shape}")
        if self.small_scale_model == "rayleigh":
            if self.alpha == 0.0:
                return _cn(rng, self.shape, self._var_col)
            innovation = _cn(rng, self.shape, self._var_col)
            return self.alpha * h + np.sqrt(1.0 - self.alpha**2) * innovation
        mean = self.mean
        if self.alpha == 0.0:
            return mean + _cn(rng, self.shape, self._var_col)
        innovation = _cn(rng, self.shape, self._var_col)
        return mean + self.alpha * (h - mean) + np.sqrt(1.0 - self.alpha**2) * innovation

    @property
    def reference_gain(self) -> float:
        """``beta_ref``: the common error scale for the absolute/normalized modes."""
        return float(np.mean(self.path_gain))

    def estimate(self, h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return the CSIT estimate ``H_hat`` of the true channel ``h``.

        Dispatches on :attr:`csit_error_mode` (see the module docstring).  With
        ``csit_error_std == 0`` this returns ``h`` unchanged (a copy) in every
        mode, so a perfect-CSIT run is bit-identical across modes.
        """
        if h.shape != self.shape:
            raise ValueError(f"expected channel of shape {self.shape}, got {h.shape}")
        sigma = self.csit_error_std
        if sigma == 0.0:
            return h.copy()

        mode = self.csit_error_mode
        if mode == "relative":
            error = _cn(rng, self.shape, self._var_col)
            if self.small_scale_model == "rayleigh":
                return np.sqrt(1.0 - sigma**2) * h + sigma * error
            mean = self.mean
            return mean + np.sqrt(1.0 - sigma**2) * (h - mean) + sigma * error

        # absolute / normalized share a common absolute error power beta_ref
        error = _cn(rng, self.shape, self.reference_gain)
        estimate = np.sqrt(1.0 - sigma**2) * h + sigma * error
        if mode == "absolute":
            return estimate
        # normalized: restore E[|h_hat_k|^2] = beta_k per user
        expected = (1.0 - sigma**2) * self.path_gain + sigma**2 * self.reference_gain
        scale = np.sqrt(self.path_gain / np.clip(expected, 1e-300, None))
        return estimate * scale[:, None]

    def sample_true_given_estimate(
        self,
        h_hat: np.ndarray,
        rng: np.random.Generator,
        num_samples: int = 1,
    ) -> np.ndarray:
        """Sample the exact Gaussian posterior ``H | H_hat`` in relative mode.

        For the mean-preserving observation model

        ``H_hat-m = a(H-m) + sigma E``, ``a=sqrt(1-sigma**2)``,

        the marginal covariance of ``H_hat-m`` equals that of ``H-m``.  Hence
        the conditional distribution has mean ``m+a(H_hat-m)`` and covariance
        ``sigma**2 * scatter_covariance``.  Returned shape is
        ``(num_samples, K, Nt)``.
        """
        h_hat = np.asarray(h_hat)
        if h_hat.shape != self.shape:
            raise ValueError(f"expected estimate of shape {self.shape}, got {h_hat.shape}")
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if self.csit_error_mode != "relative":
            raise NotImplementedError(
                "posterior sampling is currently defined only for relative CSIT"
            )
        sigma = self.csit_error_std
        if sigma == 0.0:
            return np.repeat(h_hat[None, ...], num_samples, axis=0)
        a = np.sqrt(1.0 - sigma**2)
        conditional_mean = self.mean + a * (h_hat - self.mean)
        innovation = _cn(
            rng, (num_samples, *self.shape), self.scatter_variance[None, :, None]
        )
        return conditional_mean[None, ...] + sigma * innovation
