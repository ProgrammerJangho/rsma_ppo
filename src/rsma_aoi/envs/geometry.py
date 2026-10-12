"""User geometry and large-scale (path-loss) channel gains.

The small-scale process is implemented in :mod:`rsma_aoi.envs.channel`.
Here ``beta_k`` always means total per-antenna power
``E[|h_{k,n}|^2]``, for both Rayleigh and mean-preserving Rician channels.

Path-loss models
----------------
``homogeneous``
    ``beta_k = 1`` for every user - the setting used by every earlier
    experiment in this project.

``explicit_near_far``
    ``beta_k = 10^(g_k / 10)`` from a per-user gain list in dB.  Convenient for
    a controlled near/far disparity.

``distance_pathloss``
    ``beta_k = C0 * (d_k / d0)^(-eta) * 10^(X_k / 10)`` with ``C0`` the
    reference gain at ``d0``, ``eta`` the path-loss exponent and ``X_k`` optional
    log-normal shadowing in dB.  Distances are either given explicitly or drawn
    uniformly *in area* between ``min_distance`` and ``cell_radius``.

Normalisation
-------------
Raw ``beta`` values span orders of magnitude, which interacts with the fixed
``P_max`` and ``sigma_n^2``.  Three modes make that explicit:

``none``
    Use the raw values.  The absolute SNR then depends on the geometry.
``mean_one``
    Scale so ``mean_k beta_k = 1``.  The *average* user sees the configured
    ``snr_db``; the disparity between users is preserved exactly.
``cell_edge_reference``
    Scale so the **weakest** user has ``beta = 1``, i.e. ``snr_db`` is the
    cell-edge SNR and every other user is better off.

Whichever is used is recorded in the experiment manifest together with the
resulting per-user average SNR, because "SNR = 10 dB" means different things
under different normalisations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

PATHLOSS_MODELS = ("homogeneous", "explicit_near_far", "distance_pathloss")
BETA_NORMALIZATIONS = ("none", "mean_one", "cell_edge_reference")


@dataclass(frozen=True)
class Geometry:
    """Resolved per-user large-scale state of one cell."""

    beta: np.ndarray
    """Linear large-scale gains, shape ``(K,)``."""
    distances: np.ndarray | None
    shadowing_db: np.ndarray | None
    model: str
    normalization: str
    raw_beta: np.ndarray
    """Gains before normalisation (the physical values)."""

    @property
    def num_users(self) -> int:
        return int(self.beta.shape[0])

    def average_snr(self, total_power: float, noise_power: float) -> np.ndarray:
        """Per-user average received SNR ``beta_k P_max / sigma_n^2`` (linear)."""
        return self.beta * float(total_power) / float(noise_power)

    def average_snr_db(self, total_power: float, noise_power: float) -> np.ndarray:
        return 10.0 * np.log10(np.clip(self.average_snr(total_power, noise_power), 1e-300, None))

    def disparity_db(self) -> float:
        """Spread between the strongest and weakest user, in dB."""
        return float(
            10.0 * np.log10(self.beta.max() / np.clip(self.beta.min(), 1e-300, None))
        )

    def to_dict(self, total_power: float | None = None,
                noise_power: float | None = None) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "normalization": self.normalization,
            "beta": self.beta.tolist(),
            "beta_db": (10.0 * np.log10(np.clip(self.beta, 1e-300, None))).tolist(),
            "raw_beta": self.raw_beta.tolist(),
            "disparity_db": self.disparity_db(),
            "distances": None if self.distances is None else self.distances.tolist(),
            "shadowing_db": None if self.shadowing_db is None else self.shadowing_db.tolist(),
        }
        if total_power is not None and noise_power is not None:
            payload["average_snr_db"] = self.average_snr_db(total_power, noise_power).tolist()
        return payload


@dataclass
class GeometrySpec:
    """Configuration of the large-scale channel (mirrors the EnvConfig fields)."""

    num_users: int
    model: str = "homogeneous"
    path_gains_db: list[float] | None = None
    distances: list[float] | None = None
    cell_radius: float = 500.0
    min_distance: float = 35.0
    pathloss_exponent: float = 3.5
    reference_distance: float = 1.0
    reference_gain_db: float = 0.0
    shadowing_std_db: float = 0.0
    normalization: str = "mean_one"
    geometry_seed: int = 0
    user_classes: list[str] | None = None
    """Optional per-user class label ('near'/'middle'/'far') for reporting."""

    def validate(self) -> None:
        if self.model not in PATHLOSS_MODELS:
            raise ValueError(f"pathloss_model must be one of {PATHLOSS_MODELS}")
        if self.normalization not in BETA_NORMALIZATIONS:
            raise ValueError(f"beta_normalization must be one of {BETA_NORMALIZATIONS}")
        if self.model == "explicit_near_far" and self.path_gains_db is None:
            raise ValueError("explicit_near_far requires path_gains_db")
        if self.path_gains_db is not None and len(self.path_gains_db) != self.num_users:
            raise ValueError("path_gains_db length must equal num_users")
        if self.distances is not None and len(self.distances) != self.num_users:
            raise ValueError("distances length must equal num_users")
        if self.cell_radius <= 0 or self.min_distance <= 0:
            raise ValueError("cell_radius and min_distance must be > 0")
        if self.min_distance >= self.cell_radius:
            raise ValueError("min_distance must be < cell_radius")
        if self.pathloss_exponent <= 0:
            raise ValueError("pathloss_exponent must be > 0")
        if self.reference_distance <= 0:
            raise ValueError("reference_distance must be > 0")
        if self.shadowing_std_db < 0:
            raise ValueError("shadowing_std_db must be >= 0")
        if self.user_classes is not None and len(self.user_classes) != self.num_users:
            raise ValueError("user_classes length must equal num_users")


def _normalise(beta: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return beta
    if mode == "mean_one":
        return beta / np.clip(beta.mean(), 1e-300, None)
    if mode == "cell_edge_reference":
        return beta / np.clip(beta.min(), 1e-300, None)
    raise ValueError(f"unknown normalisation {mode!r}")


def build_geometry(spec: GeometrySpec) -> Geometry:
    """Resolve a :class:`GeometrySpec` into concrete per-user gains.

    Distances (when drawn) use an explicit generator seeded by
    ``geometry_seed``, so a geometry is exactly reproducible and can be shared
    across algorithms.
    """
    spec.validate()
    k = spec.num_users
    rng = np.random.default_rng(spec.geometry_seed)
    distances = None
    shadowing = None

    if spec.model == "homogeneous":
        raw = np.ones(k, dtype=np.float64)
    elif spec.model == "explicit_near_far":
        raw = np.asarray(
            [10.0 ** (g / 10.0) for g in spec.path_gains_db], dtype=np.float64
        )
    else:  # distance_pathloss
        if spec.distances is not None:
            distances = np.asarray(spec.distances, dtype=np.float64)
        else:
            # uniform in area between min_distance and cell_radius
            u = rng.random(k)
            distances = np.sqrt(
                spec.min_distance**2 + u * (spec.cell_radius**2 - spec.min_distance**2)
            )
        shadowing = (
            rng.normal(0.0, spec.shadowing_std_db, size=k)
            if spec.shadowing_std_db > 0
            else np.zeros(k)
        )
        c0 = 10.0 ** (spec.reference_gain_db / 10.0)
        raw = (
            c0
            * (distances / spec.reference_distance) ** (-spec.pathloss_exponent)
            * 10.0 ** (shadowing / 10.0)
        )

    raw = np.clip(raw, 1e-300, None)
    beta = _normalise(raw, spec.normalization)
    if not np.all(np.isfinite(beta)) or np.any(beta <= 0):
        raise FloatingPointError("path-loss model produced non-positive or non-finite gains")
    return Geometry(
        beta=beta, distances=distances, shadowing_db=shadowing, model=spec.model,
        normalization=spec.normalization, raw_beta=raw,
    )


def near_far_gains_db(num_users: int, disparity_db: float, groups: int = 2) -> list[float]:
    """Evenly spaced class gains spanning ``disparity_db``, strongest first.

    ``groups = 2`` gives near/far, ``3`` gives near/middle/far.  Users are split
    as evenly as possible, and the strongest class always sits at 0 dB so the
    disparity is a pure attenuation of the weaker classes.
    """
    if groups < 1 or groups > num_users:
        raise ValueError("groups must lie in [1, num_users]")
    levels = (
        np.zeros(1) if groups == 1 else np.linspace(0.0, -abs(disparity_db), groups)
    )
    sizes = [num_users // groups] * groups
    for i in range(num_users - sum(sizes)):
        sizes[i] += 1
    out: list[float] = []
    for level, size in zip(levels, sizes):
        out.extend([float(level)] * size)
    return out


def class_labels(num_users: int, groups: int = 2) -> list[str]:
    """Class label per user matching :func:`near_far_gains_db`'s split."""
    names = {1: ["near"], 2: ["near", "far"], 3: ["near", "middle", "far"]}.get(
        groups, [f"class{i}" for i in range(groups)]
    )
    sizes = [num_users // groups] * groups
    for i in range(num_users - sum(sizes)):
        sizes[i] += 1
    out: list[str] = []
    for name, size in zip(names, sizes):
        out.extend([name] * size)
    return out


def aoi_weights_for(
    mode: str, geometry: Geometry, explicit: list[float] | None = None,
    class_priority: dict[str, float] | None = None,
    classes: list[str] | None = None,
) -> np.ndarray:
    """AoI weights ``w_k`` under the requested weighting policy.

    ``equal``
        All ones.
    ``inverse_snr``
        ``w_k ∝ 1 / beta_k``, normalised to mean 1 - prioritises far users,
        which is the natural fairness counterweight to a near/far geometry.
    ``class_priority``
        Per-class weights from ``class_priority``, normalised to mean 1.
    ``explicit``
        The list given in the config, used verbatim.

    The same weights are used by **every** algorithm in a comparison; the
    fairness auditor checks this.
    """
    k = geometry.num_users
    if mode == "equal":
        return np.ones(k, dtype=np.float64)
    if mode == "explicit":
        if explicit is None:
            raise ValueError("aoi_weight_mode='explicit' requires aoi_weights")
        return np.asarray(explicit, dtype=np.float64)
    if mode == "inverse_snr":
        w = 1.0 / np.clip(geometry.beta, 1e-300, None)
        return w / w.mean()
    if mode == "class_priority":
        if not class_priority or classes is None:
            raise ValueError("class_priority weighting requires class_priority and classes")
        w = np.asarray([float(class_priority.get(c, 1.0)) for c in classes], dtype=np.float64)
        return w / np.clip(w.mean(), 1e-300, None)
    raise ValueError(f"unknown aoi_weight_mode {mode!r}")
