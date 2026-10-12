"""Resource-parity audit and machine-readable experiment manifests.

Comparing OMA, SDMA and RSMA is only meaningful if every scheme consumes the
same slot resources.  This module *measures* that from the physical layer
instead of asserting that the configs look similar, and records the result in a
manifest that travels with every experiment.

Accounting conventions
----------------------
* **Superposition schemes** (``rsma``, ``sdma``, ``noma``) transmit all streams
  simultaneously over the same ``n`` channel uses, so the slot consumes ``n``
  channel uses in total regardless of how many users are active.
* **Orthogonal access** (``oma``) splits the slot, so its consumption is
  ``sum_k n_k``, which must also equal ``n``.
* Radiated energy is ``(time-averaged power) x (channel uses)``.  For OMA the
  PHY reports the time-averaged power (``P_max * tau_k`` per user), so the same
  product gives the true energy ``sum_k P_max n_k = P_max n``.
* Bandwidth is an *assumption*, not a measurement: every scheme is modelled on
  one band of ``n`` channel uses per slot.  It is recorded so the assumption is
  explicit.

RSMA gain sign convention (fixed project-wide)
----------------------------------------------
    gain_vs_sdma = (AoI_SDMA - AoI_RSMA) / AoI_SDMA
    gain_vs_oma  = (AoI_OMA  - AoI_RSMA) / AoI_OMA

**Positive means RSMA is better** (lower AoI).  Every table and figure in the
project uses this sign.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer, SlotAction
from rsma_aoi.utils.config import EnvConfig

#: Schemes whose streams share the slot rather than partition it.
SUPERPOSITION_SCHEMES = ("rsma", "sdma", "noma")
ORTHOGONAL_SCHEMES = ("oma",)


def rsma_gain(aoi_reference: float, aoi_rsma: float) -> float:
    """Relative AoI gain of RSMA over a reference scheme.

    ``(AoI_reference - AoI_rsma) / AoI_reference``; **positive means RSMA wins**.
    Returns NaN for a non-positive reference.
    """
    if not np.isfinite(aoi_reference) or aoi_reference <= 0.0:
        return float("nan")
    return float((aoi_reference - aoi_rsma) / aoi_reference)


def gain_table(aoi_by_scheme: dict[str, float], rsma_key: str = "rsma") -> dict[str, float]:
    """Gains of ``rsma_key`` against every other scheme, same sign convention."""
    if rsma_key not in aoi_by_scheme:
        raise KeyError(f"{rsma_key!r} missing from {sorted(aoi_by_scheme)}")
    target = aoi_by_scheme[rsma_key]
    return {
        f"gain_vs_{name}": rsma_gain(value, target)
        for name, value in aoi_by_scheme.items()
        if name != rsma_key
    }


@dataclass(frozen=True)
class SchemeBudget:
    """Per-slot resources actually consumed by one multiple-access scheme."""

    scheme: str
    channel_uses: float
    radiated_energy: float
    peak_power: float
    offered_payload_bits: float
    num_scheduled: int
    slot_blocklength: int
    bandwidth_channel_uses: int
    """Bandwidth assumption marker: every scheme uses one band of ``n`` uses."""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def measure_scheme_budget(
    cfg: EnvConfig,
    scheme: str,
    active: np.ndarray,
    h_true: np.ndarray,
    h_hat: np.ndarray,
    action: SlotAction | None = None,
) -> SchemeBudget:
    """Run one slot under ``scheme`` and measure what it consumed.

    Args:
        cfg: base environment config (``multiple_access`` is overridden).
        scheme: ``'rsma' | 'sdma' | 'oma' | 'noma'``.
        active: ``(K,)`` schedule shared by every scheme under audit.
        h_true / h_hat: the same channel realisation for every scheme.
        action: optional explicit action; defaults to an equal split with a
            0.2 common share (harmless for schemes without a common stream).
    """
    from dataclasses import replace

    scheme_cfg = replace(cfg, multiple_access=scheme)
    phy = RSMAPhysicalLayer(scheme_cfg)
    if action is None:
        k = cfg.num_users
        num_active = max(int(np.count_nonzero(active)), 1)
        pf = np.zeros(k + 1)
        pf[0] = 0.2
        pf[1:][active] = 0.8 / num_active
        rf = np.zeros(k)
        rf[active] = 1.0 / num_active
        action = SlotAction(active, pf, rf, 0.5)

    result = phy.transmit(h_true, h_hat, action)
    orthogonal = scheme in ORTHOGONAL_SCHEMES
    channel_uses = (
        float(np.sum(result.effective_blocklength))
        if orthogonal
        else float(cfg.blocklength)
    )
    return SchemeBudget(
        scheme=scheme,
        channel_uses=channel_uses,
        radiated_energy=float(result.total_power) * channel_uses,
        peak_power=float(result.total_power),
        offered_payload_bits=float(
            result.common_payload + np.sum(result.private_payload)
        ),
        num_scheduled=result.num_active,
        slot_blocklength=int(cfg.blocklength),
        bandwidth_channel_uses=int(cfg.blocklength),
    )


@dataclass
class ParityReport:
    """Outcome of a resource-parity audit across schemes."""

    budgets: dict[str, dict[str, Any]]
    mismatches: list[str] = field(default_factory=list)
    tolerance: float = 1e-6

    @property
    def passed(self) -> bool:
        return not self.mismatches

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise AssertionError(
                "resource parity audit failed:\n  " + "\n  ".join(self.mismatches)
            )


#: Geometry / weighting quantities that must be identical across schemes when
#: the cell is heterogeneous.  Unlike the resource fields these are not measured
#: from a transmission - they are properties of the scenario every algorithm is
#: handed, and a mismatch means the algorithms are not solving the same problem.
GEOMETRY_FIELDS = (
    "beta", "distances", "average_snr_db", "aoi_weights", "geometry_seed",
    "pathloss_model", "beta_normalization", "csit_error_mode", "payload_bits",
    "user_classes",
)


def geometry_fingerprint(cfg: EnvConfig) -> dict[str, Any]:
    """Everything about the scenario that every algorithm must share."""
    geometry = cfg.geometry
    return {
        "beta": [round(float(b), 12) for b in geometry.beta],
        "distances": (
            None if geometry.distances is None
            else [round(float(d), 9) for d in geometry.distances]
        ),
        "average_snr_db": [
            round(float(v), 9)
            for v in geometry.average_snr_db(cfg.total_power, cfg.noise_power)
        ],
        "aoi_weights": [round(float(w), 12) for w in cfg.weight_vector],
        "geometry_seed": int(cfg.geometry_seed),
        "pathloss_model": cfg.pathloss_model,
        "beta_normalization": cfg.beta_normalization,
        "csit_error_mode": cfg.csit_error_mode,
        "payload_bits": [round(float(d), 9) for d in cfg.payload_vector],
        "user_classes": list(cfg.class_vector),
        "disparity_db": round(float(geometry.disparity_db()), 9),
    }


def audit_geometry_parity(
    configs: dict[str, EnvConfig], reference: str | None = None
) -> ParityReport:
    """Check that a set of per-algorithm configs describe the same cell.

    Args:
        configs: ``label -> EnvConfig``, one per algorithm/arm.
        reference: label to compare against (default: the first).

    Returns:
        A :class:`ParityReport` whose ``budgets`` holds each arm's geometry
        fingerprint.
    """
    if not configs:
        raise ValueError("configs must be non-empty")
    labels = list(configs)
    reference = reference or labels[0]
    prints = {label: geometry_fingerprint(cfg) for label, cfg in configs.items()}
    base = prints[reference]
    mismatches: list[str] = []
    for label in labels:
        if label == reference:
            continue
        for field_name in GEOMETRY_FIELDS:
            if prints[label][field_name] != base[field_name]:
                mismatches.append(
                    f"{field_name}: {reference}={base[field_name]} vs "
                    f"{label}={prints[label][field_name]}"
                )
    return ParityReport(budgets=prints, mismatches=mismatches, tolerance=0.0)


#: Quantities that must match across schemes for the comparison to be fair.
PARITY_FIELDS = (
    "channel_uses",
    "radiated_energy",
    "peak_power",
    "offered_payload_bits",
    "slot_blocklength",
    "bandwidth_channel_uses",
    "num_scheduled",
)


def audit_resource_parity(
    cfg: EnvConfig,
    schemes: tuple[str, ...] = ("rsma", "sdma", "oma"),
    seed: int = 0,
    num_states: int = 32,
    tolerance: float = 1e-6,
    active: np.ndarray | None = None,
) -> ParityReport:
    """Check that every scheme consumes identical slot resources.

    Averages over ``num_states`` independent channel realisations so the audit
    is not an artefact of one draw.  The *same* realisation and the *same*
    schedule are given to every scheme.

    Returns:
        A :class:`ParityReport`; ``passed`` is False when any field in
        :data:`PARITY_FIELDS` differs by more than ``tolerance`` (relative).
    """
    from rsma_aoi.envs.channel import ChannelModel

    channel = ChannelModel.from_config(cfg)
    rng = np.random.default_rng(seed)
    if active is None:
        active = np.zeros(cfg.num_users, dtype=bool)
        active[: max(1, min(cfg.num_users, cfg.num_antennas))] = True

    totals: dict[str, list[SchemeBudget]] = {s: [] for s in schemes}
    for _ in range(num_states):
        h_true = channel.sample_initial(rng)
        h_hat = channel.estimate(h_true, rng)
        for scheme in schemes:
            totals[scheme].append(measure_scheme_budget(cfg, scheme, active, h_true, h_hat))

    averaged = {
        scheme: {
            field_name: float(np.mean([getattr(b, field_name) for b in budgets]))
            for field_name in PARITY_FIELDS
        }
        for scheme, budgets in totals.items()
    }
    for scheme in schemes:
        averaged[scheme]["scheme"] = scheme

    reference = schemes[0]
    mismatches: list[str] = []
    for field_name in PARITY_FIELDS:
        base = averaged[reference][field_name]
        for scheme in schemes[1:]:
            value = averaged[scheme][field_name]
            denom = max(abs(base), 1e-12)
            if abs(value - base) / denom > tolerance:
                mismatches.append(
                    f"{field_name}: {reference}={base:.6g} vs {scheme}={value:.6g}"
                )
    return ParityReport(budgets=averaged, mismatches=mismatches, tolerance=tolerance)


@dataclass
class ExperimentManifest:
    """Machine-readable record of what an experiment actually held constant."""

    name: str
    env_config: dict[str, Any]
    parity: dict[str, Any]
    geometry: dict[str, Any] = field(default_factory=dict)
    gain_convention: dict[str, str] = field(
        default_factory=lambda: {
            "gain_vs_sdma": "(AoI_SDMA - AoI_RSMA) / AoI_SDMA",
            "gain_vs_oma": "(AoI_OMA - AoI_RSMA) / AoI_OMA",
            "sign": "positive means RSMA is better (lower AoI)",
        }
    )
    assumptions: dict[str, Any] = field(
        default_factory=lambda: {
            "bandwidth": "single band; every scheme uses n channel uses per slot",
            "slot_duration": "n channel uses, identical for every scheme",
            "energy": "time-averaged power x channel uses; OMA transmits at peak "
                      "power for a fraction of the slot, giving equal slot energy",
            "channel_realisation": "identical per evaluation seed across schemes "
                                   "(the environment draws a fixed number of "
                                   "variates per slot regardless of the action)",
            "csit_realisation": "identical; drawn from the same generator state",
            "aoi_update_timing": "Delta(t+1) formed after the slot-t transmission "
                                 "for every scheme",
            "large_scale_fading": "h_k = sqrt(beta_k) g_k with beta_k fixed per "
                                  "geometry seed and identical for every algorithm",
            "aoi_weights": "identical across algorithms; see geometry.aoi_weights",
        }
    )
    extra: dict[str, Any] = field(default_factory=dict)
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    environment: dict[str, str] = field(
        default_factory=lambda: {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        return path


def build_manifest(
    name: str,
    cfg: EnvConfig,
    schemes: tuple[str, ...] = ("rsma", "sdma", "oma"),
    seed: int = 0,
    **extra: Any,
) -> tuple[ExperimentManifest, ParityReport]:
    """Audit parity and wrap the result in a manifest.

    The manifest records the resolved geometry (``beta_k``, distances, per-user
    average SNR, path-loss model, normalisation mode, AoI weights, geometry
    seed) because "SNR = 10 dB" is ambiguous without them.
    """
    report = audit_resource_parity(cfg, schemes=schemes, seed=seed)
    geometry = cfg.geometry
    manifest = ExperimentManifest(
        name=name,
        env_config=cfg.to_dict(),
        parity={"passed": report.passed, "mismatches": report.mismatches,
                "budgets": report.budgets, "tolerance": report.tolerance},
        geometry={
            **geometry.to_dict(cfg.total_power, cfg.noise_power),
            "geometry_seed": cfg.geometry_seed,
            "csit_error_mode": cfg.csit_error_mode,
            "csit_error_std": cfg.csit_error_std,
            "aoi_weight_mode": cfg.aoi_weight_mode,
            "aoi_weights": list(cfg.weight_vector),
            "user_classes": list(cfg.class_vector),
        },
        extra=extra,
    )
    return manifest, report
