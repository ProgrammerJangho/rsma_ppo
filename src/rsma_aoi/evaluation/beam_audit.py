"""Paired common-beam allocation audit for the FBL-AoI objective.

Every beam receives the same CEM budget and random seed.  Candidate selection
uses only ``h_hat`` in the default estimated-CSIT mode; ``h_true`` is reserved
for final scoring.  An independently optimised common-off SDMA boundary is
also offered as a safeguarded candidate, selected on the same information.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

import numpy as np

from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer
from rsma_aoi.optimizers import AllocationProblem, CEMAllocator, make_problem
from rsma_aoi.utils.config import EnvConfig


def _estimated_gain(problem: AllocationProblem, latent: np.ndarray) -> float:
    return float(problem.evaluate(latent[None, :])[0])


def _boundary_allocation(power_fractions: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Map an SDMA optimiser output to the exact ``P_c=D_c=0`` RSMA boundary."""
    private = np.clip(np.asarray(power_fractions[1:], dtype=float), 0.0, None)
    private = np.where(active, private, 0.0)
    if private.sum() <= 1e-12:
        private = active.astype(float)
    private /= private.sum()
    return np.concatenate([[0.0], private])


def audit_state_allocations(
    cfg: EnvConfig,
    aoi: np.ndarray,
    h_true: np.ndarray,
    h_hat: np.ndarray,
    active: np.ndarray,
    *,
    beam_modes: Iterable[str] = ("principal", "mrt_sum", "maxmin"),
    csit_mode: str = "estimated",
    seed: int = 0,
    iterations: int = 5,
    population: int = 192,
    restarts: int = 1,
) -> list[dict[str, Any]]:
    """Re-optimise allocation for each beam on one paired state and schedule.

    The returned ``raw_*`` fields describe the beam's CEM solution.  The
    ``safe_*`` fields describe an information-feasible selector that compares
    that solution with a separately optimised SDMA boundary on ``h_hat`` (or on
    ``h_true`` only when ``csit_mode='genie'``).
    """
    active = np.asarray(active, dtype=bool)
    if int(active.sum()) < 2:
        raise ValueError("beam audit requires at least two active users")
    if cfg.multiple_access != "rsma":
        raise ValueError("beam audit requires an RSMA environment config")

    aoi_limit = float(cfg.aoi_max) if cfg.aoi_cap_enabled else np.inf
    allocator = CEMAllocator(
        iterations=iterations,
        population=population,
        restarts=restarts,
        label=f"paired_cem({restarts}x{iterations}x{population})",
    )

    sdma_cfg = replace(cfg, multiple_access="sdma", allocator_mode="none")
    sdma_phy = RSMAPhysicalLayer(sdma_cfg)
    sdma_problem, base = make_problem(
        sdma_phy, aoi, h_true, h_hat, np.asarray(cfg.weight_vector),
        aoi_limit, active, csit_mode=csit_mode,
    )
    sdma = allocator.solve(sdma_problem, np.random.default_rng(seed))
    if sdma.latent is None:  # pragma: no cover - CEM always returns a latent
        raise RuntimeError("CEM did not return its selected latent")
    sdma_estimated_gain = _estimated_gain(sdma_problem, sdma.latent)
    boundary_pf = _boundary_allocation(sdma.power_fractions, active)
    boundary_rf = active.astype(float) / active.sum()

    rows: list[dict[str, Any]] = []
    for mode in beam_modes:
        beam_cfg = replace(
            cfg, common_beam=str(mode), multiple_access="rsma", allocator_mode="none"
        )
        beam_cfg.validate()
        phy = RSMAPhysicalLayer(beam_cfg)
        problem, beam_base = make_problem(
            phy, aoi, h_true, h_hat, np.asarray(cfg.weight_vector),
            aoi_limit, active, csit_mode=csit_mode,
        )
        if abs(beam_base - base) > 1e-10:
            raise AssertionError("AoI base changed across paired beam problems")
        result = allocator.solve(problem, np.random.default_rng(seed))
        if result.latent is None:  # pragma: no cover - CEM always returns a latent
            raise RuntimeError("CEM did not return its selected latent")
        raw_estimated_gain = _estimated_gain(problem, result.latent)

        beams = problem.beams
        true_out = phy.evaluate_allocations(
            h_true, active, result.power_fractions[None, :],
            result.rate_fractions[None, :], np.asarray([result.rho_r]), beams=beams,
        )
        estimated_channel = h_true if csit_mode == "genie" else h_hat
        estimated_out = phy.evaluate_allocations(
            estimated_channel, active, result.power_fractions[None, :],
            result.rate_fractions[None, :], np.asarray([result.rho_r]), beams=beams,
        )
        if cfg.rate_split_mode == "per_user_fraction":
            boundary_rf = np.zeros_like(boundary_rf)
        boundary_out = phy.evaluate_allocations(
            h_true, active, boundary_pf[None, :], boundary_rf[None, :],
            np.asarray([0.0]), beams=beams,
        )
        boundary_identity_error = float(
            np.max(np.abs(boundary_out["success_prob"][0] - sdma.success_prob))
        )
        if boundary_identity_error > 1e-10:
            raise AssertionError(
                "P_c=D_c=0 RSMA boundary does not reproduce the SDMA PHY"
            )

        u_common = beams[1]
        estimated_common_gain = np.abs(np.conj(h_hat[active]) @ u_common) ** 2
        true_common_gain = np.abs(np.conj(h_true[active]) @ u_common) ** 2
        has_common = bool(
            true_out["common_power"][0] > 1e-12
            and true_out["common_payload"][0] > 1e-12
        )
        common_sinr = true_out["common_sinr"][0, active]
        common_bler = true_out["common_bler"][0, active]

        choose_boundary = sdma_estimated_gain > raw_estimated_gain
        safe_gain = sdma.gain if choose_boundary else result.gain
        safe_estimated_gain = (
            sdma_estimated_gain if choose_boundary else raw_estimated_gain
        )
        alternative_estimated_gain = (
            raw_estimated_gain if choose_boundary else sdma_estimated_gain
        )
        safe_pf = boundary_pf if choose_boundary else result.power_fractions
        safe_rf = boundary_rf if choose_boundary else result.rate_fractions
        safe_rho = 0.0 if choose_boundary else result.rho_r
        safe_success = sdma.success_prob if choose_boundary else result.success_prob
        rows.append({
            "beam": str(mode),
            "active_count": int(active.sum()),
            "base_expected_aoi": float(base),
            "raw_gain_true": float(result.gain),
            "raw_gain_estimated": raw_estimated_gain,
            "raw_expected_aoi_true": float(base - result.gain),
            "safe_gain_true": float(safe_gain),
            "safe_expected_aoi_true": float(base - safe_gain),
            "safe_gain_estimated": float(safe_estimated_gain),
            "safe_expected_aoi_estimated": float(base - safe_estimated_gain),
            "alternative_gain_estimated": float(alternative_estimated_gain),
            "alternative_expected_aoi_estimated": float(
                base - alternative_estimated_gain
            ),
            "selected_boundary": bool(choose_boundary),
            "sdma_gain_true": float(sdma.gain),
            "sdma_gain_estimated": sdma_estimated_gain,
            "boundary_identity_max_abs": boundary_identity_error,
            "common_power_fraction": float(result.power_fractions[0]),
            "rho": float(result.rho_r),
            "min_common_gain_estimated": float(np.min(estimated_common_gain)),
            "min_common_gain_true": float(np.min(true_common_gain)),
            "min_common_sinr_true": float(np.min(common_sinr)) if has_common else np.nan,
            "worst_common_bler_true": float(np.max(common_bler)) if has_common else np.nan,
            "mean_common_bler_true": float(np.mean(common_bler)) if has_common else np.nan,
            "mean_success_true": float(np.mean(true_out["success_prob"][0, active])),
            "mean_success_estimated": float(
                np.mean(estimated_out["success_prob"][0, active])
            ),
            "evaluations": int(result.evaluations),
            "runtime_s": float(result.runtime_s),
            # Full allocations are retained for behaviour-cloning labels. CSV
            # audit scripts remove them before serialisation.
            "raw_power_fractions": result.power_fractions.copy(),
            "raw_rate_fractions": result.rate_fractions.copy(),
            "raw_success_probabilities": result.success_prob.copy(),
            "safe_power_fractions": np.asarray(safe_pf).copy(),
            "safe_rate_fractions": np.asarray(safe_rf).copy(),
            "safe_rho": float(safe_rho),
            "safe_success_probabilities": np.asarray(safe_success).copy(),
            # Retained for a direct boundary-identity check in tests/audits.
            "boundary_power_fractions": boundary_pf,
            "boundary_rate_fractions": boundary_rf,
        })
    return rows
