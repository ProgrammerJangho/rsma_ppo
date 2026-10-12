"""Myopic scheduling/allocation oracle shared by the diagnostic and the dataset.

For one slot state the oracle enumerates the non-empty user subsets and, for
each, approximately maximises

    gain(A, alloc) = sum_k urgency_k * p_k(A, alloc),
    urgency_k      = w_k * (min(Delta_k + 1, Delta_max) - 1)

with a cross-entropy method (CEM) search **in the policy's own latent space**
(masked softmax over power / rate logits, sigmoid for ``rho_r``).  Maximising
``gain`` is equivalent to minimising the expected next weighted AoI

    sum_k w_k E[Delta_k(t+1)] = sum_k w_k aged_k - gain,

because ``E[Delta_k(t+1)] = p_k + (1 - p_k) aged_k``.

Success probabilities come from :meth:`RSMAPhysicalLayer.evaluate_allocations`,
which the test suite pins to be identical to the ``transmit`` path the
environment simulates.

MYOPIC: this is a finite-budget one-step reference, not a bound on stationary
long-horizon performance or an optimal policy.  Imitating it is a warm start
rather than a target.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer
from rsma_aoi.utils.config import EnvConfig
from rsma_aoi.utils.action_mapping import hard_sigmoid_np

#: CSIT modes: optimise against the true channel, or against the estimate only.
CSIT_MODES = ("genie", "estimated")


def config_hash(cfg: EnvConfig) -> str:
    """Stable short hash of an environment config (dataset provenance)."""
    data = cfg.to_dict()
    # Preserve hashes of datasets created before the optional action mapping
    # existed.  In direct mode these fields have no effect on the PHY or labels.
    if data.get("common_payload_mapping") == "direct":
        data.pop("common_payload_mapping", None)
        data.pop("common_target_bler", None)
    # Added after the first oracle datasets were frozen.  The default preserves
    # their environment identity; a non-default reward still changes the hash.
    if data.get("reward_mode") == "sampled_aoi":
        data.pop("reward_mode", None)
    payload = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def enumerate_masks(num_users: int, sizes: tuple[int, ...] | None = None) -> list[np.ndarray]:
    """All non-empty subsets as boolean masks, optionally restricted to ``sizes``."""
    masks = [
        np.array(bits, dtype=bool)
        for bits in itertools.product([False, True], repeat=num_users)
        if any(bits)
    ]
    if sizes is not None:
        masks = [m for m in masks if int(m.sum()) in sizes]
    return masks


def masked_softmax_np(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Row-wise softmax over ``mask``; masked entries are exactly 0."""
    filled = np.where(mask[None, :], logits, -np.inf)
    filled = filled - filled.max(axis=1, keepdims=True)
    exp = np.where(mask[None, :], np.exp(filled), 0.0)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-30, None)


def decode_latents(sample: np.ndarray, active: np.ndarray, num_users: int, rate_split_mode: str = "simplex_scaled"):
    """Latent rows -> ``(power simplex, rate simplex, rho_r)``.

    Uses exactly the transform the policy applies, so an oracle label is a
    valid target for the policy's own heads.
    """
    k = num_users
    if rate_split_mode == "per_user_fraction":
        if sample.shape[1] != 2*k + 1:
            raise ValueError("direct-fraction latent width must be 2K+1")
        alpha = hard_sigmoid_np(sample[:, :1])
        private = masked_softmax_np(sample[:, 1:k+1], active)
        return (
            np.concatenate([alpha, (1-alpha)*private], axis=1),
            hard_sigmoid_np(sample[:, k+1:]) * active,
            np.ones(len(sample)),
        )
    power_mask = np.concatenate([[True], active])
    pf = masked_softmax_np(sample[:, : k + 1], power_mask)
    rf = masked_softmax_np(sample[:, k + 1 : 2 * k + 1], active)
    rho = 1.0 / (1.0 + np.exp(-sample[:, -1]))
    return pf, rf, rho


@dataclass
class SubsetSolution:
    """Best allocation found for one subset."""

    active: np.ndarray
    gain: float
    expected_aoi: float
    power_fractions: np.ndarray
    rate_fractions: np.ndarray
    rho_r: float
    success_prob: np.ndarray
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def num_active(self) -> int:
        return int(self.active.sum())


def optimise_subset(
    phy: RSMAPhysicalLayer,
    h_score: np.ndarray,
    h_design: np.ndarray,
    h_opt: np.ndarray,
    active: np.ndarray,
    urgency: np.ndarray,
    rng: np.random.Generator,
    iterations: int = 5,
    population: int = 192,
    elite_frac: float = 0.15,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """CEM search for the allocation maximising ``sum_k urgency_k * p_k``.

    Args:
        phy: physical layer (beams, SINRs, FBL BLERs).
        h_score: channel the final candidate is scored on (the true one).
        h_design: channel the beamformers are designed from (``h_hat``).
        h_opt: channel the search optimises against (``h_true`` for genie,
            ``h_hat`` for the estimated-CSIT oracle).
        active: ``(K,)`` subset under test.
        urgency: ``(K,)`` linearised objective weights.
        rng: explicit generator.
        iterations / population / elite_frac: CEM settings.

    Returns:
        ``(gain, power_fractions, rate_fractions, success_prob, rho_r)`` where
        ``gain`` is evaluated on ``h_score``.
    """
    k = phy.num_users
    dim = 2 * k + (1 if phy.cfg.rate_split_mode == "per_user_fraction" else 2)
    beams = phy.rsma_beams(h_design, active, use_common=True)
    mu = np.zeros(dim)
    sigma = np.full(dim, 1.5)
    n_elite = max(4, int(population * elite_frac))
    best = (-np.inf, None, None, None)

    for _ in range(iterations):
        sample = mu[None, :] + sigma[None, :] * rng.standard_normal((population, dim))
        pf, rf, rho = decode_latents(sample, active, k, phy.cfg.rate_split_mode)
        out = phy.evaluate_allocations(h_opt, active, pf, rf, rho, beams=beams)
        gain = out["success_prob"] @ urgency
        order = np.argsort(-gain)
        elite = sample[order[:n_elite]]
        mu = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.05)
        top = order[0]
        if gain[top] > best[0]:
            best = (float(gain[top]), pf[top], rf[top], rho[top])

    _, pf_best, rf_best, rho_best = best
    scored = phy.evaluate_allocations(
        h_score, active, pf_best[None, :], rf_best[None, :], np.asarray([rho_best]), beams=beams
    )
    success = scored["success_prob"][0]
    return float(success @ urgency), pf_best, rf_best, success, float(rho_best)


def urgency_vector(
    aoi: np.ndarray, weights: np.ndarray, aoi_max: float
) -> tuple[np.ndarray, float]:
    """Return ``(urgency, base)`` where ``expected_aoi = base - gain``."""
    aged = np.minimum(aoi + 1.0, aoi_max)
    return weights * (aged - 1.0), float(weights @ aged)


def solve_state(
    phy: RSMAPhysicalLayer,
    aoi: np.ndarray,
    h_true: np.ndarray,
    h_hat: np.ndarray,
    weights: np.ndarray,
    aoi_max: float,
    csit_mode: str = "estimated",
    rng: np.random.Generator | None = None,
    sizes: tuple[int, ...] | None = None,
    **cem_kwargs: Any,
) -> list[SubsetSolution]:
    """Solve every (allowed) subset for one state, best-first.

    Returns:
        A list of :class:`SubsetSolution` sorted by increasing expected AoI, so
        element 0 is the oracle choice and element 1 the runner-up.
    """
    if csit_mode not in CSIT_MODES:
        raise ValueError(f"csit_mode must be one of {CSIT_MODES}")
    rng = rng if rng is not None else np.random.default_rng(0)
    urgency, base = urgency_vector(aoi, weights, aoi_max)
    h_opt = h_true if csit_mode == "genie" else h_hat

    solutions = []
    for active in enumerate_masks(phy.num_users, sizes):
        gain, pf, rf, success, rho = optimise_subset(
            phy, h_true, h_hat, h_opt, active, urgency, rng, **cem_kwargs
        )
        solutions.append(
            SubsetSolution(
                active=active,
                gain=gain,
                expected_aoi=base - gain,
                power_fractions=pf,
                rate_fractions=rf,
                rho_r=rho,
                success_prob=success,
            )
        )
    solutions.sort(key=lambda s: s.expected_aoi)
    return solutions


def expected_aoi_of_action(
    phy: RSMAPhysicalLayer,
    aoi: np.ndarray,
    h_true: np.ndarray,
    h_hat: np.ndarray,
    weights: np.ndarray,
    aoi_max: float,
    active: np.ndarray,
    power_fractions: np.ndarray,
    rate_fractions: np.ndarray,
    rho_r: float,
) -> float:
    """Expected next weighted AoI of a *given* action (used to score BC policies)."""
    urgency, base = urgency_vector(aoi, weights, aoi_max)
    out = phy.evaluate_allocations(
        h_true,
        np.asarray(active, dtype=bool),
        np.asarray(power_fractions, dtype=np.float64)[None, :],
        np.asarray(rate_fractions, dtype=np.float64)[None, :],
        np.asarray([float(rho_r)]),
        h_hat=h_hat,
    )
    return float(base - out["success_prob"][0] @ urgency)
