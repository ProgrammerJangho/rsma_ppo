"""Beam *directions* (unit-norm) for the private and common RSMA streams.

Convention
----------
Channels are stored row-wise as ``H[k, :] = h_k`` (shape ``(K, Nt)``) and a
precoder ``p`` enters the SINR through ``h_k^H p``.  It is therefore convenient
to work with ``G = conj(H)`` so that ``G @ p == [h_1^H p, ..., h_K^H p]``.

The policy never outputs a full complex precoder in M0-M2: it outputs *powers*
and the precoder is ``p_k = sqrt(P_k) * u_k`` with ``||u_k|| = 1`` produced here
from the *estimated* channel ``H_hat``.

Directions for inactive users are exactly zero.  Every routine is guarded
against rank-deficient / all-zero channels and never returns NaN or Inf.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _normalise(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """Return ``vec / ||vec||`` with a safe fallback for (near-)zero vectors."""
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm < _EPS:
        if fallback is not None:
            fb_norm = float(np.linalg.norm(fallback))
            if np.isfinite(fb_norm) and fb_norm >= _EPS:
                return fallback / fb_norm
        out = np.zeros_like(vec)
        out[0] = 1.0
        return out
    return vec / norm


def _active_indices(active: np.ndarray, num_users: int) -> np.ndarray:
    idx = np.flatnonzero(np.asarray(active).astype(bool))
    if idx.size == 0:
        return idx
    if idx.max() >= num_users:
        raise ValueError("active mask longer than the number of users")
    return idx


def mrt_directions(h_hat: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Maximum-ratio transmission directions.

    Args:
        h_hat: estimated channel, shape ``(K, Nt)`` complex.
        active: boolean mask, shape ``(K,)``.

    Returns:
        ``U`` of shape ``(K, Nt)``; row ``k`` is ``h_hat_k / ||h_hat_k||`` for
        active users and ``0`` otherwise.
    """
    k, _ = h_hat.shape
    out = np.zeros_like(h_hat)
    for i in _active_indices(active, k):
        out[i] = _normalise(h_hat[i])
    return out


def _regularised_inverse_directions(
    h_hat: np.ndarray, active: np.ndarray, reg: float
) -> np.ndarray:
    """Shared core of ZF (``reg = 0``) and RZF (``reg > 0``).

    Computes ``P = G_A^H (G_A G_A^H + reg I)^{-1}`` with ``G_A = conj(H_A)``,
    then normalises each column.  For ``reg = 0`` a pseudo-inverse is used so
    that overloaded (``K > Nt``) or rank-deficient channels stay finite.
    """
    k, nt = h_hat.shape
    out = np.zeros_like(h_hat)
    idx = _active_indices(active, k)
    if idx.size == 0:
        return out

    h_act = h_hat[idx]                      # (A, Nt)
    g = np.conj(h_act)                      # (A, Nt), rows are h_k^H
    gram = g @ np.conj(g).T                 # (A, A), entries h_i^H h_j
    gram = np.asarray(gram, dtype=np.complex128)
    if reg > 0.0:
        mat = gram + reg * np.eye(idx.size, dtype=np.complex128)
        try:
            inv = np.linalg.inv(mat)
        except np.linalg.LinAlgError:  # pragma: no cover - reg keeps it invertible
            inv = np.linalg.pinv(mat, rcond=1e-10)
    else:
        inv = np.linalg.pinv(gram, rcond=1e-10)

    precoders = np.conj(g).T @ inv          # (Nt, A)
    if not np.all(np.isfinite(precoders)):
        precoders = np.nan_to_num(precoders, nan=0.0, posinf=0.0, neginf=0.0)
    for col, i in enumerate(idx):
        out[i] = _normalise(precoders[:, col], fallback=h_hat[i])
    return out


def zf_directions(h_hat: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Zero-forcing directions (pseudo-inverse based, unit norm)."""
    return _regularised_inverse_directions(h_hat, active, reg=0.0)


def rzf_directions(h_hat: np.ndarray, active: np.ndarray, reg: float) -> np.ndarray:
    """Regularised zero-forcing directions with loading factor ``reg >= 0``."""
    if reg < 0.0:
        raise ValueError("RZF regularisation must be >= 0")
    return _regularised_inverse_directions(h_hat, active, reg=reg)


def default_rzf_regularization(num_active: int, noise_power: float, total_power: float) -> float:
    """Standard RZF loading ``A * sigma_n^2 / P_max`` (MMSE-optimal for equal power)."""
    if total_power <= 0.0:
        raise ValueError("total_power must be > 0")
    return float(max(num_active, 1)) * float(noise_power) / float(total_power)


def private_directions(
    h_hat: np.ndarray,
    active: np.ndarray,
    mode: str,
    noise_power: float,
    total_power: float,
    reg: float | None = None,
) -> np.ndarray:
    """Dispatch to MRT / ZF / RZF private beam directions.

    Args:
        h_hat: ``(K, Nt)`` estimated channel.
        active: ``(K,)`` boolean mask.
        mode: one of ``'mrt' | 'zf' | 'rzf'``.
        noise_power / total_power: used for the default RZF loading.
        reg: explicit RZF loading; ``None`` uses :func:`default_rzf_regularization`.

    Returns:
        ``(K, Nt)`` unit-norm rows for active users, zeros elsewhere.
    """
    if mode == "mrt":
        return mrt_directions(h_hat, active)
    if mode == "zf":
        return zf_directions(h_hat, active)
    if mode == "rzf":
        num_active = int(np.count_nonzero(np.asarray(active).astype(bool)))
        loading = (
            default_rzf_regularization(num_active, noise_power, total_power)
            if reg is None
            else float(reg)
        )
        return rzf_directions(h_hat, active, loading)
    raise ValueError(f"unknown private beam mode: {mode!r}")


def _weighted_principal_direction(
    h_act: np.ndarray, weights: np.ndarray, fallback: np.ndarray
) -> np.ndarray:
    """Dominant direction of ``sum_k weights[k] h_k h_k^H``."""
    covariance = (h_act.T * weights) @ np.conj(h_act)
    covariance = np.asarray(covariance, dtype=np.complex128)
    if not np.all(np.isfinite(covariance)):
        covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        _, vectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        return _normalise(fallback)
    return _normalise(vectors[:, -1], fallback=fallback)


def maxmin_reweighted_direction(
    h_hat: np.ndarray,
    active: np.ndarray,
    iterations: int = 30,
    damping: float = 0.3,
) -> np.ndarray:
    """Low-complexity diagnostic beam that raises weak estimated common gains.

    Starting at the principal beam, inverse-gain weights repeatedly form a
    weighted channel covariance.  The best iterate under
    ``min_k |h_hat_k^H u|^2`` is returned.  This safeguard guarantees that the
    *estimated-channel minimum gain* is no worse than the principal starting
    point; it does not guarantee optimal SINR, BLER, or AoI.
    """
    if iterations < 0:
        raise ValueError("iterations must be >= 0")
    if not 0.0 < damping <= 1.0:
        raise ValueError("damping must lie in (0, 1]")

    k, nt = h_hat.shape
    idx = _active_indices(active, k)
    if idx.size == 0:
        return np.zeros(nt, dtype=h_hat.dtype)

    h_act = h_hat[idx]
    strongest = h_act[int(np.argmax(np.linalg.norm(h_act, axis=1)))]
    weights = np.full(idx.size, 1.0 / idx.size, dtype=float)
    candidate = _weighted_principal_direction(h_act, weights, strongest)
    gains = np.abs(np.conj(h_act) @ candidate) ** 2
    best = candidate.copy()
    best_min_gain = float(np.min(gains))

    for _ in range(iterations):
        inverse = 1.0 / np.maximum(gains, _EPS)
        inverse /= float(np.sum(inverse))
        weights = (1.0 - damping) * weights + damping * inverse
        weights /= float(np.sum(weights))
        candidate = _weighted_principal_direction(h_act, weights, strongest)
        gains = np.abs(np.conj(h_act) @ candidate) ** 2
        min_gain = float(np.min(gains))
        if min_gain > best_min_gain:
            best = candidate.copy()
            best_min_gain = min_gain
    return np.asarray(best, dtype=h_hat.dtype)


def common_direction(
    h_hat: np.ndarray,
    active: np.ndarray,
    mode: str,
    *,
    maxmin_iterations: int = 30,
    maxmin_damping: float = 0.3,
) -> np.ndarray:
    """Unit-norm common-stream direction ``u_c``.

    ``'mrt_sum'``: normalised sum of the active users' unit channel directions.
    ``'principal'``: dominant right-singular vector of the active channel
    matrix, i.e. the maximiser of ``sum_{k in A} |h_k^H u|^2``.
    ``'maxmin'``: best-so-far safeguarded inverse-gain reweighted principal
    direction.  This is a diagnostic heuristic, not an exact max-min solver.

    Both reduce to MRT when a single user is active.  Returns zeros when no user
    is active.
    """
    k, nt = h_hat.shape
    idx = _active_indices(active, k)
    if idx.size == 0:
        return np.zeros(nt, dtype=h_hat.dtype)

    h_act = h_hat[idx]
    strongest = h_act[int(np.argmax(np.linalg.norm(h_act, axis=1)))]

    if mode == "mrt_sum":
        directions = np.stack([_normalise(row) for row in h_act], axis=0)
        return _normalise(directions.sum(axis=0), fallback=strongest)
    if mode == "principal":
        g = np.conj(h_act)  # rows h_k^H; maximise ||G v||^2
        if not np.all(np.isfinite(g)):
            g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            _, _, vh = np.linalg.svd(g, full_matrices=False)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            return _normalise(strongest)
        return _normalise(np.conj(vh[0, :]), fallback=strongest)
    if mode == "maxmin":
        return maxmin_reweighted_direction(
            h_hat, active, iterations=maxmin_iterations, damping=maxmin_damping
        )
    raise ValueError(f"unknown common beam mode: {mode!r}")
