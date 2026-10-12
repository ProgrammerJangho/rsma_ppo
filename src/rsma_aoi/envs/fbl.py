"""Finite-blocklength (FBL) error probability via the normal approximation.

For an AWGN channel with SINR ``gamma``, blocklength ``n`` (channel uses) and a
payload of ``d`` bits, the Polyanskiy-Poor-Verdu normal approximation gives

    C(gamma) = log2(1 + gamma)                          [bits/channel use]
    V(gamma) = (1 - 1/(1+gamma)^2) * (log2 e)^2         [bits^2/channel use]
    eps      = Q( sqrt(n) * (C(gamma) - d/n) / sqrt(V(gamma)) )

and, solved for the rate,

    R(gamma, n, eps) = C(gamma) - sqrt(V(gamma)/n) * Q^{-1}(eps).

ASSUMPTIONS
-----------
* Interference is treated as Gaussian noise, so the AWGN dispersion formula is
  applied to the *SINR* of each stream (standard in the FBL RSMA literature).
* No CRC / header overhead is modelled; ``d`` is the information payload.
* Q(.) is the Gaussian tail; ``scipy.special.ndtr`` is used for accuracy in the
  far tails.

All routines are vectorised over NumPy arrays and clip inputs so that they never
produce NaN/Inf.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri

#: Numerical guards.  ``GAMMA_MIN`` avoids ``log(0)``; ``BLER_MIN/MAX`` keep
#: success probabilities strictly inside (0, 1) so that logs stay finite.
GAMMA_MIN = 1e-12
V_MIN = 1e-12
BLER_MIN = 1e-12
BLER_MAX = 1.0 - 1e-12
LOG2E = float(np.log2(np.e))


def q_function(x: np.ndarray | float) -> np.ndarray:
    """Gaussian tail ``Q(x) = P(Z > x)`` for standard normal ``Z``."""
    return ndtr(-np.asarray(x, dtype=np.float64))


def q_function_inv(eps: np.ndarray | float) -> np.ndarray:
    """Inverse Gaussian tail ``Q^{-1}(eps)``; ``eps`` clipped to (0, 1)."""
    e = np.clip(np.asarray(eps, dtype=np.float64), BLER_MIN, BLER_MAX)
    return -ndtri(e)


def shannon_capacity(gamma: np.ndarray | float) -> np.ndarray:
    """``C(gamma) = log2(1 + gamma)`` in bits per channel use."""
    g = np.clip(np.asarray(gamma, dtype=np.float64), GAMMA_MIN, None)
    return np.log2(1.0 + g)


def channel_dispersion(gamma: np.ndarray | float) -> np.ndarray:
    """``V(gamma) = (1 - (1+gamma)^{-2}) (log2 e)^2``, floored at ``V_MIN``."""
    g = np.clip(np.asarray(gamma, dtype=np.float64), GAMMA_MIN, None)
    v = (1.0 - 1.0 / np.square(1.0 + g)) * LOG2E**2
    return np.clip(v, V_MIN, None)


def fbl_bler(
    gamma: np.ndarray | float,
    blocklength: int | np.ndarray,
    payload_bits: np.ndarray | float,
) -> np.ndarray:
    """Block error rate of the normal approximation.

    Args:
        gamma: SINR (linear), any shape.
        blocklength: ``n`` channel uses, scalar or broadcastable.
        payload_bits: ``d`` information bits, broadcastable to ``gamma``.

    Returns:
        BLER array clipped to ``[BLER_MIN, BLER_MAX]``; entries with
        ``payload_bits <= 0`` are exactly 0 (nothing has to be decoded).
    """
    g = np.clip(np.asarray(gamma, dtype=np.float64), GAMMA_MIN, None)
    n = np.asarray(blocklength, dtype=np.float64)
    d = np.asarray(payload_bits, dtype=np.float64)
    if np.any(n <= 0):
        raise ValueError("blocklength must be > 0")

    capacity = shannon_capacity(g)
    dispersion = channel_dispersion(g)
    rate = d / n
    arg = np.sqrt(n) * (capacity - rate) / np.sqrt(dispersion)
    bler = q_function(arg)
    bler = np.clip(bler, BLER_MIN, BLER_MAX)
    bler = np.where(d > 0.0, bler, 0.0)
    if not np.all(np.isfinite(bler)):  # pragma: no cover - defensive
        raise FloatingPointError("fbl_bler produced non-finite values")
    return bler


def fbl_rate(
    gamma: np.ndarray | float, blocklength: int | np.ndarray, target_bler: float
) -> np.ndarray:
    """Achievable FBL rate (bits/channel use) at a target BLER, floored at 0."""
    g = np.clip(np.asarray(gamma, dtype=np.float64), GAMMA_MIN, None)
    n = np.asarray(blocklength, dtype=np.float64)
    if np.any(n <= 0):
        raise ValueError("blocklength must be > 0")
    rate = shannon_capacity(g) - np.sqrt(channel_dispersion(g) / n) * q_function_inv(target_bler)
    return np.clip(rate, 0.0, None)


def fbl_max_payload(
    gamma: np.ndarray | float, blocklength: int, target_bler: float
) -> np.ndarray:
    """Largest payload (bits) supported at ``target_bler`` for the given ``n``."""
    return fbl_rate(gamma, blocklength, target_bler) * float(blocklength)


def shannon_bler(
    gamma: np.ndarray | float,
    blocklength: int,
    payload_bits: np.ndarray | float,
) -> np.ndarray:
    """Infinite-blocklength limit: outage indicator ``1{d/n > C(gamma)}``.

    Returned values are clipped to ``[BLER_MIN, BLER_MAX]`` so that downstream
    log-likelihood / entropy computations stay finite.
    """
    g = np.asarray(gamma, dtype=np.float64)
    d = np.asarray(payload_bits, dtype=np.float64)
    rate = d / float(blocklength)
    outage = (rate > shannon_capacity(g)).astype(np.float64)
    bler = np.clip(outage, BLER_MIN, BLER_MAX)
    return np.where(d > 0.0, bler, 0.0)


def fixed_bler(shape: tuple[int, ...], success_prob: float) -> np.ndarray:
    """Constant-BLER link model used by the M0 debug configuration."""
    if not 0.0 <= success_prob <= 1.0:
        raise ValueError("success_prob must lie in [0, 1]")
    return np.full(shape, np.clip(1.0 - success_prob, BLER_MIN, BLER_MAX), dtype=np.float64)


def link_bler(
    link_model: str,
    gamma: np.ndarray,
    blocklength: int,
    payload_bits: np.ndarray,
    fixed_success_prob: float = 0.8,
) -> np.ndarray:
    """Dispatch to the configured link model (``'fbl' | 'shannon' | 'fixed'``)."""
    if link_model == "fbl":
        return fbl_bler(gamma, blocklength, payload_bits)
    if link_model == "shannon":
        return shannon_bler(gamma, blocklength, payload_bits)
    if link_model == "fixed":
        out = fixed_bler(np.shape(gamma), fixed_success_prob)
        return np.where(np.asarray(payload_bits, dtype=np.float64) > 0.0, out, 0.0)
    raise ValueError(f"unknown link model: {link_model!r}")
