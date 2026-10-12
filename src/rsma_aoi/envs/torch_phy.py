"""Differentiable (PyTorch) mirror of the RSMA finite-blocklength physical layer.

Used by the objective-aware allocator: it lets a network be trained on the
*outcome* of its allocation (expected next-slot weighted AoI) rather than on an
L1 distance to a reference action.  Under FBL the BLER is nearly a step function
of SINR, so a small action error can flip a user between success and failure -
action imitation therefore optimises the wrong thing.

Numerical care
--------------
* ``Q(x) = ndtr(-x)`` via :func:`torch.special.ndtr`.
* SINR and dispersion are floored exactly as in the NumPy PHY.
* The BLER is clamped only at ``[1e-12, 1-1e-12]``; inside that range the
  gradient flows.  ``ndtr`` *does* saturate for ``|x| > ~8``, which zeroes the
  gradient far from the decoding threshold - that is precisely why
  :func:`fbl_margin` is exposed: the pre-Q margin has a useful gradient
  everywhere and is what the margin loss uses.

Equivalence with the NumPy/SciPy implementation is asserted in the test suite.
"""

from __future__ import annotations

import math

import torch

GAMMA_MIN = 1e-12
V_MIN = 1e-12
BLER_MIN = 1e-12
BLER_MAX = 1.0 - 1e-12
LOG2E = math.log2(math.e)
_NEG_INF = -1e9


def torch_q(x: torch.Tensor) -> torch.Tensor:
    """Gaussian tail ``Q(x) = P(Z > x)``."""
    return torch.special.ndtr(-x)


def shannon_capacity(gamma: torch.Tensor) -> torch.Tensor:
    """``log2(1 + gamma)`` in bits per channel use."""
    return torch.log2(1.0 + gamma.clamp_min(GAMMA_MIN))


def channel_dispersion(gamma: torch.Tensor) -> torch.Tensor:
    """``(1 - (1+gamma)^-2) (log2 e)^2``, floored at ``V_MIN``."""
    g = gamma.clamp_min(GAMMA_MIN)
    return ((1.0 - 1.0 / (1.0 + g) ** 2) * LOG2E**2).clamp_min(V_MIN)


def fbl_margin(
    gamma: torch.Tensor, blocklength: torch.Tensor | float, payload_bits: torch.Tensor
) -> torch.Tensor:
    """Pre-``Q`` decoding margin ``sqrt(n) (C - d/n) / sqrt(V)``.

    Positive means the rate is below capacity.  Unlike the BLER this does not
    saturate, so it is the quantity to use in a shaping loss.
    """
    g = gamma.clamp_min(GAMMA_MIN)
    n = torch.as_tensor(blocklength, dtype=g.dtype, device=g.device)
    # A per-state blocklength vector must broadcast over user dimensions.
    # Scalars and explicitly shaped tensors retain their historical behaviour.
    while n.ndim < g.ndim:
        n = n.unsqueeze(-1)
    capacity = shannon_capacity(g)
    dispersion = channel_dispersion(g)
    return torch.sqrt(n) * (capacity - payload_bits / n) / torch.sqrt(dispersion)


def fbl_bler(
    gamma: torch.Tensor, blocklength: torch.Tensor | float, payload_bits: torch.Tensor
) -> torch.Tensor:
    """Differentiable normal-approximation BLER; zero where the payload is zero."""
    bler = torch_q(fbl_margin(gamma, blocklength, payload_bits))
    bler = bler.clamp(BLER_MIN, BLER_MAX)
    return torch.where(payload_bits > 0.0, bler, torch.zeros_like(bler))


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over the unmasked coordinates; masked entries are exactly 0."""
    mask_f = mask.to(logits.dtype)
    filled = torch.where(mask, logits, torch.full_like(logits, _NEG_INF))
    filled = filled - filled.max(dim=-1, keepdim=True).values
    exp = torch.exp(filled) * mask_f
    denom = exp.sum(dim=-1, keepdim=True)
    return torch.where(denom > 0, exp / denom.clamp_min(1e-30), torch.zeros_like(exp)) * mask_f


def split_payload(
    payload: torch.Tensor,
    active: torch.Tensor,
    rate_fractions: torch.Tensor,
    rho_r: torch.Tensor,
    rate_split_mode: str = "simplex_scaled",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable payload split; direct per-user fractions are canonical.

    Mirrors :func:`rsma_aoi.envs.rsma_phy.split_payload_batch`:
    ``D_c_tot = rho_r * min_k(D_k / c_k)``, ``D_c,k = c_k D_c_tot``.

    Args:
        payload: ``(K,)`` or ``(N, K)`` per-user message sizes.
        active: ``(N, K)`` boolean mask.
        rate_fractions: ``(N, K)`` simplex rows.
        rho_r: ``(N,)`` scale in [0, 1].

    Returns:
        ``(D_c_tot (N,), D_p (N, K))``.
    """
    payload = payload.expand_as(rate_fractions) if payload.dim() == 1 else payload
    if rate_split_mode == "per_user_fraction":
        d_c = torch.where(active, rate_fractions.clamp(0, 1) * payload, torch.zeros_like(rate_fractions))
        return d_c.sum(-1), torch.where(active, payload - d_c, torch.zeros_like(d_c))
    if rate_split_mode != "simplex_scaled":
        raise ValueError("unknown rate_split_mode")
    usable = active & (rate_fractions > 1e-12)
    ratio = torch.where(
        usable, payload / rate_fractions.clamp_min(1e-12),
        torch.full_like(rate_fractions, float("inf")),
    )
    cap = ratio.min(dim=-1).values
    cap = torch.where(torch.isfinite(cap), cap, torch.zeros_like(cap))
    d_c_tot = rho_r.clamp(0.0, 1.0) * cap
    d_c = torch.where(active, rate_fractions * d_c_tot.unsqueeze(-1), torch.zeros_like(rate_fractions))
    d_c = torch.minimum(d_c, torch.where(active, payload, torch.zeros_like(payload)))
    d_p = torch.where(active, payload - d_c, torch.zeros_like(payload)).clamp_min(0.0)
    return d_c_tot, d_p


def rsma_success_probability(
    gain_priv: torch.Tensor,
    gain_common: torch.Tensor,
    active: torch.Tensor,
    power_fractions: torch.Tensor,
    rate_fractions: torch.Tensor,
    rho_r: torch.Tensor,
    payload: torch.Tensor,
    blocklength: float,
    noise_power: float,
    total_power: float | torch.Tensor,
    use_common: bool = True,
    rate_split_mode: str = "simplex_scaled",
) -> dict[str, torch.Tensor]:
    """Differentiable per-user success probability for 1-layer RSMA.

    Args:
        gain_priv: ``(N, K, K)`` (or ``(K, K)``) values ``|h_k^H u_j|^2``.
        gain_common: ``(N, K)`` values ``|h_k^H u_c|^2``.
        active: ``(N, K)`` boolean schedule.
        power_fractions: ``(N, K+1)`` simplex ``[common, private_1..K]``.
        rate_fractions: ``(N, K)`` common-rate simplex.
        rho_r: ``(N,)`` common-payload scale.
        payload: ``(K,)`` or ``(N, K)`` message sizes in bits.
        blocklength / noise_power: PHY constants. ``total_power`` is a scalar
            or one value per batch row for episode-randomized SNR.
        use_common: False evaluates the SDMA variant.

    Returns:
        Dict with ``success_prob``, ``common_sinr``, ``private_sinr``,
        ``common_bler``, ``private_bler``, ``common_margin``, ``private_margin``.
    """
    if gain_priv.dim() == 2:
        gain_priv = gain_priv.unsqueeze(0).expand(power_fractions.shape[0], -1, -1)
    pf = power_fractions
    if not use_common:
        pf = torch.cat([torch.zeros_like(pf[:, :1]), pf[:, 1:]], dim=-1)
        total = pf.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        pf = pf / total
    total_power_t = torch.as_tensor(total_power, dtype=pf.dtype, device=pf.device)
    if total_power_t.ndim == 0:
        total_power_t = total_power_t.expand(pf.shape[0])
    else:
        total_power_t = total_power_t.reshape(-1)
        if total_power_t.shape[0] != pf.shape[0]:
            raise ValueError("total_power must be scalar or have one value per batch row")
    power_common = pf[:, 0] * total_power_t
    power_private = pf[:, 1:] * total_power_t[:, None]

    interference_all = torch.einsum("nkj,nj->nk", gain_priv, power_private)
    own = power_private * torch.diagonal(gain_priv, dim1=-2, dim2=-1)
    interference_other = (interference_all - own).clamp_min(0.0)

    zeros = torch.zeros_like(own)
    gamma_c = (
        power_common.unsqueeze(-1) * gain_common / (interference_all + noise_power)
        if use_common
        else zeros
    )
    gamma_p = own / (interference_other + noise_power)
    gamma_c = torch.where(active, gamma_c, zeros)
    gamma_p = torch.where(active, gamma_p, zeros)

    if use_common:
        d_c_tot, d_p = split_payload(payload, active, rate_fractions, rho_r, rate_split_mode)
    else:
        d_c_tot = torch.zeros(pf.shape[0], dtype=pf.dtype, device=pf.device)
        d_p = torch.where(active, payload.expand_as(gamma_p), torch.zeros_like(gamma_p))

    common_payload = d_c_tot.unsqueeze(-1).expand_as(gamma_c)
    common_margin = fbl_margin(gamma_c, blocklength, common_payload)
    private_margin = fbl_margin(gamma_p, blocklength, d_p)
    bler_c = torch.where(
        active & (common_payload > 0.0),
        fbl_bler(gamma_c, blocklength, common_payload),
        torch.zeros_like(gamma_c),
    )
    bler_p = torch.where(active, fbl_bler(gamma_p, blocklength, d_p), torch.zeros_like(gamma_p))
    success = torch.where(active, (1.0 - bler_c) * (1.0 - bler_p), torch.zeros_like(gamma_p))
    return {
        "success_prob": success,
        "common_sinr": gamma_c,
        "private_sinr": gamma_p,
        "common_bler": bler_c,
        "private_bler": bler_p,
        "common_margin": common_margin,
        "private_margin": private_margin,
        "common_payload": d_c_tot,
        "private_payload": d_p,
    }


def channel_gains(h_true: torch.Tensor, u_priv: torch.Tensor, u_common: torch.Tensor):
    """``(|h_k^H u_j|^2, |h_k^H u_c|^2)`` for complex inputs.

    Args:
        h_true: ``(N, K, Nt)`` complex.
        u_priv: ``(N, K, Nt)`` complex unit-norm private directions.
        u_common: ``(N, Nt)`` complex unit-norm common direction.
    """
    proj = torch.einsum("nki,nji->nkj", torch.conj(h_true), u_priv)
    gain_priv = proj.abs() ** 2
    gain_common = torch.einsum("nki,ni->nk", torch.conj(h_true), u_common).abs() ** 2
    return gain_priv, gain_common
