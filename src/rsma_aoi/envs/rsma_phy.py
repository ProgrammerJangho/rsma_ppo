"""1-layer RSMA physical layer: power/payload mapping, SINRs, FBL BLERs.

Transmit signal (RSMA)::

    x = p_c s_c + sum_k p_k s_k,     p_c = sqrt(P_c) u_c,  p_k = sqrt(P_k) u_k

with unit-norm directions ``u`` from :mod:`rsma_aoi.envs.beamforming`, so the sum
power constraint is exactly ``P_c + sum_k P_k = rho_P * P_max <= P_max``.

Every active user first decodes the common stream, applies SIC and then decodes
its own private stream::

    gamma_c,k = |h_k^H p_c|^2 / (sum_j |h_k^H p_j|^2 + sigma_n^2)
    gamma_p,k = |h_k^H p_k|^2 / (sum_{j != k} |h_k^H p_j|^2 + sigma_n^2)

Rate split (canonical mode ``per_user_fraction``)
-------------------------------------------------
For fixed user payloads ``D_k``, the policy outputs independent fractions
``phi_k in [0,1]`` and the PHY applies

    D_c,k = phi_k D_k,  D_p,k = (1-phi_k)D_k,
    D_c_tot = sum_k D_c,k,  C_k = D_c,k / n.

All active users decode the complete ``D_c_tot``-bit common codeword.  The
endpoint-capable policy transform can output ``alpha=phi_k=0`` exactly, so the
SDMA boundary is available without a Bernoulli gate or a payload-scale action.

Legacy mode ``simplex_scaled``
------------------------------
User ``k`` has an update of ``D_k`` bits.  The agent outputs a masked-softmax
simplex ``c`` over the active users and a scalar ``rho_r in [0, 1]``.  The total
common payload is

    D_c_tot = rho_r * min_{k in A, c_k > 0} (D_k / c_k),
    D_c,k   = c_k * D_c_tot,   D_p,k = D_k - D_c,k.

The ``min`` makes ``0 <= D_c,k <= D_k`` hold *by construction* for every
``rho_r`` and every simplex ``c`` (no clipping, no constraint violation), while
still covering the whole feasible set for a given split direction ``c``.
``rho_r = 0`` is pure SDMA-like operation, ``rho_r = 1`` puts the largest
feasible amount of information on the common stream.

This legacy parameterisation remains readable only so archived experiments and
checkpoints retain their original meaning.

Success rule
------------
User ``k`` decodes the *whole* common stream (payload ``D_c_tot``) - it needs it
both for its own common part and for SIC - and then its private part::

    P_succ,k = (1 - eps_c,k) * (1 - eps_p,k)

with ``eps_c,k = 0`` when ``D_c_tot = 0`` and ``eps_p,k = 0`` when ``D_p,k = 0``.
Failures of *other* users on the common stream do not affect user ``k``.

Multiple-access variants (identical power budget, identical FBL model)
---------------------------------------------------------------------
* ``rsma``  : common + private streams as above.
* ``sdma``  : no common stream (``P_c = 0``, payload fully private, no SIC).
* ``oma``   : the ``n`` channel uses are split orthogonally among active users;
  each transmits alone with MRT at the full instantaneous power (equal total
  energy ``P_max * n``), so there is no inter-user interference.
* ``noma``  : all active users superposed on a single beam with SC-SIC in
  ascending effective-gain order.

The ``link_model='fixed'`` debug mode overrides all of the above with a constant
*end-to-end* per-user success probability (see
:meth:`RSMAPhysicalLayer._fixed_link_override`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rsma_aoi.envs import beamforming as bf
from rsma_aoi.envs.fbl import BLER_MAX, BLER_MIN, link_bler
from rsma_aoi.utils.config import EnvConfig

_EPS = 1e-12


# ---------------------------------------------------------------------------
# action / result containers
# ---------------------------------------------------------------------------
@dataclass
class SlotAction:
    """One slot's already-transformed, feasible action.

    Attributes:
        active: ``(K,)`` boolean scheduling mask.
        power_fractions: ``(K+1,)`` non-negative, ``[common, private_1..K]``,
            summing to 1 (entries of inactive users are exactly 0).
        rate_fractions: canonical ``phi_k`` vector, or a legacy rate simplex.
        common_payload_scale: legacy ``rho_r``; ignored in canonical mode.
        power_scale: ``rho_P in (0, 1]`` fraction of ``P_max`` actually radiated.
    """

    active: np.ndarray
    power_fractions: np.ndarray
    rate_fractions: np.ndarray
    common_payload_scale: float = 1.0
    power_scale: float = 1.0
    rate_split_mode: str = "simplex_scaled"

    def __post_init__(self) -> None:
        self.active = np.asarray(self.active, dtype=bool).reshape(-1)
        self.power_fractions = np.asarray(self.power_fractions, dtype=np.float64).reshape(-1)
        self.rate_fractions = np.asarray(self.rate_fractions, dtype=np.float64).reshape(-1)
        k = self.active.shape[0]
        if self.power_fractions.shape[0] != k + 1:
            raise ValueError("power_fractions must have length K+1")
        if self.rate_fractions.shape[0] != k:
            raise ValueError("rate_fractions must have length K")
        self.common_payload_scale = float(self.common_payload_scale)
        self.power_scale = float(self.power_scale)

    @property
    def num_users(self) -> int:
        return int(self.active.shape[0])


@dataclass
class SlotResult:
    """Per-slot PHY outcome.  All arrays have shape ``(K,)`` unless stated."""

    active: np.ndarray
    common_sinr: np.ndarray
    private_sinr: np.ndarray
    common_bler: np.ndarray
    private_bler: np.ndarray
    success_prob: np.ndarray
    common_payload: float
    private_payload: np.ndarray
    common_power: float
    private_power: np.ndarray
    total_power: float
    constraint_violation: float
    effective_blocklength: np.ndarray
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def num_active(self) -> int:
        return int(np.count_nonzero(self.active))

    @property
    def mean_bler(self) -> float:
        """Average end-to-end block error rate over the active users."""
        if self.num_active == 0:
            return 0.0
        return float(np.mean(1.0 - self.success_prob[self.active]))


# ---------------------------------------------------------------------------
# feasibility
# ---------------------------------------------------------------------------
def action_violation(action: SlotAction, tol: float = 1e-5) -> float:
    """Return a non-negative scalar measuring constraint violation.

    Zero for actions produced by the constraint-preserving parameterisation.
    Components: negative fractions, total power > 1, non-zero allocation for
    inactive users, and a rate simplex that does not sum to one.
    """
    active = action.active
    pf = action.power_fractions
    rf = action.rate_fractions
    inactive = ~active

    violation = 0.0
    violation += float(np.sum(np.clip(-pf, 0.0, None)))
    violation += float(np.sum(np.clip(-rf, 0.0, None)))
    violation += float(max(0.0, pf.sum() - 1.0 - tol))
    violation += float(np.sum(np.abs(pf[1:][inactive])))
    violation += float(np.sum(np.abs(rf[inactive])))
    if action.rate_split_mode == "per_user_fraction":
        violation += float(np.clip(rf - 1.0 - tol, 0.0, None).sum())
    elif np.any(active):
        violation += float(max(0.0, abs(rf[active].sum() - 1.0) - tol))
    if not 0.0 - tol <= action.common_payload_scale <= 1.0 + tol:
        violation += float(abs(action.common_payload_scale - 0.5) - 0.5)
    if not 0.0 - tol <= action.power_scale <= 1.0 + tol:
        violation += float(abs(action.power_scale - 0.5) - 0.5)
    return max(violation, 0.0)


# ---------------------------------------------------------------------------
# payload split
# ---------------------------------------------------------------------------
def split_payload_batch(
    payload: np.ndarray,
    active: np.ndarray,
    rate_fractions: np.ndarray,
    common_payload_scale: np.ndarray,
    mode: str = "simplex_scaled",
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised :func:`split_payload` over a batch of allocations.

    Args:
        payload: ``(K,)`` per-user message sizes ``D_k``.
        active: ``(K,)`` boolean mask (shared by the whole batch).
        rate_fractions: ``(N, K)`` simplex rows (or per-user fractions).
        common_payload_scale: ``(N,)`` values of ``rho_r``.
        mode: ``'simplex_scaled'`` or ``'per_user_fraction'``.

    Returns:
        ``(D_c_tot, D_p)`` of shapes ``(N,)`` and ``(N, K)``.
    """
    payload = np.asarray(payload, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    c = np.atleast_2d(np.asarray(rate_fractions, dtype=np.float64))
    rho = np.atleast_1d(np.asarray(common_payload_scale, dtype=np.float64))
    n = c.shape[0]
    if not np.any(active):
        return np.zeros(n), np.zeros((n, payload.shape[0]))

    if mode == "simplex_scaled":
        rho = np.clip(rho, 0.0, 1.0)
        usable = active[None, :] & (c > _EPS)
        ratio = np.where(usable, payload[None, :] / np.where(usable, c, 1.0), np.inf)
        cap = ratio.min(axis=1)
        cap = np.where(np.isfinite(cap), cap, 0.0)
        d_c_tot = rho * cap
        d_c = np.where(active[None, :], c * d_c_tot[:, None], 0.0)
    elif mode == "per_user_fraction":
        phi = np.clip(c, 0.0, 1.0)
        d_c = np.where(active[None, :], phi * payload[None, :], 0.0)
        d_c_tot = d_c.sum(axis=1)
    else:
        raise ValueError(f"unknown rate split mode: {mode!r}")

    d_c = np.minimum(d_c, np.where(active, payload, 0.0)[None, :])
    d_p = np.clip(np.where(active[None, :], payload[None, :] - d_c, 0.0), 0.0, None)
    if mode == "per_user_fraction":
        d_c_tot = d_c.sum(axis=1)
    return d_c_tot, d_p


def split_payload(
    payload: np.ndarray,
    active: np.ndarray,
    rate_fractions: np.ndarray,
    common_payload_scale: float,
    mode: str = "simplex_scaled",
) -> tuple[float, np.ndarray]:
    """Split per-user payloads into a common part and private parts.

    Args:
        payload: ``(K,)`` per-user message sizes ``D_k`` in bits.
        active: ``(K,)`` boolean mask.
        rate_fractions: ``(K,)`` simplex over active users (``c_k``), or the
            per-user common fractions ``phi_k`` when ``mode='per_user_fraction'``.
        common_payload_scale: ``rho_r in [0, 1]`` (ignored in per-user mode).
        mode: ``'simplex_scaled'`` or ``'per_user_fraction'``.

    Returns:
        ``(D_c_tot, D_p)`` where ``D_c_tot`` is the total common-stream payload
        in bits and ``D_p`` is the ``(K,)`` vector of private payloads.
        Inactive users get ``D_p = 0`` and contribute nothing to ``D_c_tot``.
    """
    d_c_tot, d_p = split_payload_batch(
        payload,
        active,
        np.asarray(rate_fractions, dtype=np.float64)[None, :],
        np.asarray([common_payload_scale], dtype=np.float64),
        mode,
    )
    return float(d_c_tot[0]), d_p[0]


# ---------------------------------------------------------------------------
# physical layer
# ---------------------------------------------------------------------------
class RSMAPhysicalLayer:
    """Maps a :class:`SlotAction` and a channel realisation to a :class:`SlotResult`.

    The class is deterministic: ACK sampling (if any) happens in the environment.
    """

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self.num_users = cfg.num_users
        self.num_antennas = cfg.num_antennas
        self.noise_power = float(cfg.noise_power)
        self.total_power = float(cfg.total_power)
        self.blocklength = int(cfg.blocklength)
        self.payload = np.asarray(cfg.payload_vector, dtype=np.float64)
        self.mode = cfg.multiple_access

    # -- helpers ---------------------------------------------------------
    def _channel_gains(self, h: np.ndarray, directions: np.ndarray) -> np.ndarray:
        """``|h_k^H u_j|^2`` for all ``(k, j)``; shape ``(K, K)`` or ``(K,)``."""
        proj = np.conj(h) @ directions.T if directions.ndim == 2 else np.conj(h) @ directions
        return np.abs(proj) ** 2

    def _bler(self, gamma: np.ndarray, payload: np.ndarray, blocklength) -> np.ndarray:
        return link_bler(
            self.cfg.link_model,
            gamma,
            blocklength,
            payload,
            fixed_success_prob=self.cfg.fixed_success_prob,
        )

    def _fixed_link_override(
        self, active: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """End-to-end constant-success link used by the M0 debug configuration.

        ``fixed_success_prob`` is interpreted as the *per-user end-to-end*
        success probability, so the rate split has no effect: M0 is an
        AoI/scheduling-only testbed.  The whole error is attributed to the
        private stream for reporting purposes.

        Returns:
            ``(common_bler, private_bler, success_prob)``, each ``(K,)``.
        """
        k = self.num_users
        prob = float(np.clip(self.cfg.fixed_success_prob, 0.0, 1.0))
        common = np.zeros(k, dtype=np.float64)
        private = np.where(active, 1.0 - prob, 0.0)
        success = np.where(active, prob, 0.0)
        return common, private, success

    def _empty_result(self, action: SlotAction) -> SlotResult:
        k = self.num_users
        zeros = np.zeros(k, dtype=np.float64)
        return SlotResult(
            active=np.zeros(k, dtype=bool),
            common_sinr=zeros.copy(),
            private_sinr=zeros.copy(),
            common_bler=zeros.copy(),
            private_bler=zeros.copy(),
            success_prob=zeros.copy(),
            common_payload=0.0,
            private_payload=zeros.copy(),
            common_power=0.0,
            private_power=zeros.copy(),
            total_power=0.0,
            constraint_violation=action_violation(action, self.cfg.feasibility_tol),
            effective_blocklength=np.zeros(k, dtype=np.float64),
        )

    # -- main entry point -------------------------------------------------
    def transmit(self, h_true: np.ndarray, h_hat: np.ndarray, action: SlotAction) -> SlotResult:
        """Evaluate one slot.

        Args:
            h_true: ``(K, Nt)`` true channel used for SINR evaluation.
            h_hat: ``(K, Nt)`` estimated channel used for beam design.
            action: feasible slot action.

        Returns:
            :class:`SlotResult` with per-user SINRs, BLERs and success
            probabilities.  Inactive users have zeros everywhere.
        """
        if h_true.shape != (self.num_users, self.num_antennas):
            raise ValueError(f"h_true must have shape {(self.num_users, self.num_antennas)}")
        if h_hat.shape != h_true.shape:
            raise ValueError("h_hat and h_true must have the same shape")
        if action.num_users != self.num_users:
            raise ValueError("action does not match the configured number of users")

        if not np.any(action.active):
            return self._empty_result(action)

        if self.mode == "rsma":
            return self._transmit_rsma(h_true, h_hat, action, use_common=True)
        if self.mode == "sdma":
            return self._transmit_rsma(h_true, h_hat, action, use_common=False)
        if self.mode == "oma":
            return self._transmit_oma(h_true, h_hat, action)
        if self.mode == "noma":
            return self._transmit_noma(h_true, h_hat, action)
        raise ValueError(f"unknown multiple access scheme: {self.mode!r}")

    # -- vectorised RSMA/SDMA core ----------------------------------------
    def default_use_common(self) -> bool:
        """Whether this scheme transmits a common stream."""
        return self.mode == "rsma"

    def rsma_beams(
        self, h_hat: np.ndarray, active: np.ndarray, use_common: bool | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Unit-norm beam directions for a given schedule.

        The directions depend only on ``(h_hat, active)``, never on the power or
        rate allocation, so they can be computed once and reused across a whole
        batch of candidate allocations.  OMA serves one user at a time, so its
        private beams are MRT rather than the multi-user design.

        Args:
            use_common: ``None`` derives it from the configured scheme.

        Returns:
            ``(u_priv, u_common)`` of shapes ``(K, Nt)`` and ``(Nt,)``.
        """
        cfg = self.cfg
        if use_common is None:
            use_common = self.default_use_common()
        if self.mode == "oma":
            return (
                bf.mrt_directions(h_hat, active),
                np.zeros(self.num_antennas, dtype=h_hat.dtype),
            )
        u_priv = bf.private_directions(
            h_hat, active, cfg.private_beam, self.noise_power,
            self.total_power, cfg.rzf_regularization,
        )
        u_common = (
            bf.common_direction(
                h_hat,
                active,
                cfg.common_beam,
                maxmin_iterations=cfg.common_beam_iters,
                maxmin_damping=cfg.common_beam_damping,
            )
            if use_common
            else np.zeros(self.num_antennas, dtype=h_hat.dtype)
        )
        return u_priv, u_common

    def _evaluate_oma_batch(
        self,
        h_true: np.ndarray,
        active: np.ndarray,
        power_fractions: np.ndarray,
        power_scale: np.ndarray,
        beams: tuple[np.ndarray, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Vectorised OMA evaluation matching :meth:`_transmit_oma`."""
        k = self.num_users
        n_batch = power_fractions.shape[0]
        scale = power_scale * self.total_power
        u_priv = beams[0]

        frac = np.clip(power_fractions[:, 1:], 0.0, None) * active[None, :]
        totals = frac.sum(axis=1)
        fallback = np.broadcast_to(active.astype(np.float64), frac.shape)
        frac = np.where(totals[:, None] > _EPS, frac, fallback)
        tau = frac / frac.sum(axis=1, keepdims=True)
        n_k = np.where(
            active[None, :], np.maximum(np.round(tau * self.blocklength), 1.0), 0.0
        )

        gain = np.abs(np.sum(np.conj(h_true) * u_priv, axis=1)) ** 2      # (K,)
        gamma_p = np.where(
            active[None, :], scale[:, None] * gain[None, :] / self.noise_power, 0.0
        )
        d_p = np.broadcast_to(np.where(active, self.payload, 0.0), (n_batch, k)).copy()
        safe_n = np.where(n_k > 0.0, n_k, 1.0)
        bler_p = np.where(active[None, :], self._bler(gamma_p, d_p, safe_n), 0.0)
        success = np.where(active[None, :], 1.0 - bler_p, 0.0)
        if self.cfg.link_model == "fixed":
            _, over_p, over_s = self._fixed_link_override(active)
            bler_p = np.broadcast_to(over_p, (n_batch, k)).copy()
            success = np.broadcast_to(over_s, (n_batch, k)).copy()

        power_private = np.where(active[None, :], scale[:, None] * tau, 0.0)
        zeros = np.zeros((n_batch, k))
        return {
            "common_sinr": zeros.copy(),
            "private_sinr": gamma_p,
            "common_bler": zeros.copy(),
            "private_bler": bler_p,
            "success_prob": np.clip(success, 0.0, 1.0),
            "common_payload": np.zeros(n_batch),
            "private_payload": d_p,
            "common_power": np.zeros(n_batch),
            "private_power": power_private,
            "total_power": power_private.sum(axis=1),
            "degenerate": np.zeros(n_batch, dtype=bool),
            "effective_blocklength": n_k,
            "time_fractions": tau,
        }

    def evaluate_allocations(
        self,
        h_true: np.ndarray,
        active: np.ndarray,
        power_fractions: np.ndarray,
        rate_fractions: np.ndarray,
        common_payload_scale: np.ndarray,
        power_scale: np.ndarray | None = None,
        use_common: bool | None = None,
        beams: tuple[np.ndarray, np.ndarray] | None = None,
        h_hat: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Evaluate ``N`` candidate allocations for one channel and schedule.

        This is the single implementation of the RSMA/SDMA link: :meth:`transmit`
        calls it with ``N = 1``, so a batch evaluation can never drift away from
        what the environment actually simulates.

        Args:
            h_true: ``(K, Nt)`` true channel.
            active: ``(K,)`` boolean schedule.
            power_fractions: ``(N, K+1)`` ``[common, private_1..K]`` rows.
            rate_fractions: ``(N, K)`` common-rate rows.
            common_payload_scale: ``(N,)`` values of ``rho_r``.
            power_scale: ``(N,)`` values of ``rho_P`` (default: all ones).
            use_common: ``None`` (the default) derives the behaviour from the
                configured ``multiple_access`` scheme, so an SDMA or OMA physical
                layer is evaluated as SDMA or OMA rather than silently as RSMA.
                Pass an explicit bool only to force the RSMA/SDMA variant.
            beams: optional precomputed ``(u_priv, u_common)`` from :meth:`rsma_beams`.
            h_hat: needed only when ``beams`` is not given.

        Returns:
            Dict of arrays: ``common_sinr``/``private_sinr``/``common_bler``/
            ``private_bler``/``success_prob``/``private_payload`` of shape
            ``(N, K)``, and ``common_payload``/``common_power``/``total_power``
            of shape ``(N,)``.  ``private_power`` has shape ``(N, K)``.
        """
        cfg = self.cfg
        k = self.num_users
        if use_common is None:
            if self.mode == "noma":
                raise NotImplementedError(
                    "batch evaluation is not implemented for NOMA; use transmit()"
                )
            use_common = self.default_use_common()
        active = np.asarray(active, dtype=bool)
        pf = np.atleast_2d(np.asarray(power_fractions, dtype=np.float64)).copy()
        rf = np.atleast_2d(np.asarray(rate_fractions, dtype=np.float64))
        rho_r = np.atleast_1d(np.asarray(common_payload_scale, dtype=np.float64))
        n_batch = pf.shape[0]
        rho_p = (
            np.ones(n_batch)
            if power_scale is None
            else np.clip(np.atleast_1d(np.asarray(power_scale, dtype=np.float64)), 0.0, 1.0)
        )
        scale = rho_p * self.total_power

        pf = np.clip(pf, 0.0, None)
        pf[:, 1:][:, ~active] = 0.0
        if not use_common:
            pf[:, 0] = 0.0
        total_frac = pf.sum(axis=1)
        degenerate = total_frac <= _EPS
        if not use_common:
            safe = np.where(degenerate, 1.0, total_frac)
            pf = pf / safe[:, None]  # SDMA re-uses the whole budget on private streams
        power_common = pf[:, 0] * scale                      # (N,)
        power_private = pf[:, 1:] * scale[:, None]           # (N, K)

        if beams is None:
            if h_hat is None:
                raise ValueError("either beams or h_hat must be provided")
            beams = self.rsma_beams(h_hat, active, use_common)
        if self.mode == "oma":
            return self._evaluate_oma_batch(h_true, active, pf, rho_p, beams)
        u_priv, u_common = beams

        gain_priv = self._channel_gains(h_true, u_priv)       # (K, K)
        gain_common = (
            self._channel_gains(h_true, u_common) if use_common else np.zeros(k)
        )                                                    # (K,)

        interference_all = power_private @ gain_priv.T        # (N, K)
        own = power_private * np.diag(gain_priv)[None, :]     # (N, K)
        interference_other = np.clip(interference_all - own, 0.0, None)

        gamma_c = np.zeros((n_batch, k))
        if use_common:
            gamma_c = (
                power_common[:, None] * gain_common[None, :]
                / (interference_all + self.noise_power)
            )
        gamma_p = own / (interference_other + self.noise_power)
        gamma_c = np.where(active[None, :], gamma_c, 0.0)
        gamma_p = np.where(active[None, :], gamma_p, 0.0)

        if use_common:
            d_c_tot, d_p = split_payload_batch(
                self.payload, active, rf, rho_r, cfg.rate_split_mode
            )
        else:
            d_c_tot = np.zeros(n_batch)
            d_p = np.broadcast_to(np.where(active, self.payload, 0.0), (n_batch, k)).copy()

        n = self.blocklength
        bler_c = np.zeros((n_batch, k))
        has_common = d_c_tot > 0.0
        if np.any(has_common):
            common_payload = np.broadcast_to(d_c_tot[:, None], (n_batch, k))
            bler_c = np.where(
                active[None, :] & has_common[:, None],
                self._bler(gamma_c, common_payload, n),
                0.0,
            )
        bler_p = np.where(active[None, :], self._bler(gamma_p, d_p, n), 0.0)
        success = np.where(active[None, :], (1.0 - bler_c) * (1.0 - bler_p), 0.0)
        if cfg.link_model == "fixed":
            c_over, p_over, s_over = self._fixed_link_override(active)
            bler_c = np.broadcast_to(c_over, (n_batch, k)).copy()
            bler_p = np.broadcast_to(p_over, (n_batch, k)).copy()
            success = np.broadcast_to(s_over, (n_batch, k)).copy()

        if np.any(degenerate):  # nothing radiated -> nothing decoded
            success[degenerate] = 0.0
            power_common = np.where(degenerate, 0.0, power_common)
            power_private[degenerate] = 0.0

        return {
            "common_sinr": gamma_c,
            "private_sinr": gamma_p,
            "common_bler": bler_c,
            "private_bler": bler_p,
            "success_prob": np.clip(success, 0.0, 1.0),
            "common_payload": d_c_tot,
            "private_payload": d_p,
            "common_power": power_common,
            "private_power": power_private,
            "total_power": power_common + power_private.sum(axis=1),
            "degenerate": degenerate,
        }

    # -- RSMA / SDMA ------------------------------------------------------
    def _transmit_rsma(
        self, h_true: np.ndarray, h_hat: np.ndarray, action: SlotAction, use_common: bool
    ) -> SlotResult:
        """Single-slot RSMA (or SDMA) evaluation via :meth:`evaluate_allocations`."""
        cfg = self.cfg
        active = action.active
        out = self.evaluate_allocations(
            h_true,
            active,
            action.power_fractions[None, :],
            action.rate_fractions[None, :],
            np.asarray([action.common_payload_scale]),
            np.asarray([action.power_scale]),
            use_common=use_common,
            h_hat=h_hat,
        )
        if bool(out["degenerate"][0]):  # degenerate action: nothing radiated
            return self._empty_result(action)

        return SlotResult(
            active=active.copy(),
            common_sinr=out["common_sinr"][0],
            private_sinr=out["private_sinr"][0],
            common_bler=out["common_bler"][0],
            private_bler=out["private_bler"][0],
            success_prob=out["success_prob"][0],
            common_payload=float(out["common_payload"][0]),
            private_payload=out["private_payload"][0],
            common_power=float(out["common_power"][0]),
            private_power=out["private_power"][0],
            total_power=float(out["total_power"][0]),
            constraint_violation=action_violation(action, cfg.feasibility_tol),
            effective_blocklength=np.where(active, float(self.blocklength), 0.0),
        )

    # -- OMA --------------------------------------------------------------
    def _transmit_oma(
        self, h_true: np.ndarray, h_hat: np.ndarray, action: SlotAction
    ) -> SlotResult:
        """Orthogonal (time-division) access.

        The private power fractions are re-interpreted as *time* fractions
        ``tau_k`` over the active users; user ``k`` transmits alone during
        ``n_k = round(tau_k * n)`` channel uses with MRT at instantaneous power
        ``rho_P * P_max`` (equal total energy to the RSMA schemes).
        """
        cfg = self.cfg
        active = action.active
        k = self.num_users
        scale = float(np.clip(action.power_scale, 0.0, 1.0)) * self.total_power

        frac = np.clip(action.power_fractions[1:], 0.0, None) * active
        if frac.sum() <= _EPS:
            frac = active.astype(np.float64)
        tau = frac / frac.sum()
        n_k = np.where(active, np.maximum(np.round(tau * self.blocklength), 1.0), 0.0)

        u_priv = bf.mrt_directions(h_hat, active)
        gain = np.abs(np.sum(np.conj(h_true) * u_priv, axis=1)) ** 2
        gamma_p = np.where(active, scale * gain / self.noise_power, 0.0)

        d_p = np.where(active, self.payload, 0.0)
        safe_n = np.where(n_k > 0.0, n_k, 1.0)
        bler_p = np.where(active, self._bler(gamma_p, d_p, safe_n), 0.0)
        success = np.where(active, 1.0 - bler_p, 0.0)
        if cfg.link_model == "fixed":
            _, bler_p, success = self._fixed_link_override(active)

        power_private = np.where(active, scale * tau, 0.0)  # time-averaged power
        return SlotResult(
            active=active.copy(),
            common_sinr=np.zeros(k),
            private_sinr=gamma_p,
            common_bler=np.zeros(k),
            private_bler=bler_p,
            success_prob=np.clip(success, 0.0, 1.0),
            common_payload=0.0,
            private_payload=d_p,
            common_power=0.0,
            private_power=power_private,
            total_power=float(power_private.sum()),
            constraint_violation=action_violation(action, cfg.feasibility_tol),
            effective_blocklength=n_k,
            extras={"time_fractions": tau},
        )

    # -- NOMA -------------------------------------------------------------
    def _transmit_noma(
        self, h_true: np.ndarray, h_hat: np.ndarray, action: SlotAction
    ) -> SlotResult:
        """Single-beam SC-SIC NOMA baseline.

        All active users are superposed on one beam (the configured common
        direction).  With effective gains ``g_k = |h_k^H u|^2`` sorted
        ascending, user ``i`` decodes messages ``1..i`` sequentially, treating
        later ones as noise, and succeeds only if the whole SIC chain succeeds.
        """
        cfg = self.cfg
        active = action.active
        k = self.num_users
        scale = float(np.clip(action.power_scale, 0.0, 1.0)) * self.total_power

        # NOMA has no common stream: its power fraction is redistributed over
        # the layers by renormalising the private fractions to sum to one.
        pf = np.clip(action.power_fractions[1:], 0.0, None) * active
        if pf.sum() <= _EPS:
            pf = active.astype(np.float64)
        power = pf / pf.sum() * scale

        u = bf.common_direction(
            h_hat,
            active,
            cfg.common_beam,
            maxmin_iterations=cfg.common_beam_iters,
            maxmin_damping=cfg.common_beam_damping,
        )
        gain = np.abs(np.conj(h_true) @ u) ** 2
        idx = np.flatnonzero(active)
        order = idx[np.argsort(gain[idx], kind="stable")]  # ascending gain

        d_p = np.where(active, self.payload, 0.0)
        n = self.blocklength
        success = np.zeros(k)
        gamma_own = np.zeros(k)
        bler_own = np.zeros(k)
        for pos, user in enumerate(order):
            prob = 1.0
            for j in range(pos + 1):
                target = order[j]
                later = order[j + 1:]
                inter = gain[user] * float(power[later].sum())
                gamma = gain[user] * power[target] / (inter + self.noise_power)
                bler = float(self._bler(np.asarray([gamma]), np.asarray([d_p[target]]), n)[0])
                prob *= 1.0 - bler
                if target == user:
                    gamma_own[user] = gamma
                    bler_own[user] = bler
            success[user] = prob

        bler_noma = np.clip(1.0 - success, BLER_MIN, BLER_MAX) * active
        if cfg.link_model == "fixed":
            _, bler_noma, success = self._fixed_link_override(active)
            bler_own = bler_noma.copy()

        return SlotResult(
            active=active.copy(),
            common_sinr=np.zeros(k),
            private_sinr=gamma_own,
            common_bler=np.zeros(k),
            private_bler=bler_noma,
            success_prob=np.clip(success, 0.0, 1.0),
            common_payload=0.0,
            private_payload=d_p,
            common_power=0.0,
            private_power=power,
            total_power=float(power.sum()),
            constraint_violation=action_violation(action, cfg.feasibility_tol),
            effective_blocklength=np.where(active, float(n), 0.0),
            extras={"noma_order": order.astype(np.float64), "sic_bler": bler_own},
        )
