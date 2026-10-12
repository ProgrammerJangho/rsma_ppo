"""Gymnasium environment: AoI minimisation over an FBL 1-layer RSMA downlink.

Slot timeline
-------------
1. The agent observes ``Delta(t)`` and the CSIT ``H_hat(t)``.
2. It picks a schedule, a power split and a rate split (already feasible).
3. The PHY evaluates SINRs with the *true* ``H(t)`` and returns per-user
   success probabilities.
4. ACK/NACK is drawn (``ack_mode='sampled'``) or the expectation is used
   (``ack_mode='expected'``).
5. AoI is updated, the reward is formed from ``Delta(t+1)``.
6. The channel evolves to ``H(t+1)`` and the next observation is built.

AoI recursion (generate-at-will)::

    Delta_k(t+1) = 1                                if user k succeeded
    Delta_k(t+1) = min(Delta_k(t) + 1, Delta_max)   otherwise

Set aoi_cap_enabled=False to use Delta_k(t)+1 without a physical age cap.
Observation scaling/clipping is separate from the physical age and reward.

With ``ack_mode='expected'`` the *expected* transition is used instead, which
yields fractional AoI; it is meant for debugging / low-variance sanity checks,
not for reporting final results.

ASSUMPTION: generate-at-will status updates (a fresh update is always available
at the transmitter).  Stochastic arrivals / queues are not modelled in M0-M2;
the hook ``_generate_arrivals`` marks where they would enter.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rsma_aoi.envs.channel import ChannelModel
from rsma_aoi.envs.fbl import fbl_max_payload
from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer, SlotAction, SlotResult, action_violation
from rsma_aoi.utils.config import EnvConfig

_OBS_CLIP = 10.0
_EPS = 1e-12


class AoIRSMAEnv(gym.Env):
    """Single-cell MU-MISO downlink AoI environment.

    Action space (``gymnasium.spaces.Dict``):
        ``schedule``: ``MultiBinary(K)`` - user activity mask.
        Canonical ``per_user_fraction`` mode:
        ``common_power_fraction``: scalar ``alpha``.
        ``private_power_fractions``: private-power simplex ``q_k``.
        ``common_payload_fractions``: independent ``phi_k`` values with
            ``D_c,k=phi_k D_k`` and ``C_k=D_c,k/n``.

        Legacy ``simplex_scaled`` mode:
        ``power_fractions`` and ``rate_fractions`` contain the old power/rate
            simplexes.
        ``common_payload_scale``: legacy-only ``rho_r`` key, absent from the
            canonical ``per_user_fraction`` action.
        ``power_scale``: ``Box(0, 1, (1,))`` - only when
            ``cfg.power_scale_action`` is True.

    Observation space: flat ``Box`` in ``[-10, 10]`` (float32), see
    :meth:`_build_observation` for the layout.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, cfg: EnvConfig, render_mode: str | None = None) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.render_mode = render_mode

        self.num_users = cfg.num_users
        self.num_antennas = cfg.num_antennas
        self.aoi_max = float(cfg.aoi_max)
        self.weights = np.asarray(cfg.weight_vector, dtype=np.float64)
        self.payload = np.asarray(cfg.payload_vector, dtype=np.float64)
        self.total_power = cfg.total_power
        self._current_snr_db = float(cfg.snr_db) if cfg.snr_db is not None else None
        self._current_blocklength = int(cfg.blocklength)
        self.reward_scale = cfg.effective_reward_scale

        self.channel = ChannelModel.from_config(cfg)
        self.phy = RSMAPhysicalLayer(cfg)
        self._allocator = self._build_allocator()
        self._allocator_calls = 0
        self._allocator_evaluations = 0
        self._common_payload_action = 0.0
        self._common_payload_effective = 0.0
        self._channel_scale = 1.0 / float(np.sqrt(np.mean(cfg.path_gain_linear)))

        self.action_space = self._make_action_space()
        self.observation_space = spaces.Box(
            low=-_OBS_CLIP, high=_OBS_CLIP, shape=(self._observation_dim(),), dtype=np.float32
        )

        # episode state -----------------------------------------------------
        self._aoi = np.ones(self.num_users, dtype=np.float64)
        self._h_true = np.zeros((self.num_users, self.num_antennas), dtype=np.complex128)
        self._h_hat = np.zeros_like(self._h_true)
        self._prev_ack = np.zeros(self.num_users, dtype=np.float64)
        self._prev_power_fractions = np.zeros(self.num_users + 1, dtype=np.float64)
        self._prev_schedule = np.zeros(self.num_users, dtype=np.float64)
        self._step_count = 0
        self._episode_return = 0.0
        self._episode_weighted_aoi = 0.0
        self._episode_success = 0.0
        self._episode_attempts = 0.0
        self._episode_power = 0.0
        self._episode_violation = 0.0
        self._last_info: dict[str, Any] = {}
        self._episode_geometry_index: int | None = None
        self._episode_snr_index: int | None = None
        self._episode_blocklength_index: int | None = None

    def _build_allocator(self):
        """Low-level allocator for the hierarchical action space (or None)."""
        mode = self.cfg.allocator_mode
        if mode == "none":
            return None
        from rsma_aoi.optimizers import CachedAllocator, build_allocator

        if mode == "cached":
            return CachedAllocator(build_allocator("cem_fast"), tol=self.cfg.allocator_cache_tol)
        return build_allocator(mode)

    @property
    def is_hierarchical(self) -> bool:
        return self._allocator is not None

    def _solve_allocation(
        self, active: np.ndarray, rho_r: float
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        """Replace the agent's allocation with the optimiser's for this subset.

        The beams and the objective use the *estimated* channel, exactly like a
        real transmitter; only the environment's own evaluation later uses the
        true channel.

        Returns:
            ``(power_fractions, rate_fractions, rho_r, evaluations)``.
        """
        from rsma_aoi.optimizers import AllocationProblem
        from rsma_aoi.evaluation.oracle import urgency_vector

        urgency, _ = urgency_vector(
            self._aoi, self.weights, self.aoi_max if self.cfg.aoi_cap_enabled else np.inf
        )
        use_common = True
        if self.cfg.allocator_common_gate:
            use_common = rho_r > 0.5
        problem = AllocationProblem(
            phy=self.phy, active=active, urgency=urgency,
            h_score=self._h_hat, h_design=self._h_hat, h_opt=self._h_hat,
            beams=self.phy.rsma_beams(self._h_hat, active, use_common=use_common),
        )
        rng = np.random.default_rng(
            self.cfg.allocator_seed * 1_000_003 + self._step_count
        )
        result = self._allocator.solve(problem, rng)
        power = result.power_fractions.copy()
        rho = result.rho_r
        if not use_common:
            power[0] = 0.0
            total = power.sum()
            if total > 0:
                power = power / total
            rho = 0.0
        return power, result.rate_fractions, rho, result.evaluations

    # ------------------------------------------------------------------
    # spaces
    # ------------------------------------------------------------------
    def _make_action_space(self) -> spaces.Dict:
        k = self.num_users
        if self.cfg.rate_split_mode == "per_user_fraction":
            components: dict[str, spaces.Space] = {
                "schedule": spaces.MultiBinary(k),
                "common_power_fraction": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "private_power_fractions": spaces.Box(0.0, 1.0, shape=(k,), dtype=np.float32),
                "common_payload_fractions": spaces.Box(0.0, 1.0, shape=(k,), dtype=np.float32),
            }
        else:
            components = {
                "schedule": spaces.MultiBinary(k),
                "power_fractions": spaces.Box(0.0, 1.0, shape=(k + 1,), dtype=np.float32),
                "rate_fractions": spaces.Box(0.0, 1.0, shape=(k,), dtype=np.float32),
                "common_payload_scale": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            }
        if self.cfg.power_scale_action:
            components["power_scale"] = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        return spaces.Dict(components)

    def _channel_feature_dim(self) -> int:
        k, nt = self.num_users, self.num_antennas
        mode = self.cfg.obs_channel_mode
        if mode == "complex_flat":
            return 2 * k * nt
        if mode == "gain_features":
            return k + (k * (k - 1)) // 2
        if mode == "beam_sinr_features":
            base = k + (k * (k - 1)) // 2
            return base + k + k * k + 2 * k
        return 0

    def _observation_dim(self) -> int:
        k = self.num_users
        dim = k                                # normalised AoI
        dim += self._channel_feature_dim()
        dim += k                               # previous ACK
        if self.cfg.obs_include_prev_action:
            dim += (k + 1) + k                 # previous power fractions + schedule
        if self.cfg.obs_include_snr:
            dim += 1                           # current transmit SNR
        if self.cfg.obs_include_blocklength:
            dim += 1                           # current blocklength ratio
        return dim

    @property
    def observation_layout(self) -> list[tuple[str, int]]:
        """Named observation blocks and their widths (for logging/debugging)."""
        k = self.num_users
        layout = [("aoi_normalised", k)]
        if self._channel_feature_dim():
            layout.append((f"channel:{self.cfg.obs_channel_mode}", self._channel_feature_dim()))
        layout.append(("prev_ack", k))
        if self.cfg.obs_include_prev_action:
            layout.append(("prev_power_fractions", k + 1))
            layout.append(("prev_schedule", k))
        if self.cfg.obs_include_snr:
            layout.append(("snr_db_div_20", 1))
        if self.cfg.obs_include_blocklength:
            layout.append(("blocklength_ratio", 1))
        return layout

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------
    def _channel_features(self) -> np.ndarray:
        mode = self.cfg.obs_channel_mode
        if mode == "none":
            return np.zeros(0, dtype=np.float64)
        h = self._h_hat * self._channel_scale
        if mode == "complex_flat":
            return np.concatenate([h.real.ravel(), h.imag.ravel()])
        # gain_features: per-user normalised gain + pairwise channel correlation
        gains = np.sum(np.abs(h) ** 2, axis=1) / float(self.num_antennas)
        norms = np.linalg.norm(h, axis=1)
        feats = [gains]
        corr = []
        for i in range(self.num_users):
            for j in range(i + 1, self.num_users):
                denom = norms[i] * norms[j]
                corr.append(abs(np.vdot(h[i], h[j])) / denom if denom > _EPS else 0.0)
        if corr:
            feats.append(np.asarray(corr, dtype=np.float64))
        if mode == "beam_sinr_features":
            active = np.ones(self.num_users, dtype=bool)
            u_priv, u_common = self.phy.rsma_beams(self._h_hat, active, use_common=True)
            common_gain = np.abs(self._h_hat.conj() @ u_common) ** 2
            private_gain = np.abs(self._h_hat.conj() @ u_priv.T) ** 2
            snr_scale = self.total_power / self.cfg.noise_power
            feats.extend([
                np.log1p(snr_scale * common_gain),
                np.log1p(snr_scale * private_gain).ravel(),
            ])

            # The current action is not yet known.  Apply the previous power
            # allocation to the current estimated channel; on the first slot,
            # use a documented alpha=0.25/equal-private reference allocation.
            pf = self._prev_power_fractions.copy()
            if float(pf.sum()) <= _EPS:
                alpha_ref = 0.25
                pf = np.concatenate((
                    np.asarray([alpha_ref]),
                    np.full(self.num_users, (1.0 - alpha_ref) / self.num_users),
                ))
            nominal = self.phy.evaluate_allocations(
                self._h_hat, active, pf[None, :],
                np.zeros((1, self.num_users), dtype=np.float64),
                np.zeros(1, dtype=np.float64), use_common=True,
                beams=(u_priv, u_common),
            )
            feats.extend([
                np.log1p(nominal["common_sinr"][0]),
                np.log1p(nominal["private_sinr"][0]),
            ])
        return np.concatenate(feats)

    def _build_observation(self) -> np.ndarray:
        parts = [self._aoi / self.aoi_max, self._channel_features(), self._prev_ack]
        if self.cfg.obs_include_prev_action:
            parts.append(self._prev_power_fractions)
            parts.append(self._prev_schedule)
        if self.cfg.obs_include_snr:
            if self._current_snr_db is None:
                raise RuntimeError("SNR observation requires snr_db rather than p_max")
            parts.append(np.asarray([self._current_snr_db / 20.0], dtype=np.float64))
        if self.cfg.obs_include_blocklength:
            parts.append(np.asarray([
                self._current_blocklength / float(self.cfg.blocklength)
            ], dtype=np.float64))
        obs = np.concatenate(parts).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=_OBS_CLIP, neginf=-_OBS_CLIP)
        return np.clip(obs, -_OBS_CLIP, _OBS_CLIP)

    # ------------------------------------------------------------------
    # gym API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        rng = self.np_random

        if self.cfg.aoi_init == "one":
            self._aoi = np.ones(self.num_users, dtype=np.float64)
        else:  # 'random': uniform over a small initial range (documented default)
            high = int(min(self.cfg.aoi_max, 10))
            self._aoi = rng.integers(1, high + 1, size=self.num_users).astype(np.float64)

        self._episode_geometry_index = self.channel.resample_episode_geometry(rng)
        self._episode_snr_index = None
        self._episode_blocklength_index = None
        self._current_snr_db = float(self.cfg.snr_db) if self.cfg.snr_db is not None else None
        if self.cfg.snr_db_values is not None:
            self._episode_snr_index = int(rng.integers(len(self.cfg.snr_db_values)))
            self._current_snr_db = float(self.cfg.snr_db_values[self._episode_snr_index])
        self.total_power = (
            float(self.cfg.noise_power) * 10.0 ** (self._current_snr_db / 10.0)
            if self._current_snr_db is not None else float(self.cfg.total_power)
        )
        self.phy.total_power = self.total_power
        self._current_blocklength = int(self.cfg.blocklength)
        if self.cfg.blocklength_values is not None:
            self._episode_blocklength_index = int(rng.integers(len(self.cfg.blocklength_values)))
            self._current_blocklength = int(
                self.cfg.blocklength_values[self._episode_blocklength_index]
            )
        self.phy.blocklength = self._current_blocklength
        self._h_true = self.channel.sample_initial(rng)
        self._h_hat = self.channel.estimate(self._h_true, rng)
        self._prev_ack = np.zeros(self.num_users, dtype=np.float64)
        self._prev_power_fractions = np.zeros(self.num_users + 1, dtype=np.float64)
        self._prev_schedule = np.zeros(self.num_users, dtype=np.float64)
        self._step_count = 0
        self._episode_return = 0.0
        self._episode_weighted_aoi = 0.0
        self._episode_success = 0.0
        self._episode_attempts = 0.0
        self._episode_power = 0.0
        self._episode_violation = 0.0
        self._allocator_calls = 0
        self._allocator_evaluations = 0
        self._last_info = {}
        reset_info: dict[str, Any] = {}
        if self.channel.user_angles_deg is not None:
            reset_info["user_angles_deg"] = self.channel.user_angles_deg.copy()
        if self._episode_geometry_index is not None:
            reset_info["geometry_index"] = self._episode_geometry_index
        if self._current_snr_db is not None:
            reset_info["snr_db"] = self._current_snr_db
            reset_info["total_power"] = self.total_power
        if self._episode_snr_index is not None:
            reset_info["snr_index"] = self._episode_snr_index
        reset_info["blocklength"] = self._current_blocklength
        if self._episode_blocklength_index is not None:
            reset_info["blocklength_index"] = self._episode_blocklength_index
        return self._build_observation(), reset_info

    def step(
        self, action: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        slot_action, violation = self._parse_action(action)
        result = self.phy.transmit(self._h_true, self._h_hat, slot_action)

        ack = self._sample_ack(result)
        aoi_next = self._update_aoi(result, ack)
        reward, components = self._reward(aoi_next, result, violation)

        self._aoi = aoi_next
        self._prev_ack = ack.astype(np.float64)
        self._prev_power_fractions = slot_action.power_fractions.copy()
        self._prev_schedule = slot_action.active.astype(np.float64)
        self._step_count += 1

        # transition the channel *after* the transmission of slot t
        rng = self.np_random
        self._generate_arrivals(rng)
        self._h_true = self.channel.evolve(self._h_true, rng)
        self._h_hat = self.channel.estimate(self._h_true, rng)

        terminated = False
        truncated = self._step_count >= self.cfg.episode_length
        info = self._build_info(result, ack, aoi_next, reward, components, violation)
        if truncated:
            info["episode"] = self._episode_summary()
        self._last_info = info
        return self._build_observation(), float(reward), terminated, truncated, info

    def close(self) -> None:  # pragma: no cover - nothing to release
        return None

    def render(self) -> None:  # pragma: no cover - debugging aid
        if self.render_mode != "human":
            return
        aoi = np.array2string(self._aoi, precision=1)
        print(f"[t={self._step_count:4d}] AoI={aoi} sched={self._prev_schedule.astype(int)}")

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _generate_arrivals(self, rng: np.random.Generator) -> None:
        """Hook for stochastic arrivals / queueing (disabled in M0-M2).

        Generate-at-will means a fresh update is always available, so this is a
        no-op.  A queueing extension would update per-user buffers here.
        """
        return None

    def _parse_action(self, action: dict[str, np.ndarray]) -> tuple[SlotAction, float]:
        """Validate, measure violation, and minimally sanitise the raw action.

        The agent's parameterisation is feasible by construction, so the
        sanitisation below is a no-op for it.  It exists so that arbitrary
        in-space actions (Gymnasium's env checker, random baselines) still yield
        a physically valid transmission, with the deviation reported through
        ``constraint_violation``.
        """
        if not isinstance(action, dict):
            raise TypeError("action must be a dict matching the Dict action space")
        # Direct actions generated before the public-schema cleanup used a
        # combined power vector and the misleading key ``rate_fractions``.
        # Accept those keys as an input adapter for archived scripts, while all
        # new policies and the declared Gym space use alpha/q/phi explicitly.
        if self.cfg.rate_split_mode == "per_user_fraction" and not {
            "common_power_fraction", "private_power_fractions",
            "common_payload_fractions",
        }.issubset(action):
            if "power_fractions" in action and "rate_fractions" in action:
                old_pf = np.asarray(action["power_fractions"], dtype=np.float64).reshape(-1)
                old_alpha = float(old_pf[0])
                private_mass = max(0.0, 1.0 - old_alpha)
                old_q = old_pf[1:] / private_mass if private_mass > _EPS else np.zeros(self.num_users)
                action = dict(action)
                action["common_power_fraction"] = np.asarray([old_alpha])
                action["private_power_fractions"] = old_q
                action["common_payload_fractions"] = np.asarray(action["rate_fractions"])

        missing = set(self.action_space.spaces) - set(action)
        if missing:
            raise ValueError(f"action is missing keys: {sorted(missing)}")

        active = np.asarray(action["schedule"], dtype=np.int64).reshape(-1).astype(bool)
        if active.shape[0] != self.num_users:
            raise ValueError("schedule must have length num_users")
        if not self.cfg.scheduling_enabled:
            active = np.ones(self.num_users, dtype=bool)

        direct = self.cfg.rate_split_mode == "per_user_fraction"
        if direct:
            alpha_raw = float(np.asarray(action["common_power_fraction"]).reshape(-1)[0])
            q_raw = np.asarray(action["private_power_fractions"], dtype=np.float64).reshape(-1).copy()
            rf = np.asarray(action["common_payload_fractions"], dtype=np.float64).reshape(-1).copy()
            if q_raw.shape != (self.num_users,) or rf.shape != (self.num_users,):
                raise ValueError(
                    "private_power_fractions and common_payload_fractions must "
                    "both have length num_users"
                )
            # Measure violations in the public alpha/q/phi coordinates.
            violation = float(max(0.0, -alpha_raw) + max(0.0, alpha_raw - 1.0))
            violation += float(np.clip(-q_raw, 0.0, None).sum())
            violation += float(np.clip(q_raw - 1.0, 0.0, None).sum())
            violation += float(np.clip(-rf, 0.0, None).sum())
            violation += float(np.clip(rf - 1.0, 0.0, None).sum())
            violation += float(np.abs(q_raw[~active]).sum() + np.abs(rf[~active]).sum())
            if np.any(active):
                violation += float(max(0.0, abs(q_raw[active].sum() - 1.0) - self.cfg.feasibility_tol))
            alpha = float(np.clip(alpha_raw, 0.0, 1.0))
            q = np.clip(q_raw, 0.0, None)
            q[~active] = 0.0
            q_sum = float(q.sum())
            if np.any(active):
                q = q / q_sum if q_sum > _EPS else active.astype(np.float64) / active.sum()
            rf = np.clip(rf, 0.0, 1.0)
            rf[~active] = 0.0
            pf = np.concatenate(([alpha], (1.0 - alpha) * q))
            rho_r = 1.0  # ignored by the canonical PHY mapping
        else:
            pf = np.asarray(action["power_fractions"], dtype=np.float64).reshape(-1).copy()
            rf = np.asarray(action["rate_fractions"], dtype=np.float64).reshape(-1).copy()
            rho_r = float(np.asarray(action["common_payload_scale"]).reshape(-1)[0])
        rho_p = (
            float(np.asarray(action["power_scale"]).reshape(-1)[0])
            if self.cfg.power_scale_action
            else 1.0
        )

        raw = SlotAction(active, pf, rf, rho_r, rho_p, self.cfg.rate_split_mode)
        if not direct:
            violation = action_violation(raw, self.cfg.feasibility_tol)

        # -- sanitisation (identity for feasible actions) -------------------
        pf = np.clip(pf, 0.0, None)
        pf[1:][~active] = 0.0
        total = float(pf.sum())
        if total > 1.0:
            pf /= total
        rf = np.clip(rf, 0.0, None)
        rf[~active] = 0.0
        rf_sum = float(rf.sum())
        if direct:
            rf = np.clip(rf, 0.0, 1.0)
        elif np.any(active):
            rf = rf / rf_sum if rf_sum > _EPS else active.astype(np.float64) / active.sum()
        else:
            rf = np.zeros_like(rf)
        rho_r = float(np.clip(rho_r, 0.0, 1.0))
        rho_p = float(np.clip(rho_p, 0.0, 1.0))

        # In the optional FBL-normalised action, the policy's scalar is a
        # utilisation of the common payload that the *estimated* channel can
        # support at the configured worst-user BLER.  Power and per-user common
        # shares remain direct PPO decisions.  This is a transmitter-feasible
        # mapping: h_true is never consulted.
        self._common_payload_action = rho_r
        if self.cfg.common_payload_mapping == "fbl_normalized" and np.any(active):
            rho_r = self._fbl_normalized_common_payload(
                active, pf, rf, rho_r, rho_p
            )
        self._common_payload_effective = rho_r

        if self._allocator is not None and np.any(active):
            # The agent's continuous output is discarded in hierarchical mode, so
            # it must not be charged for it: the reported violation is that of
            # the action actually transmitted (the allocator guarantees ~0).
            pf, rf, rho_r, evaluations = self._solve_allocation(active, rho_r)
            self._allocator_calls += 1
            self._allocator_evaluations += evaluations
            violation = action_violation(
                SlotAction(active, pf, rf, rho_r, rho_p, self.cfg.rate_split_mode), self.cfg.feasibility_tol
            )
            self._common_payload_effective = rho_r
        return SlotAction(active, pf, rf, rho_r, rho_p, self.cfg.rate_split_mode), violation

    def _fbl_normalized_common_payload(
        self,
        active: np.ndarray,
        power_fractions: np.ndarray,
        rate_fractions: np.ndarray,
        utilisation: float,
        power_scale: float,
    ) -> float:
        """Map a unit action to an estimated-CSIT FBL-decodable payload scale.

        For the selected power and common-rate simplex, ``rho=1`` corresponds
        to the smaller of (a) the message-splitting feasibility cap and (b) the
        payload supported by every active common decoder at
        ``common_target_bler``.  Values below one use that budget linearly.

        The guarantee is conditional on ``h_hat`` being the realised channel
        and on the normal approximation.  With imperfect CSIT it is a nominal
        action mapping, not a reliability guarantee on ``h_true``.
        """
        utilisation = float(np.clip(utilisation, 0.0, 1.0))
        if utilisation <= 0.0 or power_fractions[0] <= _EPS:
            return 0.0
        usable = active & (rate_fractions > _EPS)
        if not np.any(usable):
            return 0.0
        split_cap = float(np.min(self.payload[usable] / rate_fractions[usable]))
        if split_cap <= _EPS:
            return 0.0

        beams = self.phy.rsma_beams(self._h_hat, active, use_common=True)
        nominal = self.phy.evaluate_allocations(
            self._h_hat,
            active,
            power_fractions[None, :],
            rate_fractions[None, :],
            np.asarray([0.0]),
            power_scale=np.asarray([power_scale]),
            use_common=True,
            beams=beams,
        )
        common_sinr = nominal["common_sinr"][0, active]
        fbl_cap = float(np.min(fbl_max_payload(
            common_sinr, self._current_blocklength, self.cfg.common_target_bler
        )))
        payload_budget = min(split_cap, max(0.0, fbl_cap))
        return float(utilisation * payload_budget / split_cap)

    def _sample_ack(self, result: SlotResult) -> np.ndarray:
        """Draw ACKs (Bernoulli) or return the success probabilities."""
        if self.cfg.ack_mode == "expected":
            return result.success_prob.copy()
        draws = self.np_random.random(self.num_users)
        return (draws < result.success_prob).astype(np.float64)

    def _update_aoi(self, result: SlotResult, ack: np.ndarray) -> np.ndarray:
        """Apply the generate-at-will AoI recursion (expected form if configured)."""
        aged = self._aoi + 1.0
        if self.cfg.aoi_cap_enabled:
            aged = np.minimum(aged, self.aoi_max)
        if self.cfg.ack_mode == "expected":
            p = np.clip(ack, 0.0, 1.0)
            return p * 1.0 + (1.0 - p) * aged
        return np.where(ack > 0.5, 1.0, aged)

    def _reward(
        self, aoi_next: np.ndarray, result: SlotResult, violation: float
    ) -> tuple[float, dict[str, float]]:
        cfg = self.cfg
        if cfg.reward_mode == "expected_aoi":
            aged = self._aoi + 1.0
            if cfg.aoi_cap_enabled:
                aged = np.minimum(aged, self.aoi_max)
            p = np.clip(result.success_prob, 0.0, 1.0)
            expected_next = p + (1.0 - p) * aged
            weighted = float(np.dot(self.weights, expected_next))
            # Conditional expectation of the nonlinear tail penalty, rather
            # than the penalty of E[AoI].  This keeps the reward unbiased when
            # lambda_tail is used.
            reset_tail = np.clip(1.0 - cfg.aoi_tail_threshold, 0.0, None)
            aged_tail = np.clip(aged - cfg.aoi_tail_threshold, 0.0, None)
            tail = float(np.sum(p * reset_tail + (1.0 - p) * aged_tail))
        else:
            weighted = float(np.dot(self.weights, aoi_next))
            tail = float(np.sum(np.clip(aoi_next - cfg.aoi_tail_threshold, 0.0, None)))
        power_ratio = float(result.total_power / self.total_power)
        raw = (
            -weighted
            - cfg.lambda_power * power_ratio
            - cfg.lambda_tail * tail
            - cfg.lambda_violation * violation
        )
        reward = raw * self.reward_scale
        components = {
            "reward_aoi": -weighted * self.reward_scale,
            "reward_power": -cfg.lambda_power * power_ratio * self.reward_scale,
            "reward_tail": -cfg.lambda_tail * tail * self.reward_scale,
            "reward_violation": -cfg.lambda_violation * violation * self.reward_scale,
            "reward_raw": raw,
            "reward_expected_aoi": float(cfg.reward_mode == "expected_aoi"),
        }
        return reward, components

    def _build_info(
        self,
        result: SlotResult,
        ack: np.ndarray,
        aoi_next: np.ndarray,
        reward: float,
        components: dict[str, float],
        violation: float,
    ) -> dict[str, Any]:
        weighted = float(np.dot(self.weights, aoi_next))
        tail = float(np.sum(np.clip(aoi_next - self.cfg.aoi_tail_threshold, 0.0, None)))
        num_active = result.num_active
        attempts = float(num_active)
        self._episode_return += reward
        self._episode_weighted_aoi += weighted
        self._episode_success += float(np.sum(ack)) if self.cfg.ack_mode == "sampled" else float(
            np.sum(result.success_prob)
        )
        self._episode_attempts += attempts
        self._episode_power += float(result.total_power)
        self._episode_violation += float(violation > self.cfg.feasibility_tol)

        info: dict[str, Any] = {
            "weighted_aoi": weighted,
            "mean_aoi": float(np.mean(aoi_next)),
            "max_aoi": float(np.max(aoi_next)),
            "aoi": aoi_next.copy(),
            "tail_excess": tail,
            "total_power": float(result.total_power),
            "common_power": float(result.common_power),
            "private_power": result.private_power.copy(),
            "common_sinr": result.common_sinr.copy(),
            "private_sinr": result.private_sinr.copy(),
            "common_bler": result.common_bler.copy(),
            "private_bler": result.private_bler.copy(),
            "success_probabilities": result.success_prob.copy(),
            "ack": ack.copy(),
            "scheduled_users": result.active.astype(np.int8),
            "num_scheduled": num_active,
            "constraint_violation": float(violation),
            "allocator_calls": float(self._allocator_calls),
            "allocator_evaluations": float(self._allocator_evaluations),
            "common_payload": float(result.common_payload),
            "common_payload_action": float(self._common_payload_action),
            "common_payload_effective_scale": float(self._common_payload_effective),
            "private_payload": result.private_payload.copy(),
            "mean_bler": result.mean_bler,
            "reward": float(reward),
        }
        if self.cfg.rate_split_mode == "per_user_fraction":
            info.pop("common_payload_action", None)
            info.pop("common_payload_effective_scale", None)
        common_bits = np.where(result.active, self.payload-result.private_payload, 0.0)
        info["common_bits_per_user"] = common_bits
        info["common_fraction_per_user"] = common_bits / self.payload
        info["common_rate_per_user"] = common_bits / self._current_blocklength
        info["blocklength"] = self._current_blocklength
        info.update(components)
        return info

    def _episode_summary(self) -> dict[str, float]:
        steps = max(self._step_count, 1)
        return {
            "r": float(self._episode_return),
            "l": int(self._step_count),
            "weighted_aoi": float(self._episode_weighted_aoi / steps),
            "success_rate": float(
                self._episode_success / self._episode_attempts
                if self._episode_attempts > 0
                else 0.0
            ),
            "mean_power": float(self._episode_power / steps),
            "violation_rate": float(self._episode_violation / steps),
        }

    # ------------------------------------------------------------------
    # accessors used by baselines / evaluation
    # ------------------------------------------------------------------
    def get_state(self) -> dict[str, Any]:
        """Structured slot state for non-learning policies.

        Baselines see exactly what the transmitter sees: AoI, the CSIT estimate
        ``H_hat`` (never the true channel), the AoI weights and the payloads.
        """
        return {
            "aoi": self._aoi.copy(),
            "h_hat": self._h_hat.copy(),
            "weights": self.weights.copy(),
            "payload": self.payload.copy(),
            "num_users": self.num_users,
            "num_antennas": self.num_antennas,
            "step": self._step_count,
            "snr_db": self._current_snr_db,
            "noise_power": float(self.cfg.noise_power),
            "total_power": float(self.total_power),
            "blocklength": int(self._current_blocklength),
        }

    @property
    def aoi(self) -> np.ndarray:
        return self._aoi.copy()

    @property
    def true_channel(self) -> np.ndarray:
        """The realised channel ``H(t)``, for **offline analysis only**.

        Policies must never see this - they observe ``h_hat`` through
        :meth:`get_state` / the observation.  It is exposed so that genie-aided
        diagnostics (e.g. ``scripts/diagnose_scheduling.py``) can compute the
        success probabilities the environment itself would produce.
        """
        return self._h_true.copy()

    @property
    def last_info(self) -> dict[str, Any]:
        return self._last_info
