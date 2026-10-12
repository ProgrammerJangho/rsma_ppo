"""Per-episode AoI / PHY metrics.

Definitions
-----------
* **average AoI**: time average of ``Delta_k(t)`` after the update of slot ``t``.
* **weighted average AoI**: ``(1/T) sum_t sum_k w_k Delta_k(t)`` - the objective.
* **peak AoI**: mean over the AoI *peaks*, i.e. the values reached immediately
  before a successful reset (the standard peak-AoI definition).  Users that
  never reset contribute their episode maximum.
* **tail AoI**: 95th / 99th percentiles of the pooled per-user AoI samples plus
  the mean excess over ``aoi_tail_threshold``.
* **Jain fairness**: computed on the per-user *mean AoI*; 1.0 means all users
  experience the same average age.
* **schedule entropy**: Shannon entropy (nats) of the empirical distribution over
  the *scheduled subsets* within an episode.  0 means the policy always picks the
  same subset; ``log(2^K - 1)`` means it spreads uniformly over all subsets.  It
  is a behavioural statistic of the realised schedule, not the policy's own
  action-distribution entropy.
* **schedule count histogram**: fraction of slots in which exactly ``c`` users
  were scheduled, for ``c = 0..K``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


def jain_fairness(values: np.ndarray) -> float:
    """Jain's fairness index ``(sum x)^2 / (n sum x^2)`` in ``(0, 1]``."""
    x = np.asarray(values, dtype=np.float64).ravel()
    denom = x.shape[0] * float(np.sum(np.square(x)))
    if denom <= 0.0:
        return 1.0
    return float(np.square(np.sum(x)) / denom)


def schedule_statistics(masks: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Summarise a ``(T, K)`` sequence of scheduling masks.

    Returns:
        ``(count_hist, entropy_nats, activation_rate)`` where ``count_hist`` has
        length ``K+1`` and sums to 1, ``entropy_nats`` is the Shannon entropy of
        the empirical subset distribution, and ``activation_rate[k]`` is the
        fraction of slots in which user ``k`` was scheduled.
    """
    masks = np.asarray(masks, dtype=bool)
    steps, users = masks.shape
    counts = masks.sum(axis=1)
    hist = np.bincount(counts, minlength=users + 1)[: users + 1] / max(steps, 1)

    codes = masks @ (1 << np.arange(users))
    _, occurrences = np.unique(codes, return_counts=True)
    probs = occurrences / max(steps, 1)
    entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, None))))
    return hist, entropy, masks.mean(axis=0)


def peak_aoi(aoi_trace: np.ndarray) -> np.ndarray:
    """Mean peak AoI per user from a ``(T, K)`` AoI trace.

    A peak is an AoI value immediately followed by a reset to 1.  Users without
    any reset contribute their maximum observed age.
    """
    trace = np.asarray(aoi_trace, dtype=np.float64)
    steps, users = trace.shape
    peaks = np.zeros(users, dtype=np.float64)
    for k in range(users):
        column = trace[:, k]
        resets = np.flatnonzero(column[1:] <= 1.0 + 1e-9) + 1
        if resets.size == 0:
            peaks[k] = float(column.max())
        else:
            peaks[k] = float(np.mean(column[resets - 1]))
    return peaks


@dataclass
class EpisodeMetrics:
    """Metrics of a single evaluation episode."""

    episode: int
    seed: int
    steps: int
    episode_return: float
    weighted_avg_aoi: float
    mean_aoi: float
    max_aoi: float
    peak_aoi: float
    aoi_p95: float
    aoi_p99: float
    aoi_jain: float
    tail_excess: float
    avg_bler: float
    success_rate: float
    avg_power: float
    avg_common_power_ratio: float
    avg_num_scheduled: float
    violation_rate: float
    inference_latency_ms: float
    avg_common_payload_ratio: float = 0.0
    schedule_entropy: float = 0.0
    per_user_aoi: list[float] = field(default_factory=list)
    schedule_count_hist: list[float] = field(default_factory=list)
    activation_rate: list[float] = field(default_factory=list)
    per_class: dict[str, float] = field(default_factory=dict)
    """Per-class aggregates (``aoi_class_far``, ``success_class_near``, ...).

    With a heterogeneous geometry the pooled mean hides the thing that matters -
    whether far users are being starved - so every class-level quantity is
    reported alongside the pooled one."""

    def to_row(self) -> dict[str, float]:
        """Flatten to a CSV-friendly row.

        List fields expand to ``aoi_user_k``, ``sched_frac_c`` and
        ``activation_rate_k`` columns.
        """
        row = asdict(self)
        for source, prefix in (
            ("per_user_aoi", "aoi_user_"),
            ("schedule_count_hist", "sched_frac_"),
            ("activation_rate", "activation_rate_"),
        ):
            values = row.pop(source)
            for index, value in enumerate(values):
                row[f"{prefix}{index}"] = float(value)
        row.update({k: float(v) for k, v in row.pop("per_class").items()})
        return row


def per_class_metrics(
    aoi_trace: np.ndarray,
    masks: np.ndarray,
    acks: np.ndarray,
    classes: list[str] | None,
) -> dict[str, float]:
    """Mean AoI, success rate and scheduling frequency for each user class.

    Args:
        aoi_trace: ``(T, K)`` AoI after each slot.
        masks: ``(T, K)`` boolean schedule.
        acks: ``(T, K)`` per-user ACK indicators.
        classes: per-user class label; ``None`` disables per-class reporting.
    """
    if not classes or len(set(classes)) <= 1:
        return {}
    out: dict[str, float] = {}
    labels = np.asarray(classes)
    for name in sorted(set(classes)):
        members = labels == name
        out[f"aoi_class_{name}"] = float(aoi_trace[:, members].mean())
        out[f"p95_class_{name}"] = float(np.percentile(aoi_trace[:, members], 95))
        scheduled = masks[:, members]
        out[f"sched_freq_class_{name}"] = float(scheduled.mean())
        attempts = scheduled.sum()
        out[f"success_class_{name}"] = (
            float(acks[:, members].sum() / attempts) if attempts > 0 else 0.0
        )
    return out


def summarise_episode(
    episode_index: int,
    seed: int,
    aoi_trace: np.ndarray,
    reward_trace: np.ndarray,
    info_trace: list[dict],
    weights: np.ndarray,
    tail_threshold: float,
    total_power: float,
    latency_ms: float,
    classes: list[str] | None = None,
) -> EpisodeMetrics:
    """Aggregate one episode's traces into :class:`EpisodeMetrics`.

    Args:
        aoi_trace: ``(T, K)`` AoI after each slot.
        reward_trace: ``(T,)`` scaled rewards.
        info_trace: list of the environment's per-step info dicts.
        weights: ``(K,)`` AoI weights.
        tail_threshold: ``tau`` for the tail-excess metric.
        total_power: ``P_max`` used to normalise the radiated power.
        latency_ms: mean policy inference latency in milliseconds.
    """
    aoi = np.asarray(aoi_trace, dtype=np.float64)
    steps, users = aoi.shape
    per_user_mean = aoi.mean(axis=0)

    masks = np.stack([np.asarray(i["scheduled_users"], dtype=bool) for i in info_trace])
    count_hist, sched_entropy, activation = schedule_statistics(masks)
    scheduled_payload = np.array(
        [float(np.sum(i["private_payload"]) + i["common_payload"]) for i in info_trace]
    )
    common_payload = np.array([float(i["common_payload"]) for i in info_trace])
    payload_ratio = float(
        np.mean(common_payload[scheduled_payload > 0] / scheduled_payload[scheduled_payload > 0])
        if np.any(scheduled_payload > 0)
        else 0.0
    )

    successes = np.sum([float(np.sum(i["ack"])) for i in info_trace])
    attempts = np.sum([float(i["num_scheduled"]) for i in info_trace])
    blers = [i["mean_bler"] for i in info_trace if i["num_scheduled"] > 0]
    powers = np.array([i["total_power"] for i in info_trace], dtype=np.float64)
    common = np.array([i["common_power"] for i in info_trace], dtype=np.float64)
    violations = np.array([i["constraint_violation"] for i in info_trace], dtype=np.float64)

    return EpisodeMetrics(
        episode=episode_index,
        seed=seed,
        steps=steps,
        episode_return=float(np.sum(reward_trace)),
        weighted_avg_aoi=float(np.mean(aoi @ np.asarray(weights, dtype=np.float64))),
        mean_aoi=float(per_user_mean.mean()),
        max_aoi=float(aoi.max()),
        peak_aoi=float(np.mean(peak_aoi(aoi))),
        aoi_p95=float(np.percentile(aoi, 95)),
        aoi_p99=float(np.percentile(aoi, 99)),
        aoi_jain=jain_fairness(per_user_mean),
        tail_excess=float(np.mean(np.clip(aoi - tail_threshold, 0.0, None).sum(axis=1))),
        avg_bler=float(np.mean(blers)) if blers else 0.0,
        success_rate=float(successes / attempts) if attempts > 0 else 0.0,
        avg_power=float(powers.mean()),
        avg_common_power_ratio=float(common.mean() / total_power),
        avg_num_scheduled=float(np.mean([i["num_scheduled"] for i in info_trace])),
        violation_rate=float(np.mean(violations > 1e-5)),
        inference_latency_ms=float(latency_ms),
        avg_common_payload_ratio=payload_ratio,
        schedule_entropy=sched_entropy,
        per_user_aoi=[float(v) for v in per_user_mean],
        schedule_count_hist=[float(v) for v in count_hist],
        activation_rate=[float(v) for v in activation],
        per_class=per_class_metrics(
            aoi, masks,
            np.stack([np.asarray(i["ack"], dtype=np.float64) for i in info_trace]),
            classes,
        ),
    )
