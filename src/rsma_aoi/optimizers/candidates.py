"""Candidate scheduling subsets without enumerating all ``2^K - 1`` of them.

Enumeration is fine for ``K = 4`` (15 subsets) but not for ``K = 8`` (255) and
impossible beyond that.  These generators propose a small, valid candidate set
using the quantities the objective actually depends on:

* **urgency** ``w_k (min(Delta_k + 1, Delta_max) - 1)`` - who needs a packet;
* **channel strength** - who can be served cheaply;
* **channel compatibility** - which users can share the spatial dimensions.

Every generator returns non-empty boolean masks whose sizes lie in the requested
``sizes`` set, de-duplicated and capped at ``max_candidates``.
"""

from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np

DEFAULT_SIZES = (2, 3, 4)


def _mask_from_indices(indices: Iterable[int], num_users: int) -> np.ndarray:
    mask = np.zeros(num_users, dtype=bool)
    mask[list(indices)] = True
    return mask


def _dedup(masks: list[np.ndarray], max_candidates: int | None) -> list[np.ndarray]:
    seen: set[bytes] = set()
    out: list[np.ndarray] = []
    for mask in masks:
        if not mask.any():
            continue
        key = mask.tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(mask)
        if max_candidates is not None and len(out) >= max_candidates:
            break
    return out


def normalise_sizes(sizes: Iterable[int] | None, num_users: int) -> tuple[int, ...]:
    """Clamp requested subset sizes to ``1..K`` (default :data:`DEFAULT_SIZES`)."""
    if sizes is None:
        sizes = DEFAULT_SIZES
    valid = sorted({int(s) for s in sizes if 1 <= int(s) <= num_users})
    return tuple(valid) if valid else (min(2, num_users),)


def top_m_candidates(score: np.ndarray, sizes: Iterable[int], num_users: int) -> list[np.ndarray]:
    """Highest-``m`` users by ``score``, one mask per requested size."""
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
    return [_mask_from_indices(order[:m], num_users) for m in normalise_sizes(sizes, num_users)]


def channel_gain_score(h_hat: np.ndarray) -> np.ndarray:
    """Per-user channel strength ``||h_k||^2``."""
    return np.sum(np.abs(h_hat) ** 2, axis=1)


def compatibility_candidates(
    h_hat: np.ndarray, sizes: Iterable[int], num_users: int, seeds: int = 2
) -> list[np.ndarray]:
    """Greedy spatially-compatible groups (low mutual channel correlation).

    Starting from each of the ``seeds`` strongest users, repeatedly add the user
    whose normalised channel is least correlated with the ones already chosen.
    """
    norms = np.linalg.norm(h_hat, axis=1)
    safe = np.where(norms > 1e-12, norms, 1.0)
    unit = h_hat / safe[:, None]
    correlation = np.abs(unit @ np.conj(unit).T)
    np.fill_diagonal(correlation, 0.0)
    strength = channel_gain_score(h_hat)
    starts = np.argsort(-strength, kind="stable")[:max(1, seeds)]

    masks: list[np.ndarray] = []
    for size in normalise_sizes(sizes, num_users):
        for start in starts:
            chosen = [int(start)]
            while len(chosen) < size:
                penalty = correlation[chosen].max(axis=0)
                penalty[chosen] = np.inf
                nxt = int(np.argmin(penalty))
                if not np.isfinite(penalty[nxt]):
                    break
                chosen.append(nxt)
            if len(chosen) == size:
                masks.append(_mask_from_indices(chosen, num_users))
    return masks


def beam_search_candidates(
    objective, sizes: Iterable[int], num_users: int, beam_width: int = 3
) -> list[np.ndarray]:
    """Greedy sequential selection with a beam, using an arbitrary objective.

    Args:
        objective: callable ``mask -> float`` (higher is better).  It is called
            ``O(beam_width * K * max(sizes))`` times, never ``2^K``.
        sizes: subset sizes to emit.
        beam_width: partial solutions retained at each step.
    """
    sizes = normalise_sizes(sizes, num_users)
    target = max(sizes)
    beam: list[tuple[float, list[int]]] = [(0.0, [])]
    emitted: list[np.ndarray] = []
    for step in range(1, target + 1):
        expanded: list[tuple[float, list[int]]] = []
        for _, chosen in beam:
            for user in range(num_users):
                if user in chosen:
                    continue
                candidate = sorted(chosen + [user])
                expanded.append((float(objective(_mask_from_indices(candidate, num_users))),
                                 candidate))
        if not expanded:
            break
        expanded.sort(key=lambda item: -item[0])
        unique: list[tuple[float, list[int]]] = []
        seen: set[tuple[int, ...]] = set()
        for value, chosen in expanded:
            key = tuple(chosen)
            if key in seen:
                continue
            seen.add(key)
            unique.append((value, chosen))
            if len(unique) >= beam_width:
                break
        beam = unique
        if step in sizes:
            emitted.extend(_mask_from_indices(chosen, num_users) for _, chosen in beam)
    return emitted


def generate_candidates(
    aoi: np.ndarray,
    weights: np.ndarray,
    h_hat: np.ndarray,
    aoi_max: float,
    sizes: Iterable[int] | None = None,
    max_candidates: int | None = 10,
    include_exhaustive_below: int = 4,
) -> list[np.ndarray]:
    """Candidate subsets for one state.

    Combines max-AoI, channel-strength, AoI x channel and spatial-compatibility
    proposals.  For ``K <= include_exhaustive_below`` it simply enumerates every
    subset of the requested sizes, so small problems stay exact.

    Returns:
        A de-duplicated list of non-empty boolean masks.
    """
    num_users = int(np.asarray(aoi).shape[0])
    sizes = normalise_sizes(sizes, num_users)

    if num_users <= include_exhaustive_below:
        masks = [
            _mask_from_indices(combo, num_users)
            for size in sizes
            for combo in itertools.combinations(range(num_users), size)
        ]
        return _dedup(masks, None)

    aged = np.minimum(np.asarray(aoi, dtype=np.float64) + 1.0, aoi_max)
    urgency = np.asarray(weights, dtype=np.float64) * (aged - 1.0)
    gains = channel_gain_score(h_hat)

    masks: list[np.ndarray] = []
    masks += top_m_candidates(urgency, sizes, num_users)
    masks += top_m_candidates(gains, sizes, num_users)
    masks += top_m_candidates(urgency * gains, sizes, num_users)
    masks += compatibility_candidates(h_hat, sizes, num_users)
    return _dedup(masks, max_candidates)


def candidates_are_valid(
    masks: list[np.ndarray], num_users: int, sizes: Iterable[int] | None = None
) -> bool:
    """True when every mask is non-empty, correctly shaped and of an allowed size."""
    allowed = set(normalise_sizes(sizes, num_users))
    for mask in masks:
        mask = np.asarray(mask)
        if mask.shape != (num_users,) or mask.dtype != bool or not mask.any():
            return False
        if int(mask.sum()) not in allowed:
            return False
    return True
