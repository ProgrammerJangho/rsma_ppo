"""Welford running mean/variance used for observation and reward normalisation."""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Parallel (Chan et al.) running estimate of mean and variance.

    Attributes:
        mean: ``(shape,)`` running mean.
        var: ``(shape,)`` running variance.
        count: pseudo-count (starts at ``epsilon`` for numerical stability).
    """

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, batch: np.ndarray) -> None:
        """Update from a batch of shape ``(N, *shape)``."""
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == self.mean.ndim:
            batch = batch[None, ...]
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot
        self.mean = new_mean
        self.var = np.maximum(m2 / tot, 1e-12)
        self.count = tot

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        """Standardise ``x`` and clip to ``[-clip, clip]``."""
        out = (np.asarray(x, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(out, -clip, clip).astype(np.float32)

    def state_dict(self) -> dict[str, object]:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])  # type: ignore[arg-type]
