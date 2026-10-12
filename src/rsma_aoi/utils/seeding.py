"""Centralised seeding helpers.

The project never relies on the *global* NumPy / PyTorch random state for
anything that must be reproducible.  Environments own an explicit
``numpy.random.Generator``; agents own an explicit ``torch.Generator``.
``seed_everything`` is only used to make library-internal (e.g. weight
initialisation) randomness reproducible.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


def make_rng(seed: int | None) -> np.random.Generator:
    """Return an independent NumPy generator (``None`` -> OS entropy)."""
    return np.random.default_rng(seed)


def spawn_rngs(seed: int | None, n: int) -> list[np.random.Generator]:
    """Return ``n`` statistically independent generators derived from ``seed``.

    Uses :class:`numpy.random.SeedSequence` spawning so that vectorised
    environments never share or correlate streams.
    """
    seq = np.random.SeedSequence(seed)
    return [np.random.default_rng(s) for s in seq.spawn(n)]


def spawn_seeds(seed: int | None, n: int) -> list[int]:
    """Return ``n`` well-separated integer seeds derived from ``seed``.

    Uses :class:`numpy.random.SeedSequence` spawning rather than ``seed + i`` so
    that parallel environments cannot end up with correlated streams.
    """
    seq = np.random.SeedSequence(seed)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in seq.spawn(n)]


def make_torch_generator(seed: int | None, device: str | torch.device = "cpu") -> torch.Generator:
    """Return an explicit torch generator bound to ``device``."""
    gen = torch.Generator(device=device)
    if seed is None:
        gen.seed()
    else:
        gen.manual_seed(int(seed))
    return gen


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed python / numpy / torch global RNGs (used for weight init only)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on hardware
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(False)  # keep CPU kernels fast
        os.environ.setdefault("PYTHONHASHSEED", str(seed))


@dataclass(frozen=True)
class SeedBundle:
    """Grouped seeds so that env / eval / agent streams never collide."""

    base: int

    @property
    def env(self) -> int:
        return self.base * 1000 + 1

    @property
    def eval_env(self) -> int:
        return self.base * 1000 + 2

    @property
    def agent(self) -> int:
        return self.base * 1000 + 3
