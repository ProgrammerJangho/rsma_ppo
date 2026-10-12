"""Seed management: independence, reproducibility and stream separation."""

from __future__ import annotations

import numpy as np
import torch

from rsma_aoi.utils.seeding import (
    SeedBundle, make_rng, make_torch_generator, seed_everything, spawn_rngs, spawn_seeds,
)


def test_make_rng_is_reproducible():
    assert make_rng(7).random() == make_rng(7).random()
    assert make_rng(7).random() != make_rng(8).random()


def test_spawn_seeds_are_distinct_and_reproducible():
    seeds = spawn_seeds(0, 8)
    assert len(seeds) == len(set(seeds)) == 8
    assert seeds == spawn_seeds(0, 8)
    assert seeds != spawn_seeds(1, 8)
    assert all(isinstance(s, int) and s >= 0 for s in seeds)


def test_spawned_streams_are_uncorrelated():
    rngs = spawn_rngs(0, 4)
    samples = np.stack([r.standard_normal(5000) for r in rngs])
    corr = np.corrcoef(samples)
    off_diagonal = corr[~np.eye(4, dtype=bool)]
    assert np.max(np.abs(off_diagonal)) < 0.05


def test_adjacent_integer_seeds_do_not_produce_correlated_streams():
    """The trainer uses spawn_seeds precisely to avoid this failure mode."""
    seeds = spawn_seeds(123, 4)
    samples = np.stack([np.random.default_rng(s).standard_normal(5000) for s in seeds])
    corr = np.corrcoef(samples)
    assert np.max(np.abs(corr[~np.eye(4, dtype=bool)])) < 0.05


def test_torch_generator_is_explicit_and_reproducible():
    a = torch.randn(4, generator=make_torch_generator(3))
    b = torch.randn(4, generator=make_torch_generator(3))
    c = torch.randn(4, generator=make_torch_generator(4))
    torch.testing.assert_close(a, b)
    assert not torch.allclose(a, c)


def test_seed_bundle_separates_the_streams():
    bundle = SeedBundle(5)
    assert len({bundle.env, bundle.eval_env, bundle.agent}) == 3
    assert SeedBundle(5).env == bundle.env
    assert SeedBundle(6).env != bundle.env


def test_seed_everything_makes_weight_initialisation_reproducible():
    def init() -> torch.Tensor:
        seed_everything(11)
        return torch.nn.Linear(4, 4).weight.detach().clone()

    torch.testing.assert_close(init(), init())
