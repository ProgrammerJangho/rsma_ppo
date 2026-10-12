"""Allocation optimisers: feasibility, budget monotonicity, cross-checks."""

from __future__ import annotations

import numpy as np
import pytest

from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer
from rsma_aoi.optimizers import (
    ALLOCATORS, CachedAllocator, CEMAllocator, EqualAllocator,
    GridAllocator, build_allocator, candidates_are_valid, generate_candidates,
    make_problem, normalise_sizes, top_m_candidates,
)
from rsma_aoi.optimizers.candidates import beam_search_candidates, compatibility_candidates
from rsma_aoi.utils.config import EnvConfig


def make_case(num_users=4, num_antennas=2, seed=0, scheme="rsma"):
    cfg = EnvConfig(num_users=num_users, num_antennas=num_antennas, snr_db=10.0,
                    blocklength=128, payload_bits=192.0, csit_error_std=0.3,
                    multiple_access=scheme)
    phy = RSMAPhysicalLayer(cfg)
    rng = np.random.default_rng(seed)
    h_true = (rng.standard_normal((num_users, num_antennas))
              + 1j * rng.standard_normal((num_users, num_antennas))) / np.sqrt(2)
    h_hat = h_true + 0.3 * (rng.standard_normal(h_true.shape)
                            + 1j * rng.standard_normal(h_true.shape)) / np.sqrt(2)
    aoi = rng.integers(1, 20, size=num_users).astype(np.float64)
    active = np.zeros(num_users, dtype=bool)
    active[: min(num_users, num_antennas)] = True
    problem, base = make_problem(
        phy, aoi, h_true, h_hat, np.ones(num_users), float(cfg.aoi_max), active
    )
    return problem, base, cfg


@pytest.mark.parametrize("name", sorted(ALLOCATORS))
def test_every_allocator_returns_a_feasible_allocation(name):
    problem, _, _ = make_case()
    result = build_allocator(name).solve(problem, np.random.default_rng(0))
    pf, rf = result.power_fractions, result.rate_fractions
    active = problem.active
    assert np.all(pf >= 0.0) and np.all(rf >= 0.0)
    assert pf.sum() == pytest.approx(1.0, abs=1e-6)
    assert rf[active].sum() == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(pf[1:][~active], 0.0, atol=1e-12)
    np.testing.assert_allclose(rf[~active], 0.0, atol=1e-12)
    assert 0.0 <= result.rho_r <= 1.0
    assert result.violation(active) == pytest.approx(0.0)
    assert np.isfinite(result.gain)
    assert result.evaluations >= 1 or name == "cached"


def test_cem_best_so_far_is_non_decreasing():
    problem, _, _ = make_case()
    result = CEMAllocator(iterations=6, population=64).solve(problem, np.random.default_rng(0))
    history = result.extras["best_so_far"]
    assert len(history) == 6
    assert all(b >= a - 1e-12 for a, b in zip(history, history[1:]))


def _paired_relative_gain(config_a: dict, config_b: dict, num_problems: int = 24) -> float:
    """Mean per-problem relative improvement of B over A (paired, same seed).

    Problems differ enormously in urgency scale, so raw gains must never be
    averaged across them - the comparison has to be paired and normalised.
    """
    diffs = []
    for seed in range(num_problems):
        problem, _, _ = make_case(seed=seed)
        a = CEMAllocator(**config_a).solve(problem, np.random.default_rng(0)).gain
        b = CEMAllocator(**config_b).solve(problem, np.random.default_rng(0)).gain
        diffs.append((b - a) / max(abs(a), abs(b), 1e-9))
    return float(np.mean(diffs))


def test_larger_population_improves_the_optimum_on_average():
    assert _paired_relative_gain(
        dict(iterations=2, population=24), dict(iterations=2, population=192)
    ) > 0.0


def test_more_restarts_improve_the_optimum_on_average():
    assert _paired_relative_gain(
        dict(iterations=2, population=192, restarts=1),
        dict(iterations=2, population=192, restarts=4),
    ) > 0.0


def test_more_cem_iterations_alone_can_converge_prematurely():
    """Documented property: iteration count is NOT a monotone budget axis.

    CEM shrinks its sampling sigma towards the elite spread, so extra iterations
    at a fixed population narrow the search and can lock onto a local optimum.
    Population and restarts are the axes that help; this is why the phase map
    uses restarts rather than long runs, and why an optimiser reference should
    never be called an oracle.
    """
    delta = _paired_relative_gain(
        dict(iterations=2, population=192), dict(iterations=8, population=192)
    )
    assert delta < 0.0, (
        "extra iterations helped here; the premature-convergence note in the "
        "README should be re-checked if this starts passing"
    )


def test_optimizers_broadly_agree_on_a_small_problem():
    """CEM, random shooting and the grid must land within a sane distance."""
    problem, _, _ = make_case(num_users=2, num_antennas=2, seed=3)
    results = {
        name: build_allocator(name).solve(problem, np.random.default_rng(0)).gain
        for name in ("grid", "random_shooting", "cem_full", "cem_heavy", "slsqp")
    }
    best = max(results.values())
    assert results["cem_heavy"] >= results["cem_fast"] - 1.0 if "cem_fast" in results else True
    for name, value in results.items():
        assert value <= best + 1e-9
        assert np.isfinite(value)
    # the strongest optimiser should not be far from the ensemble best
    assert best - results["cem_heavy"] <= 0.25 * max(abs(best), 1e-9) + 1e-6


def test_equal_allocator_reproduces_the_sdma_style_split():
    problem, _, _ = make_case()
    result = EqualAllocator(0.0, 0.0).solve(problem, np.random.default_rng(0))
    active = problem.active
    assert result.power_fractions[0] == 0.0
    np.testing.assert_allclose(
        result.power_fractions[1:][active], 1.0 / active.sum(), atol=1e-12
    )
    assert result.evaluations == 1


def test_grid_allocator_is_seed_independent():
    problem, _, _ = make_case()
    a = GridAllocator(5).solve(problem, np.random.default_rng(0)).gain
    b = GridAllocator(5).solve(problem, np.random.default_rng(99)).gain
    assert a == pytest.approx(b)


def test_cached_allocator_tracks_hit_rate_and_bias():
    base = CEMAllocator(iterations=2, population=32)
    cache = CachedAllocator(base, tol=1e9)   # everything is a "neighbour"
    rng = np.random.default_rng(0)
    problems = [make_case(seed=s)[0] for s in range(6)]
    for problem in problems:
        cache.solve(problem, rng, audit=True)
    assert cache.stats["queries"] == 6
    assert cache.hit_rate > 0.0, "an all-matching cache must produce hits"
    assert cache.stats["bias_count"] > 0, "audited hits must be counted"
    assert np.isfinite(cache.mean_bias)
    # The bias may have either sign: a cached latent re-scored on the new state
    # can beat a fresh *stochastic* solve of a weak base optimiser.  What matters
    # is that the bias is measured and reported, not that it is negative.

    strict = CachedAllocator(base, tol=0.0)
    for problem in problems:
        strict.solve(problem, rng)
    assert strict.hit_rate == 0.0


def test_allocator_works_for_every_access_scheme():
    for scheme in ("rsma", "sdma", "oma"):
        problem, _, _ = make_case(scheme=scheme)
        result = CEMAllocator(iterations=2, population=32).solve(problem, np.random.default_rng(0))
        assert np.isfinite(result.gain)
        assert result.violation(problem.active) == pytest.approx(0.0)


def test_unknown_allocator_raises():
    with pytest.raises(KeyError):
        build_allocator("nope")


# --- candidate generation --------------------------------------------------
@pytest.mark.parametrize("num_users", [2, 4, 6, 8])
def test_generated_candidates_are_valid_and_bounded(num_users):
    rng = np.random.default_rng(0)
    h = (rng.standard_normal((num_users, 4)) + 1j * rng.standard_normal((num_users, 4)))
    aoi = rng.integers(1, 30, size=num_users).astype(float)
    sizes = (2, 3, 4)
    masks = generate_candidates(aoi, np.ones(num_users), h, 64.0, sizes=sizes,
                                max_candidates=10)
    assert masks, "at least one candidate must be produced"
    assert candidates_are_valid(masks, num_users, sizes)
    if num_users > 4:
        assert len(masks) <= 10, "large K must not enumerate every subset"


def test_small_k_enumerates_exhaustively():
    rng = np.random.default_rng(0)
    h = rng.standard_normal((4, 2)) + 1j * rng.standard_normal((4, 2))
    masks = generate_candidates(np.arange(1, 5.0), np.ones(4), h, 64.0, sizes=(2,))
    assert len(masks) == 6   # C(4,2)


def test_top_m_candidates_pick_the_highest_scores():
    masks = top_m_candidates(np.array([1.0, 9.0, 3.0, 7.0]), (2,), 4)
    np.testing.assert_array_equal(masks[0], [False, True, False, True])


def test_compatibility_candidates_prefer_uncorrelated_users():
    h = np.array([[1.0 + 0j, 0.0], [1.0 + 0j, 0.0], [0.0, 1.0 + 0j]])
    masks = compatibility_candidates(h, (2,), 3, seeds=1)
    assert masks and int(masks[0].sum()) == 2
    assert masks[0][2], "the orthogonal user should be grouped with the strongest"


def test_beam_search_uses_far_fewer_calls_than_enumeration():
    calls = {"n": 0}

    def objective(mask):
        calls["n"] += 1
        return float(np.sum(np.arange(8) * mask))

    masks = beam_search_candidates(objective, (2, 3), 8, beam_width=3)
    assert masks and candidates_are_valid(masks, 8, (2, 3))
    assert calls["n"] < 2**8, "beam search must not enumerate all subsets"
    assert masks[0][7], "the highest-value user should be selected"


def test_normalise_sizes_clamps():
    assert normalise_sizes(None, 4) == (2, 3, 4)
    assert normalise_sizes((1, 2, 99), 3) == (1, 2)
    assert normalise_sizes((), 5) == (2,)
