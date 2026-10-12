"""Allocation optimisers for one slot and one scheduled subset.

Given a state and a chosen subset ``A``, every allocator returns a **feasible**
RSMA allocation - a power simplex over ``[common, private_A]``, a common-rate
simplex over ``A`` and ``rho_r in (0, 1)`` - maximising the linearised one-step
objective

    gain(alloc) = sum_k urgency_k * p_k(alloc),
    urgency_k   = w_k * (min(Delta_k + 1, Delta_max) - 1)

so that ``expected next weighted AoI = base - gain``.  Feasibility is structural:
every allocator works in the same unconstrained latent space and maps to the
simplex with the masked softmax the policy uses, so no clipping or projection is
required.

NAMING: these are **optimizer references**, not oracles.  None of them is proved
to find the global optimum; :mod:`scripts.verify_optimizer` measures how far
apart they are and how their best-so-far value converges with budget.  Where a
"reference" value is quoted it means *the best value any optimiser here found*,
which is a lower bound on the achievable objective.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer, action_violation, SlotAction
from rsma_aoi.evaluation.oracle import decode_latents

LATENT_SIGMA = 1.5


@dataclass
class AllocationProblem:
    """One (state, subset) allocation instance.

    Attributes:
        phy: physical layer.
        active: ``(K,)`` boolean subset.
        urgency: ``(K,)`` objective weights.
        h_score: channel the returned allocation is finally scored on.
        h_design: channel the beamformers are designed from (CSIT).
        h_opt: channel the search optimises against.
        beams: cached ``(u_priv, u_common)`` for ``(h_design, active)``.
    """

    phy: RSMAPhysicalLayer
    active: np.ndarray
    urgency: np.ndarray
    h_score: np.ndarray
    h_design: np.ndarray
    h_opt: np.ndarray
    beams: tuple[np.ndarray, np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.active = np.asarray(self.active, dtype=bool)
        self.urgency = np.asarray(self.urgency, dtype=np.float64)
        if self.beams is None:
            self.beams = self.phy.rsma_beams(self.h_design, self.active, use_common=True)

    @property
    def num_users(self) -> int:
        return self.phy.num_users

    @property
    def latent_dim(self) -> int:
        return 2 * self.num_users + (1 if self.phy.cfg.rate_split_mode == "per_user_fraction" else 2)

    def evaluate(self, latents: np.ndarray, on_score_channel: bool = False) -> np.ndarray:
        """Objective of a ``(N, latent_dim)`` batch of latents."""
        pf, rf, rho = decode_latents(np.atleast_2d(latents), self.active, self.num_users, self.phy.cfg.rate_split_mode)
        channel = self.h_score if on_score_channel else self.h_opt
        out = self.phy.evaluate_allocations(
            channel, self.active, pf, rf, rho, beams=self.beams
        )
        return out["success_prob"] @ self.urgency

    def score(self, latent: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float, np.ndarray]:
        """Final scoring of one latent on ``h_score``."""
        pf, rf, rho = decode_latents(latent[None, :], self.active, self.num_users, self.phy.cfg.rate_split_mode)
        out = self.phy.evaluate_allocations(
            self.h_score, self.active, pf, rf, rho, beams=self.beams
        )
        success = out["success_prob"][0]
        return float(success @ self.urgency), pf[0], rf[0], float(rho[0]), success


@dataclass
class AllocationResult:
    """What an allocator found."""

    power_fractions: np.ndarray
    rate_fractions: np.ndarray
    rho_r: float
    gain: float
    """Objective on ``h_score`` (higher is better)."""
    success_prob: np.ndarray
    evaluations: int
    runtime_s: float
    allocator: str
    latent: np.ndarray | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    rate_split_mode: str = "simplex_scaled"

    def to_action(self, active: np.ndarray) -> SlotAction:
        return SlotAction(active, self.power_fractions, self.rate_fractions, self.rho_r, rate_split_mode=self.rate_split_mode)

    def violation(self, active: np.ndarray, tol: float = 1e-5) -> float:
        return action_violation(self.to_action(active), tol)


class Allocator(Protocol):
    """Common allocator interface."""

    name: str

    def solve(self, problem: AllocationProblem, rng: np.random.Generator) -> AllocationResult:
        ...


def _finish(
    problem: AllocationProblem, latent: np.ndarray, evaluations: int, start: float, name: str,
    extras: dict[str, Any] | None = None,
) -> AllocationResult:
    gain, pf, rf, rho, success = problem.score(latent)
    return AllocationResult(
        power_fractions=pf, rate_fractions=rf, rho_r=rho, gain=gain,
        success_prob=success, evaluations=evaluations,
        runtime_s=time.perf_counter() - start, allocator=name, latent=latent,
        extras=extras or {}, rate_split_mode=problem.phy.cfg.rate_split_mode,
    )


# ---------------------------------------------------------------------------
class EqualAllocator:
    """Fixed equal split - the cheapest possible reference (1 evaluation).

    ``common_fraction = 0`` reproduces the SDMA-like allocation that the
    ``max_aoi_first`` baseline uses.
    """

    def __init__(self, common_fraction: float = 0.0, rho_r: float = 0.0) -> None:
        self.common_fraction = float(common_fraction)
        self.rho_r = float(rho_r)
        self.name = f"equal(fc={self.common_fraction:g},rho={self.rho_r:g})"

    def solve(self, problem: AllocationProblem, rng: np.random.Generator) -> AllocationResult:
        start = time.perf_counter()
        k = problem.num_users
        active = problem.active
        num_active = max(int(active.sum()), 1)
        pf = np.zeros(k + 1)
        pf[0] = self.common_fraction
        pf[1:][active] = (1.0 - self.common_fraction) / num_active
        rf = np.zeros(k)
        rf[active] = 1.0 / num_active
        out = problem.phy.evaluate_allocations(
            problem.h_score, active, pf[None, :], rf[None, :],
            np.asarray([self.rho_r]), beams=problem.beams,
        )
        success = out["success_prob"][0]
        return AllocationResult(
            power_fractions=pf, rate_fractions=rf, rho_r=self.rho_r,
            gain=float(success @ problem.urgency), success_prob=success,
            evaluations=1, runtime_s=time.perf_counter() - start, allocator=self.name,
        )


class RandomShootingAllocator:
    """Sample ``budget`` latents from ``N(0, sigma^2)`` and keep the best."""

    def __init__(self, budget: int = 960, sigma: float = LATENT_SIGMA) -> None:
        self.budget = int(budget)
        self.sigma = float(sigma)
        self.name = f"random_shooting({self.budget})"

    def solve(self, problem: AllocationProblem, rng: np.random.Generator) -> AllocationResult:
        start = time.perf_counter()
        samples = self.sigma * rng.standard_normal((self.budget, problem.latent_dim))
        gains = problem.evaluate(samples)
        return _finish(problem, samples[int(np.argmax(gains))], self.budget, start, self.name)


class GridAllocator:
    """Coarse structured grid over interpretable allocation shapes.

    Sweeps the common power fraction and ``rho_r`` on a regular grid, crossed
    with a small set of private-power *shapes* (equal, urgency-proportional,
    gain-proportional and their inverses).  This is deliberately low-dimensional:
    it is a sanity reference that cannot exploit fine per-user structure, not a
    competitor to CEM.
    """

    def __init__(self, steps: int = 7) -> None:
        self.steps = int(steps)
        self.name = f"grid({self.steps})"

    def _shapes(self, problem: AllocationProblem) -> list[np.ndarray]:
        active = problem.active
        urgency = np.where(active, np.clip(problem.urgency, 1e-9, None), 0.0)
        gains = np.abs(np.conj(problem.h_score) @ problem.beams[0].T)
        own = np.where(active, np.clip(np.diag(gains) ** 2, 1e-12, None), 0.0)
        equal = active.astype(np.float64)
        candidates = [equal, urgency, own, 1.0 / np.where(own > 0, own, np.inf)]
        out = []
        for shape in candidates:
            shape = np.where(active, shape, 0.0)
            total = shape.sum()
            out.append(shape / total if total > 0 else equal / max(equal.sum(), 1))
        return out

    def solve(self, problem: AllocationProblem, rng: np.random.Generator) -> AllocationResult:
        start = time.perf_counter()
        k = problem.num_users
        active = problem.active
        levels = np.linspace(0.0, 0.9, self.steps)
        rhos = np.linspace(0.0, 1.0, self.steps)
        shapes = self._shapes(problem)
        rate_shapes = shapes[:2]

        rows_pf, rows_rf, rows_rho = [], [], []
        for shape in shapes:
            for rate_shape in rate_shapes:
                for fc in levels:
                    for rho in rhos:
                        pf = np.zeros(k + 1)
                        pf[0] = fc
                        pf[1:] = (1.0 - fc) * shape
                        rows_pf.append(pf)
                        rows_rf.append(rate_shape)
                        rows_rho.append(rho)
        pf = np.stack(rows_pf)
        rf = np.stack(rows_rf)
        rho = np.asarray(rows_rho)
        out = problem.phy.evaluate_allocations(
            problem.h_opt, active, pf, rf, rho, beams=problem.beams
        )
        gains = out["success_prob"] @ problem.urgency
        best = int(np.argmax(gains))

        scored = problem.phy.evaluate_allocations(
            problem.h_score, active, pf[best][None, :], rf[best][None, :],
            rho[best][None], beams=problem.beams,
        )
        success = scored["success_prob"][0]
        return AllocationResult(
            power_fractions=pf[best], rate_fractions=rf[best], rho_r=float(rho[best]),
            gain=float(success @ problem.urgency), success_prob=success,
            evaluations=int(pf.shape[0]), runtime_s=time.perf_counter() - start,
            allocator=self.name,
        )


class CEMAllocator:
    """Cross-entropy method in the latent space, with optional restarts.

    ``evaluations = restarts * iterations * population``.
    """

    def __init__(
        self,
        iterations: int = 5,
        population: int = 192,
        elite_frac: float = 0.15,
        restarts: int = 1,
        sigma: float = LATENT_SIGMA,
        min_sigma: float = 0.05,
        label: str | None = None,
    ) -> None:
        self.iterations = int(iterations)
        self.population = int(population)
        self.elite_frac = float(elite_frac)
        self.restarts = int(restarts)
        self.sigma = float(sigma)
        self.min_sigma = float(min_sigma)
        self.name = label or (
            f"cem({self.restarts}x{self.iterations}x{self.population})"
        )

    def solve(self, problem: AllocationProblem, rng: np.random.Generator) -> AllocationResult:
        start = time.perf_counter()
        dim = problem.latent_dim
        n_elite = max(4, int(self.population * self.elite_frac))
        best_latent, best_gain = None, -np.inf
        history: list[float] = []

        for restart in range(self.restarts):
            mu = np.zeros(dim) if restart == 0 else self.sigma * rng.standard_normal(dim)
            sigma = np.full(dim, self.sigma)
            for _ in range(self.iterations):
                sample = mu[None, :] + sigma[None, :] * rng.standard_normal((self.population, dim))
                gains = problem.evaluate(sample)
                order = np.argsort(-gains)
                elite = sample[order[:n_elite]]
                mu = elite.mean(axis=0)
                sigma = np.maximum(elite.std(axis=0), self.min_sigma)
                if gains[order[0]] > best_gain:
                    best_gain = float(gains[order[0]])
                    best_latent = sample[order[0]].copy()
                history.append(best_gain)

        evaluations = self.restarts * self.iterations * self.population
        return _finish(problem, best_latent, evaluations, start, self.name,
                       {"best_so_far": history})


class SLSQPAllocator:
    """Gradient-based local search (SciPy) from one or more random starts.

    The latent space is unconstrained - feasibility comes from the softmax - so
    this is a plain bounded minimisation, not a constrained programme.  Gradients
    are finite-difference; the FBL objective is piecewise-steep, so multiple
    starts matter more than the exact method.
    """

    def __init__(self, restarts: int = 4, maxiter: int = 60, method: str = "SLSQP",
                 bound: float = 6.0) -> None:
        self.restarts = int(restarts)
        self.maxiter = int(maxiter)
        self.method = method
        self.bound = float(bound)
        self.name = f"{method.lower()}({self.restarts}x{self.maxiter})"

    def solve(self, problem: AllocationProblem, rng: np.random.Generator) -> AllocationResult:
        from scipy.optimize import minimize

        start = time.perf_counter()
        dim = problem.latent_dim
        counter = {"n": 0}

        def negative(x: np.ndarray) -> float:
            counter["n"] += 1
            return -float(problem.evaluate(x[None, :])[0])

        bounds = [(-self.bound, self.bound)] * dim
        best_latent, best_value = None, np.inf
        for restart in range(self.restarts):
            x0 = np.zeros(dim) if restart == 0 else LATENT_SIGMA * rng.standard_normal(dim)
            res = minimize(negative, x0, method=self.method, bounds=bounds,
                           options={"maxiter": self.maxiter})
            if res.fun < best_value:
                best_value, best_latent = float(res.fun), np.asarray(res.x, dtype=np.float64)
        return _finish(problem, best_latent, counter["n"], start, self.name)


class CachedAllocator:
    """Nearest-neighbour cache in a normalised feature space.

    WARNING: this is an **approximation**, not memoisation. Entries are reused
    when the query feature vector lies within ``tol`` of a stored one, so a hit
    returns the allocation of a *similar* state, not of this state. The hit rate
    and the induced objective bias are recorded in :attr:`stats` and must be
    reported wherever the cache is used.
    """

    def __init__(
        self,
        base: Allocator,
        feature_fn: Callable[[AllocationProblem], np.ndarray] | None = None,
        tol: float = 0.15,
        capacity: int = 20_000,
    ) -> None:
        self.base = base
        self.tol = float(tol)
        self.capacity = int(capacity)
        self.feature_fn = feature_fn or default_features
        self.name = f"cached({base.name},tol={self.tol:g})"
        self._keys: list[np.ndarray] = []
        self._values: list[AllocationResult] = []
        self.stats = {"queries": 0, "hits": 0, "bias_sum": 0.0, "bias_count": 0}

    @property
    def hit_rate(self) -> float:
        return self.stats["hits"] / max(self.stats["queries"], 1)

    @property
    def mean_bias(self) -> float:
        """Mean ``gain_cached - gain_exact`` on audited hits (negative = worse)."""
        return self.stats["bias_sum"] / max(self.stats["bias_count"], 1)

    def solve(
        self, problem: AllocationProblem, rng: np.random.Generator, audit: bool = False
    ) -> AllocationResult:
        self.stats["queries"] += 1
        key = np.asarray(self.feature_fn(problem), dtype=np.float64)
        if self._keys:
            table = np.stack(self._keys)
            distances = np.linalg.norm(table - key[None, :], axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] <= self.tol:
                self.stats["hits"] += 1
                cached = self._values[nearest]
                latent = cached.latent
                if latent is None:
                    return cached
                gain, pf, rf, rho, success = problem.score(latent)
                result = AllocationResult(
                    power_fractions=pf, rate_fractions=rf, rho_r=rho, gain=gain,
                    success_prob=success, evaluations=0, runtime_s=0.0,
                    allocator=self.name, latent=latent, extras={"cache": "hit"},
                    rate_split_mode=problem.phy.cfg.rate_split_mode,
                )
                if audit:
                    exact = self.base.solve(problem, rng)
                    self.stats["bias_sum"] += result.gain - exact.gain
                    self.stats["bias_count"] += 1
                return result

        result = self.base.solve(problem, rng)
        if len(self._keys) < self.capacity:
            self._keys.append(key)
            self._values.append(result)
        return result


def default_features(problem: AllocationProblem) -> np.ndarray:
    """Scale-normalised state descriptor used by :class:`CachedAllocator`.

    Contains the active mask, the normalised urgency profile and the normalised
    effective channel gains of the active users - the quantities the allocation
    actually depends on.
    """
    active = problem.active.astype(np.float64)
    urgency = problem.urgency * active
    total = urgency.sum()
    urgency_norm = urgency / total if total > 0 else active / max(active.sum(), 1)
    gains = np.abs(np.conj(problem.h_design) @ problem.beams[0].T)
    own = np.diag(gains) ** 2 * active
    scale = own.max()
    own_norm = own / scale if scale > 0 else own
    return np.concatenate([active, urgency_norm, own_norm])


# ---------------------------------------------------------------------------
def make_problem(
    phy: RSMAPhysicalLayer,
    aoi: np.ndarray,
    h_true: np.ndarray,
    h_hat: np.ndarray,
    weights: np.ndarray,
    aoi_max: float,
    active: np.ndarray,
    csit_mode: str = "estimated",
) -> tuple[AllocationProblem, float]:
    """Build a problem for one state/subset; returns ``(problem, base)``.

    ``expected next weighted AoI = base - gain``.
    """
    from rsma_aoi.evaluation.oracle import urgency_vector

    urgency, base = urgency_vector(aoi, weights, aoi_max)
    h_opt = h_true if csit_mode == "genie" else h_hat
    return (
        AllocationProblem(
            phy=phy, active=active, urgency=urgency,
            h_score=h_true, h_design=h_hat, h_opt=h_opt,
        ),
        base,
    )


#: Named allocator presets used by the phase map and the hierarchical policy.
ALLOCATORS: dict[str, Callable[[], Allocator]] = {
    "equal": lambda: EqualAllocator(0.0, 0.0),
    "equal_rsma": lambda: EqualAllocator(0.25, 0.3),
    "grid": lambda: GridAllocator(7),
    "random_shooting": lambda: RandomShootingAllocator(960),
    "cem_fast": lambda: CEMAllocator(iterations=3, population=48, label="cem_fast(144)"),
    "cem_full": lambda: CEMAllocator(iterations=5, population=192, label="cem_full(960)"),
    "cem_heavy": lambda: CEMAllocator(iterations=8, population=384, restarts=3,
                                      label="cem_heavy(9216)"),
    "slsqp": lambda: SLSQPAllocator(restarts=4, maxiter=60),
}


def build_allocator(name: str) -> Allocator:
    """Instantiate a named allocator preset."""
    if name not in ALLOCATORS:
        raise KeyError(f"unknown allocator {name!r}; available: {sorted(ALLOCATORS)}")
    return ALLOCATORS[name]()
