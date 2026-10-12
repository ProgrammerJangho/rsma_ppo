"""GAE: hand-computed examples and correct terminated/truncated handling."""

from __future__ import annotations

import numpy as np
import pytest

from rsma_aoi.agents.rollout_buffer import RolloutBuffer, compute_gae


def test_single_step_no_bootstrap():
    """One terminal step: A = r - V(s)."""
    adv, ret = compute_gae(
        rewards=np.array([[1.0]]), values=np.array([[0.5]]), next_values=np.array([[7.0]]),
        terminated=np.array([[True]]), truncated=np.array([[False]]),
        gamma=0.99, gae_lambda=0.95,
    )
    assert adv[0, 0] == pytest.approx(0.5)   # next_values ignored on termination
    assert ret[0, 0] == pytest.approx(1.0)


def test_truncation_bootstraps_with_the_final_value():
    """Time-limit cut: A = r + gamma * V(final) - V(s)."""
    adv, _ = compute_gae(
        rewards=np.array([[1.0]]), values=np.array([[0.5]]), next_values=np.array([[2.0]]),
        terminated=np.array([[False]]), truncated=np.array([[True]]),
        gamma=0.9, gae_lambda=0.95,
    )
    assert adv[0, 0] == pytest.approx(1.0 + 0.9 * 2.0 - 0.5)


def test_three_step_hand_calculation():
    """gamma=0.9, lambda=0.8, r=[1,1,1], V=[.5,.5,.5], terminal at t=2.

    delta_2 = 1 + 0.9*0*0   - 0.5 = 0.5     -> A_2 = 0.5
    delta_1 = 1 + 0.9*0.5*1 - 0.5 = 0.95    -> A_1 = 0.95 + 0.72*0.5  = 1.31
    delta_0 = 1 + 0.9*0.5*1 - 0.5 = 0.95    -> A_0 = 0.95 + 0.72*1.31 = 1.8932
    """
    adv, ret = compute_gae(
        rewards=np.ones((3, 1)),
        values=np.full((3, 1), 0.5),
        next_values=np.array([[0.5], [0.5], [0.0]]),
        terminated=np.array([[False], [False], [True]]),
        truncated=np.zeros((3, 1), dtype=bool),
        gamma=0.9, gae_lambda=0.8,
    )
    np.testing.assert_allclose(adv[:, 0], [1.8932, 1.31, 0.5], rtol=1e-12)
    np.testing.assert_allclose(ret[:, 0], np.array([1.8932, 1.31, 0.5]) + 0.5, rtol=1e-12)


def test_lambda_one_reduces_to_the_monte_carlo_return():
    gamma = 0.9
    rewards = np.array([[1.0], [2.0], [3.0]])
    values = np.array([[0.3], [0.7], [1.1]])
    adv, ret = compute_gae(
        rewards, values, next_values=np.array([[0.7], [1.1], [0.0]]),
        terminated=np.array([[False], [False], [True]]),
        truncated=np.zeros((3, 1), dtype=bool), gamma=gamma, gae_lambda=1.0,
    )
    discounted = 1.0 + gamma * 2.0 + gamma**2 * 3.0
    assert ret[0, 0] == pytest.approx(discounted)
    assert adv[0, 0] == pytest.approx(discounted - values[0, 0])


def test_lambda_zero_reduces_to_the_td_error():
    adv, _ = compute_gae(
        rewards=np.array([[1.0], [2.0]]), values=np.array([[0.4], [0.6]]),
        next_values=np.array([[0.6], [0.0]]),
        terminated=np.array([[False], [True]]), truncated=np.zeros((2, 1), dtype=bool),
        gamma=0.5, gae_lambda=0.0,
    )
    assert adv[0, 0] == pytest.approx(1.0 + 0.5 * 0.6 - 0.4)
    assert adv[1, 0] == pytest.approx(2.0 - 0.6)


def test_episode_boundaries_stop_the_advantage_chain():
    """A truncation at t=0 must not leak advantage from t=1 backwards."""
    adv, _ = compute_gae(
        rewards=np.ones((2, 1)), values=np.zeros((2, 1)), next_values=np.zeros((2, 1)),
        terminated=np.zeros((2, 1), dtype=bool),
        truncated=np.array([[True], [False]]), gamma=0.9, gae_lambda=0.95,
    )
    assert adv[0, 0] == pytest.approx(1.0)


def test_multiple_environments_are_independent():
    rewards = np.array([[1.0, 2.0], [1.0, 2.0]])
    adv, _ = compute_gae(
        rewards, values=np.zeros((2, 2)), next_values=np.zeros((2, 2)),
        terminated=np.array([[False, True], [True, True]]),
        truncated=np.zeros((2, 2), dtype=bool), gamma=0.9, gae_lambda=1.0,
    )
    assert adv[0, 0] == pytest.approx(1.0 + 0.9 * 1.0)
    assert adv[0, 1] == pytest.approx(2.0)


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        compute_gae(np.zeros((2, 1)), np.zeros((3, 1)), np.zeros((2, 1)),
                    np.zeros((2, 1), bool), np.zeros((2, 1), bool), 0.9, 0.95)


# --- buffer integration ----------------------------------------------------
def test_buffer_roundtrip_and_minibatching():
    import torch

    from rsma_aoi.agents.distributions import LatentAction

    steps, envs, k, obs_dim = 4, 2, 3, 5
    buffer = RolloutBuffer(steps, envs, obs_dim, k)
    rng = np.random.default_rng(0)
    for t in range(steps):
        latent = LatentAction(
            schedule_raw=torch.ones(envs, k),
            active_mask=torch.ones(envs, k, dtype=torch.bool),
            power_latent=torch.randn(envs, k + 1),
            rate_latent=torch.randn(envs, k),
            rho_r_latent=torch.rand(envs, 1),
        )
        buffer.add(
            t, rng.standard_normal((envs, obs_dim)).astype(np.float32), latent,
            log_prob=np.zeros(envs), value=np.zeros(envs), reward=np.ones(envs),
            terminated=np.zeros(envs, bool), truncated=np.zeros(envs, bool),
        )
        buffer.set_next_values(t, np.zeros(envs))
    adv, ret = buffer.compute_returns(0.99, 0.95)
    assert adv.shape == (steps, envs) and ret.shape == (steps, envs)

    batches = buffer.iter_minibatches(4, np.random.default_rng(1))
    assert len(batches) == (steps * envs) // 4
    assert batches[0].obs.shape == (4, obs_dim)
    assert batches[0].latent.power_latent.shape == (4, k + 1)
    assert batches[0].latent.active_mask.dtype == torch.bool

    with pytest.raises(ValueError):
        buffer.iter_minibatches(3, np.random.default_rng(2))
    with pytest.raises(IndexError):
        buffer.add(99, np.zeros((envs, obs_dim)), latent, np.zeros(envs), np.zeros(envs),
                   np.ones(envs), np.zeros(envs, bool), np.zeros(envs, bool))
