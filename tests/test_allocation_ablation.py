"""Fixed actions must not contribute likelihood, entropy, or head gradients."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from rsma_aoi.agents.networks import HybridActorCritic
from rsma_aoi.agents.ppo import PPOPolicy, PPOTrainer
from rsma_aoi.envs.aoi_env import AoIRSMAEnv
from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer, SlotAction
from rsma_aoi.utils.config import EnvConfig, PPOConfig, ConfigError


@pytest.mark.parametrize("mode", ["joint", "power_only", "split_only", "sdma_power"])
@pytest.mark.parametrize("scalar", ["beta", "squashed_gaussian"])
def test_only_learned_components_are_scored(mode, scalar):
    torch.manual_seed(4)
    model = HybridActorCritic(5, 2, [16], scheduling_enabled=False,
                             allocation_control=mode, scalar_dist=scalar)
    obs = torch.randn(4, 5)
    dist = model.distribution(obs)
    latent = dist.sample(torch.Generator().manual_seed(3))
    mask = latent.active_mask
    expected = torch.zeros(4)
    if mode != "split_only":
        expected = expected + dist._power_dist(mask).log_prob(latent.power_latent)
    if mode not in ("power_only", "sdma_power"):
        expected = expected + dist._rate_dist(mask).log_prob(latent.rate_latent)
        expected = expected + dist._rho_r_dist(mask).log_prob(latent.rho_r_latent)
    torch.testing.assert_close(dist.log_prob(latent), expected)
    action = dist.to_env_action(latent)
    changed = latent.detach()
    if mode in ("power_only", "sdma_power"):
        torch.testing.assert_close(action["rate_fractions"], torch.full((4, 2), 0.5))
        expected_rho = 0.0 if mode == "sdma_power" else 0.3
        torch.testing.assert_close(action["common_payload_scale"], torch.full((4, 1), expected_rho))
        changed.rate_latent = torch.randn_like(changed.rate_latent) * 100
        changed.rho_r_latent = torch.full_like(changed.rho_r_latent, 0.9)
        disabled = ["rate_head", "rho_r_head", "rate_log_std"]
        if mode == "sdma_power":
            torch.testing.assert_close(action["power_fractions"][:, 0], torch.zeros(4))
            torch.testing.assert_close(action["common_payload_scale"], torch.zeros(4, 1))
    elif mode == "split_only":
        torch.testing.assert_close(action["power_fractions"], torch.tensor([[0.25, 0.375, 0.375]]).expand(4, -1))
        changed.power_latent = torch.randn_like(changed.power_latent) * 100
        disabled = ["power_head", "power_log_std"]
    else:
        disabled = []
    torch.testing.assert_close(dist.log_prob(changed), expected)
    for key in action:
        torch.testing.assert_close(dist.to_env_action(changed)[key], action[key])
    (dist.log_prob(latent).mean() + 0.01 * dist.entropy(mask).mean()).backward()
    for name, parameter in model.named_parameters():
        if any(name.startswith(prefix) for prefix in disabled):
            assert parameter.grad is None, name
    ent = dist.head_entropies(mask)
    torch.testing.assert_close(dist.entropy(mask), sum(ent.values()))


@pytest.mark.parametrize("mode", ["power_only", "split_only", "sdma_power"])
def test_training_and_checkpoint_keep_fixed_actions(tmp_path, mode):
    env = EnvConfig(num_users=2, num_antennas=2, scheduling_enabled=False,
                    aoi_cap_enabled=False, episode_length=32)
    cfg = PPOConfig(total_timesteps=128, num_envs=2, rollout_steps=32,
                    minibatch_size=32, update_epochs=2, hidden_sizes=[16],
                    allocation_control=mode, checkpoint_interval=0)
    trainer = PPOTrainer(env, cfg, tmp_path / mode, seed=5)
    try:
        before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
        trainer.collect_rollout()
        trainer.buffer.compute_returns(cfg.gamma, cfg.gae_lambda)
        for batch in trainer.buffer.iter_minibatches(32, np.random.default_rng(1)):
            score = trainer.model.distribution(batch.obs).log_prob(batch.latent)
            torch.testing.assert_close(torch.exp(score - batch.log_probs), torch.ones_like(score), atol=1e-4, rtol=1e-4)
        metrics = trainer.update()
        assert all(np.isfinite(v) for v in metrics.values())
        fixed = ["rate_head", "rho_r_head", "rate_log_std"] if mode in ("power_only", "sdma_power") else ["power_head", "power_log_std"]
        learned = "power_head.weight" if mode in ("power_only", "sdma_power") else "rate_head.weight"
        for n, p in trainer.model.named_parameters():
            if any(n.startswith(prefix) for prefix in fixed):
                torch.testing.assert_close(p, before[n], atol=0, rtol=0)
        assert not torch.equal(dict(trainer.model.named_parameters())[learned], before[learned])
        checkpoint = trainer.save_checkpoint(tmp_path / mode / "test.pt")
        policy, _, loaded = PPOPolicy.from_checkpoint(checkpoint)
        assert loaded.allocation_control == mode
        obs = np.zeros((1, trainer.obs_dim), np.float32)
        original = trainer.make_policy().act(obs)[0]
        restored = policy.act(obs)[0]
        for key in original:
            np.testing.assert_allclose(original[key], restored[key], atol=1e-7)
    finally:
        trainer.close()


@pytest.mark.parametrize("ack_mode", ["sampled", "expected"])
def test_uncapped_age_and_reward_survive_observation_clipping(ack_mode):
    cfg = EnvConfig(num_users=2, num_antennas=2, link_model="fixed", fixed_success_prob=0,
                    aoi_max=2, aoi_cap_enabled=False, episode_length=50,
                    reward_scale=0.5, ack_mode=ack_mode)
    env = AoIRSMAEnv(cfg)
    try:
        env.reset(seed=0)
        action = {"schedule": np.ones(2), "power_fractions": np.array([0., .5, .5]),
                  "rate_fractions": np.array([.5, .5]), "common_payload_scale": np.array([0.])}
        for _ in range(30):
            obs, reward, _, _, info = env.step(action)
        np.testing.assert_allclose(info["aoi"], [31, 31])
        np.testing.assert_allclose(obs[:2], [10, 10])
        assert reward == pytest.approx(-31)
    finally:
        env.close()


def test_invalid_controls_are_rejected():
    with pytest.raises(ConfigError):
        PPOConfig(allocation_control="unknown")
    with pytest.raises(ConfigError):
        PPOConfig(fixed_common_payload_scale=1.1)
    with pytest.raises(ConfigError):
        PPOConfig(initial_common_power_fraction=0.0)
    with pytest.raises(ConfigError):
        PPOConfig(initial_common_payload_scale=1.0)
    with pytest.raises(ConfigError):
        PPOConfig(initial_scalar_concentration=0.0)


@pytest.mark.parametrize("scalar", ["beta", "squashed_gaussian"])
def test_rate_aware_initialisation_sets_requested_modal_action(scalar):
    model = HybridActorCritic(
        5, 2, [16], scheduling_enabled=False, scalar_dist=scalar,
        initial_common_power_fraction=0.2, initial_common_payload_scale=0.05,
        initial_scalar_concentration=20.0,
    )
    action = model.distribution(torch.zeros(3, 5)).to_env_action(
        model.distribution(torch.zeros(3, 5)).mode()
    )
    torch.testing.assert_close(
        action["power_fractions"], torch.tensor([[.2, .4, .4]]).expand(3, -1),
        atol=1e-6, rtol=1e-6,
    )
    torch.testing.assert_close(
        action["common_payload_scale"], torch.full((3, 1), .05),
        atol=1e-6, rtol=1e-6,
    )


def test_common_gate_has_exact_sdma_atom_and_consistent_score():
    torch.manual_seed(8)
    model = HybridActorCritic(
        5, 2, [16], scheduling_enabled=False, common_gate_enabled=True,
        initial_common_on_probability=0.5,
    )
    obs = torch.randn(64, 5)
    dist = model.distribution(obs)
    latent = dist.sample(torch.Generator().manual_seed(9))
    assert latent.common_gate_raw is not None
    assert (latent.common_gate_raw == 0).any() and (latent.common_gate_raw == 1).any()
    action = dist.to_env_action(latent)
    off = latent.common_gate_raw.reshape(-1) == 0
    torch.testing.assert_close(action["power_fractions"][off, 0], torch.zeros(off.sum()))
    torch.testing.assert_close(action["common_payload_scale"][off], torch.zeros(off.sum(), 1))
    torch.testing.assert_close(action["power_fractions"][off, 1:].sum(-1), torch.ones(off.sum()))
    # Continuous split latents are irrelevant at the exact common-off atom.
    changed = latent.detach()
    changed.rate_latent[off] += 100
    changed.rho_r_latent[off] = .99
    changed.power_latent[off, 0] += 100  # masked common-power coordinate when gate is off
    torch.testing.assert_close(dist.log_prob(changed)[off], dist.log_prob(latent)[off])
    torch.testing.assert_close(dist.log_prob(changed)[~off], dist.log_prob(latent)[~off])
    # Entropy is H(gate)+H(power)+P(on)*(H(rate)+H(rho)); breakdown matches it.
    torch.testing.assert_close(dist.entropy(latent.active_mask), sum(dist.head_entropies(latent.active_mask).values()))


def test_common_gate_roundtrips_through_buffer_and_checkpoint(tmp_path):
    env = EnvConfig(num_users=2, num_antennas=2, scheduling_enabled=False,
                    aoi_cap_enabled=False, episode_length=32)
    cfg = PPOConfig(total_timesteps=128, num_envs=2, rollout_steps=32,
                    minibatch_size=32, update_epochs=2, hidden_sizes=[16],
                    common_gate_enabled=True, checkpoint_interval=0)
    trainer = PPOTrainer(env, cfg, tmp_path, seed=14)
    try:
        trainer.collect_rollout()
        assert trainer.buffer.common_gate_raw is not None
        assert set(np.unique(trainer.buffer.common_gate_raw)) <= {0, 1}
        trainer.buffer.compute_returns(cfg.gamma, cfg.gae_lambda)
        for batch in trainer.buffer.iter_minibatches(32, np.random.default_rng(0)):
            score = trainer.model.distribution(batch.obs).log_prob(batch.latent)
            torch.testing.assert_close(torch.exp(score-batch.log_probs), torch.ones_like(score), atol=1e-4, rtol=1e-4)
        trainer.update()
        checkpoint = trainer.save_checkpoint(tmp_path / "gate.pt")
        policy, _, loaded = PPOPolicy.from_checkpoint(checkpoint)
        assert loaded.common_gate_enabled
        action = policy.act(np.zeros((1, trainer.obs_dim), np.float32))[0]
        assert action["common_payload_scale"][0] == 0 or action["power_fractions"][0] > 0
    finally:
        trainer.close()


def test_sdma_power_policy_is_exactly_the_sdma_phy_boundary():
    cfg = EnvConfig(num_users=2, num_antennas=2, scheduling_enabled=False,
                    snr_db=None, p_max=10, payload_bits=100, blocklength=100)
    model = HybridActorCritic(5, 2, [16], scheduling_enabled=False,
                             allocation_control="sdma_power")
    dist = model.distribution(torch.randn(1, 5))
    action = dist.to_env_action(dist.sample(torch.Generator().manual_seed(2)))
    slot = SlotAction(
        active=np.ones(2, bool),
        power_fractions=action["power_fractions"][0].detach().numpy(),
        rate_fractions=action["rate_fractions"][0].detach().numpy(),
        common_payload_scale=float(action["common_payload_scale"][0, 0]),
    )
    h = np.array([[1+0j, .2+.1j], [.3-.1j, 1+0j]])
    rsma = RSMAPhysicalLayer(cfg).transmit(h, h, slot)
    sdma_cfg = replace(cfg, multiple_access="sdma")
    sdma = RSMAPhysicalLayer(sdma_cfg).transmit(h, h, slot)
    assert slot.power_fractions[0] == 0 and slot.common_payload_scale == 0
    np.testing.assert_allclose(rsma.success_prob, sdma.success_prob, rtol=1e-12, atol=1e-12)
    # The SDMA branch renormalises the float32 fractions once more; equality is
    # therefore up to the policy-output rounding level rather than bitwise.
    np.testing.assert_allclose(rsma.private_sinr, sdma.private_sinr, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(rsma.private_power, sdma.private_power, rtol=1e-6, atol=1e-9)
