"""Direct allocation physics, PPO replay, gradients, and boundary regressions."""
from dataclasses import replace
import numpy as np
import torch
import pytest
from rsma_aoi.agents.ppo import PPOTrainer, PPOPolicy
from rsma_aoi.envs.aoi_env import AoIRSMAEnv
from rsma_aoi.envs.rsma_phy import RSMAPhysicalLayer, split_payload_batch
from rsma_aoi.envs.torch_phy import rsma_success_probability
from rsma_aoi.utils.config import EnvConfig, PPOConfig
from rsma_aoi.optimizers import CEMAllocator, make_problem
from rsma_aoi.baselines.cem_allocation import CEMAllocationPolicy
from rsma_aoi.evaluation.oracle import decode_latents
from rsma_aoi.utils.action_mapping import HARD_SIGMOID_LIMIT

def config():
    return EnvConfig(num_users=2, num_antennas=2, scheduling_enabled=False,
        rate_split_mode="per_user_fraction", blocklength=128, payload_bits=[128.,256.],
        episode_length=16)

def test_physical_payload_and_zero_boundary():
    cfg = config()
    env = AoIRSMAEnv(cfg)
    env.reset(seed=12)
    action = dict(
        schedule=np.ones(2), common_power_fraction=np.array([.25]),
        private_power_fractions=np.array([.4,.6]),
        common_payload_fractions=np.array([.2,.9]),
    )
    parsed, violation = env._parse_action(action)
    assert violation == 0
    np.testing.assert_array_equal(parsed.rate_fractions, [.2,.9])
    phy = RSMAPhysicalLayer(cfg)
    out = phy.transmit(env._h_true, env._h_hat, parsed)
    np.testing.assert_allclose(out.private_payload, np.array(cfg.payload_vector)*[.8,.1])
    assert out.common_payload == pytest.approx(128*.2+256*.9)
    assert out.constraint_violation == 0
    parsed.power_fractions = np.array([0.,.4,.6])
    parsed.rate_fractions = np.zeros(2)
    rs = phy.transmit(env._h_true, env._h_hat, parsed)
    sd = RSMAPhysicalLayer(replace(cfg, multiple_access="sdma")).transmit(env._h_true, env._h_hat, parsed)
    np.testing.assert_allclose(rs.success_prob, sd.success_prob, atol=1e-12)
    assert np.all(rs.common_bler == 0)


def test_direct_public_action_space_passes_gym_checker():
    from gymnasium.utils.env_checker import check_env

    env = AoIRSMAEnv(config())
    assert set(env.action_space.spaces) == {
        'schedule', 'common_power_fraction', 'private_power_fractions',
        'common_payload_fractions',
    }
    check_env(env, skip_render_check=True)

def test_torch_matches_numpy_and_gradients():
    cfg = config(); phy = RSMAPhysicalLayer(cfg)
    h = np.array([[1, .5j], [.9, .3+.2j]], complex)
    active = np.ones(2, bool); beams = phy.rsma_beams(h, active)
    pf, rf, rho = decode_latents(np.random.default_rng(2).normal(size=(10,5)), active, 2, cfg.rate_split_mode)
    npout = phy.evaluate_allocations(h, active, pf, rf, rho, beams=beams)
    gain = np.abs(h.conj() @ beams[0].T)**2
    common = np.abs(h.conj() @ beams[1])**2
    tensor = lambda a: torch.tensor(a, dtype=torch.float64)
    pt, rt = tensor(pf).requires_grad_(), tensor(rf).requires_grad_()
    out = rsma_success_probability(tensor(gain), tensor(common), torch.ones(10,2,dtype=torch.bool),
        pt, rt, tensor(rho), tensor(cfg.payload_vector), cfg.blocklength,
        cfg.noise_power, cfg.total_power, rate_split_mode=cfg.rate_split_mode)
    for key in ['success_prob','common_payload','private_payload','common_bler','private_bler']:
        np.testing.assert_allclose(out[key].detach(), npout[key], atol=1e-10)
    out['success_prob'].sum().backward()
    assert torch.isfinite(pt.grad).all() and torch.isfinite(rt.grad).all()


def test_direct_latent_mapping_reaches_exact_sdma_without_gate():
    """The continuous heads themselves contain the exact common-off boundary."""
    active = np.ones(2, dtype=bool)
    z = np.array([[-HARD_SIGMOID_LIMIT - 1.0, 0.2, -0.2,
                   -HARD_SIGMOID_LIMIT - 1.0, -HARD_SIGMOID_LIMIT - 2.0]])
    power, phi, rho = decode_latents(z, active, 2, "per_user_fraction")
    assert power[0, 0] == 0.0
    assert power[0, 1:].sum() == pytest.approx(1.0)
    np.testing.assert_array_equal(phi, np.zeros((1, 2)))
    # rho is a compatibility placeholder and is not part of the direct action.
    np.testing.assert_array_equal(rho, np.ones(1))


def test_direct_q_only_policy_forces_common_off_and_learns_private_simplex():
    """The thesis action supports a matched all-user q-only PPO-SDMA class."""
    from rsma_aoi.agents.networks import HybridActorCritic

    model = HybridActorCritic(
        obs_dim=7, num_users=3, hidden_sizes=[16], scheduling_enabled=False,
        rate_split_mode="per_user_fraction", allocation_control="sdma_power",
    )
    obs = torch.randn(5, 7)
    dist = model.distribution(obs)
    latent = dist.sample(torch.Generator().manual_seed(21))
    action = dist.to_env_action(latent)
    torch.testing.assert_close(action["common_power_fraction"], torch.zeros(5, 1))
    torch.testing.assert_close(action["common_payload_fractions"], torch.zeros(5, 3))
    torch.testing.assert_close(
        action["private_power_fractions"].sum(-1), torch.ones(5)
    )
    assert torch.all(action["private_power_fractions"] > 0)

    loss = -dist.log_prob(latent).mean()
    loss.backward()
    assert model.power_head.weight.grad is not None
    assert torch.isfinite(model.power_head.weight.grad).all()
    assert model.rate_head.weight.grad is None


def test_direct_q_only_objective_pretraining_is_exact_common_off():
    """Differentiable warm start optimises q with alpha=phi=0."""
    from rsma_aoi.agents.allocator_net import (
        LOSS_PRESETS, hybrid_actor_allocations, precompute_batch,
        sdma_actor_objective_losses,
    )
    from rsma_aoi.agents.networks import HybridActorCritic

    cfg = EnvConfig(
        num_users=3, num_antennas=2, scheduling_enabled=False,
        rate_split_mode="per_user_fraction", blocklength=128,
        payload_bits=128.0, episode_length=8,
    )
    env = AoIRSMAEnv(cfg)
    obs, _ = env.reset(seed=8)
    active = np.ones((1, 3), bool)
    batch = precompute_batch(
        cfg, obs[None], env._aoi[None], env._h_true[None], env._h_hat[None],
        active, np.array([[0.0, 0.2, 0.3, 0.5]]),
        np.array([[0.1, 0.2, 0.3]]), np.ones(1),
    )
    model = HybridActorCritic(
        obs_dim=obs.size, num_users=3, hidden_sizes=[16],
        scheduling_enabled=False, rate_split_mode="per_user_fraction",
        allocation_control="sdma_power",
    )
    power, phi, _, _ = hybrid_actor_allocations(
        model, batch.features, batch.active
    )
    torch.testing.assert_close(power[:, :1], torch.zeros(1, 1))
    torch.testing.assert_close(phi, torch.zeros(1, 3))
    loss, metrics = sdma_actor_objective_losses(
        model, batch, cfg, LOSS_PRESETS["objective_only"]
    )
    loss.backward()
    assert metrics["common_power"] == 0
    assert model.power_head.weight.grad is not None
    assert torch.isfinite(model.power_head.weight.grad).all()
    assert model.rate_head.weight.grad is None
    env.close()


def test_actor_exact_common_off_matches_sdma_end_to_end():
    """A direct PPO actor reaches the physical SDMA boundary without a gate."""
    from rsma_aoi.agents.networks import HybridActorCritic

    cfg = config()
    env = AoIRSMAEnv(cfg)
    obs, _ = env.reset(seed=41)
    model = HybridActorCritic(
        obs_dim=obs.size, num_users=2, hidden_sizes=[16],
        scheduling_enabled=False, rate_split_mode="per_user_fraction",
    )
    with torch.no_grad():
        model.power_head.weight.zero_()
        model.power_head.bias.zero_()
        model.power_head.bias[0] = -HARD_SIGMOID_LIMIT - 1.0
        model.rate_head.weight.zero_()
        model.rate_head.bias.fill_(-HARD_SIGMOID_LIMIT - 1.0)
    dist = model.distribution(torch.as_tensor(obs[None], dtype=torch.float32))
    action_t = dist.to_env_action(dist.mode())
    action = {
        "schedule": action_t["schedule"][0].detach().numpy(),
        "power_fractions": action_t["power_fractions"][0].detach().numpy(),
        "rate_fractions": action_t["rate_fractions"][0].detach().numpy(),
    }
    parsed, violation = env._parse_action(action)
    assert violation == 0.0
    assert parsed.power_fractions[0] == 0.0
    np.testing.assert_array_equal(parsed.rate_fractions, np.zeros(2))
    rsma = RSMAPhysicalLayer(cfg).transmit(env._h_true, env._h_hat, parsed)
    sdma = RSMAPhysicalLayer(replace(cfg, multiple_access="sdma")).transmit(
        env._h_true, env._h_hat, parsed
    )
    np.testing.assert_allclose(rsma.success_prob, sdma.success_prob, atol=1e-12)
    assert rsma.common_payload == 0.0
    assert rsma.common_power == 0.0

def test_ppo_replay_update_checkpoint(tmp_path):
    cfg = config()
    pc = PPOConfig(total_timesteps=64, num_envs=2, rollout_steps=16, minibatch_size=16,
        update_epochs=1, hidden_sizes=[16,16], device="cpu", checkpoint_interval=0)
    trainer = PPOTrainer(cfg, pc, tmp_path/'run', seed=4)
    assert trainer.model.rho_r_head is None and trainer.model.common_gate_head is None
    trainer.collect_rollout()
    trainer.buffer.compute_returns(pc.gamma, pc.gae_lambda)
    for batch in trainer.buffer.iter_minibatches(16, np.random.default_rng(0)):
        dist = trainer.model.distribution(batch.obs)
        torch.testing.assert_close(dist.log_prob(batch.latent), batch.log_probs, atol=1e-5, rtol=1e-5)
        before = dist.log_prob(batch.latent).detach().clone()
        batch.latent.rho_r_latent.fill_(999)
        torch.testing.assert_close(dist.log_prob(batch.latent), before)
        action = dist.to_env_action(batch.latent)
        assert 'common_payload_scale' not in action
        assert batch.latent.common_gate_raw is None
        torch.testing.assert_close(action['power_fractions'].sum(-1), torch.ones(16))
    assert all(np.isfinite(v) for v in trainer.update().values())
    path = trainer.save_checkpoint(tmp_path/'run'/'checkpoint.pt')
    policy, ec, _ = PPOPolicy.from_checkpoint(path)
    assert ec.rate_split_mode == 'per_user_fraction'
    public = policy.act(np.zeros((1,trainer.obs_dim),np.float32))[0]
    assert set(public) == {
        'schedule', 'common_power_fraction', 'private_power_fractions',
        'common_payload_fractions',
    }
    assert public['private_power_fractions'].sum() == pytest.approx(1.0)
    trainer.close()

def test_cem_and_sdma_safeguard():
    cfg = config(); env = AoIRSMAEnv(cfg); obs,_ = env.reset(seed=5)
    for mode in ['rsma_raw','sdma','rsma_safeguarded']:
        policy = CEMAllocationPolicy(cfg, mode, iterations=2, population=16)
        policy.reset(2)
        action = policy.act(obs[None], [env.get_state()])[0]
        parsed, v = env._parse_action(action)
        assert v == 0
        if mode == 'sdma':
            assert parsed.power_fractions[0] == 0
            assert np.all(parsed.rate_fractions == 0)

def test_objective_pretraining_direct():
    from rsma_aoi.agents.allocator_net import precompute_batch, hybrid_actor_objective_losses, LOSS_PRESETS
    from rsma_aoi.agents.networks import HybridActorCritic
    cfg = config(); env=AoIRSMAEnv(cfg); obs,_=env.reset(seed=3)
    batch = precompute_batch(cfg, obs[None], env._aoi[None], env._h_true[None],env._h_hat[None],
        np.ones((1,2),bool), np.array([[.3,.3,.4]]), np.array([[.2,.8]]), np.ones(1))
    model=HybridActorCritic(len(obs),2,[16],scheduling_enabled=False,rate_split_mode=cfg.rate_split_mode)
    loss, metrics=hybrid_actor_objective_losses(model,batch,cfg,LOSS_PRESETS['objective_only'])
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.rate_head.weight.grad).all()
    assert 'rho' not in metrics


def test_objective_pretraining_mapping_matches_ppo_modal_action():
    """Warm-start and deployed actor must decode identical direct latents."""
    from rsma_aoi.agents.allocator_net import hybrid_actor_allocations
    from rsma_aoi.agents.networks import HybridActorCritic

    torch.manual_seed(17)
    model = HybridActorCritic(
        obs_dim=9, num_users=3, hidden_sizes=[16, 16],
        scheduling_enabled=False, conditional_allocation=True,
        rate_split_mode="per_user_fraction",
    )
    obs = torch.randn(7, 9)
    active = torch.ones(7, 3, dtype=torch.bool)
    warm_power, warm_phi, warm_rho, warm_gate = hybrid_actor_allocations(
        model, obs, active
    )
    dist = model.distribution(obs, forced_mask=active)
    deployed = dist.to_env_action(dist.mode())

    torch.testing.assert_close(warm_power, deployed["power_fractions"])
    torch.testing.assert_close(warm_phi, deployed["rate_fractions"])
    torch.testing.assert_close(warm_rho, torch.ones_like(warm_rho))
    torch.testing.assert_close(warm_gate, torch.ones_like(warm_gate))

def test_near_zero_fbl_is_not_silently_made_sdma():
    cfg=config(); phy=RSMAPhysicalLayer(cfg)
    h=np.array([[1.,0.],[.5,np.sqrt(.75)]],complex)
    active=np.ones(2,bool); beams=phy.rsma_beams(h,active)
    eps=1e-8
    out=phy.evaluate_allocations(h,active,[[eps,(1-eps)/2,(1-eps)/2]],[[eps**2]*2],[1],beams=beams)
    assert np.all((out['common_bler']>.49)&(out['common_bler']<.51))
    assert out['common_payload'][0]>0

def test_zero_private_payload_and_user_local_decoding():
    cfg=config(); phy=RSMAPhysicalLayer(cfg)
    h=np.array([[1.,0.],[.5,np.sqrt(.75)]],complex)
    active=np.ones(2,bool); beams=phy.rsma_beams(h,active)
    out=phy.evaluate_allocations(h,active,[[.8,.1,.1]],[[1.,.5]],[1],beams=beams)
    assert out['private_bler'][0,0]==0
    np.testing.assert_allclose(out['success_prob'],(1-out['common_bler'])*(1-out['private_bler']))

def test_incompatible_pretrained_actor_rejected(tmp_path):
    from rsma_aoi.agents.networks import HybridActorCritic
    from rsma_aoi.agents.imitation import load_pretrained_actor
    old=HybridActorCritic(4,2,[8]); new=HybridActorCritic(4,2,[8],rate_split_mode='per_user_fraction')
    path=tmp_path/'old.pt'
    torch.save({'model':old.state_dict(),'env_config':{'rate_split_mode':'simplex_scaled'}},path)
    with pytest.raises(ValueError,match='parameterization differs'):
        load_pretrained_actor(new,path,strict=False)

def test_direct_beam_audit_boundary():
    from rsma_aoi.evaluation.beam_audit import audit_state_allocations
    cfg=config(); env=AoIRSMAEnv(cfg); env.reset(seed=1)
    rows=audit_state_allocations(cfg,env._aoi,env._h_true,env._h_hat,np.ones(2,bool),
        beam_modes=(cfg.common_beam,),csit_mode='estimated',seed=0,iterations=2,population=16,restarts=1)
    assert len(rows)==1
