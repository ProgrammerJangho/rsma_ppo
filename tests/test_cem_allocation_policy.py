import numpy as np

from rsma_aoi.baselines.cem_allocation import CEMAllocationPolicy
from rsma_aoi.envs.aoi_env import AoIRSMAEnv
from rsma_aoi.utils.config import EnvConfig


def _cfg() -> EnvConfig:
    return EnvConfig(
        num_users=2, num_antennas=2, small_scale_model="rician_ula",
        user_angles_deg=(-10.0, 10.0), rician_k_db=8.0,
        csit_error_std=0.2, csit_error_mode="relative", scheduling_enabled=False,
        allocator_mode="none", multiple_access="rsma", blocklength=128,
        payload_bits=256.0, snr_db=10.0,
    )


def test_sdma_cem_emits_exact_common_off_boundary():
    env = AoIRSMAEnv(_cfg())
    obs, _ = env.reset(seed=7)
    policy = CEMAllocationPolicy(_cfg(), "sdma", iterations=2, population=16)
    policy.reset(seed=7)
    action = policy.act(obs[None, :], [env.get_state()])[0]
    assert action["power_fractions"][0] == 0.0
    assert action["common_payload_scale"][0] == 0.0
    assert np.isclose(action["power_fractions"].sum(), 1.0)
    assert policy.candidate_evaluations == 32


def test_safeguarded_cem_uses_two_solves_and_best_estimated_candidate():
    env = AoIRSMAEnv(_cfg())
    obs, _ = env.reset(seed=11)
    policy = CEMAllocationPolicy(
        _cfg(), "rsma_safeguarded", iterations=2, population=16
    )
    policy.reset(seed=11)
    action = policy.act(obs[None, :], [env.get_state()])[0]
    assert policy.candidate_evaluations == 64
    rsma = policy.last_selection["rsma_estimated_gain"]
    sdma = policy.last_selection["sdma_estimated_gain"]
    expected = "sdma" if sdma > rsma else "rsma"
    assert policy.last_selection["selection"] == expected
    if expected == "sdma":
        assert action["power_fractions"][0] == 0.0
        assert action["common_payload_scale"][0] == 0.0


def test_cem_synchronises_to_episode_randomized_total_power():
    cfg = _cfg()
    env_cfg = EnvConfig(**{
        **cfg.to_dict(), "snr_db_values": [5.0, 15.0], "obs_include_snr": True,
    })
    env = AoIRSMAEnv(env_cfg)
    obs, _ = env.reset(seed=3)
    policy = CEMAllocationPolicy(env_cfg, "sdma", iterations=1, population=8)
    policy.act(obs[None, :], [env.get_state()])
    assert policy.rsma_phy.total_power == env.total_power
    assert policy.sdma_phy.total_power == env.total_power
