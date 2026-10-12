import numpy as np

from rsma_aoi.baselines.cem_scheduling import ScheduledCEMAllocationPolicy
from rsma_aoi.envs.aoi_env import AoIRSMAEnv
from rsma_aoi.utils.config import EnvConfig


def test_scheduled_cem_supports_overloaded_k3_nt2() -> None:
    cfg = EnvConfig(
        num_users=3, num_antennas=2, small_scale_model="rician_ula",
        user_angles_deg=[-10.0, 0.0, 10.0], csit_error_std=0.2,
        scheduling_enabled=True, multiple_access="rsma", episode_length=8,
        payload_bits=256.0, aoi_weights=[1.0, 1.0, 1.0],
    )
    env = AoIRSMAEnv(cfg)
    obs, _ = env.reset(seed=123)
    policy = ScheduledCEMAllocationPolicy(
        cfg, "rsma_safeguarded", schedule_size=2, iterations=2, population=16
    )
    action = policy.act(obs[None, :], [env.get_state()])[0]
    assert int(np.asarray(action["schedule"]).sum()) == 2
    assert np.isclose(np.asarray(action["power_fractions"]).sum(), 1.0)
    _, _, terminated, truncated, info = env.step(action)
    assert not terminated and not truncated
    assert info["constraint_violation"] <= cfg.feasibility_tol

