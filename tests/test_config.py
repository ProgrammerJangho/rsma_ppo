"""Configuration loading, validation and derived quantities."""

from __future__ import annotations

from pathlib import Path

import pytest

from rsma_aoi.utils.config import (
    ConfigError, EnvConfig, PPOConfig, dump_config_snapshot, load_run_configs, load_yaml,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize(
    "name",
    ["env_default.yaml", "m0_debug.yaml", "m1_rsma.yaml", "m2_hybrid.yaml",
     "m2_continuous.yaml", "thesis_rician_alignment_5deg.yaml",
     "thesis_rician_alignment_20deg.yaml", "thesis_rician_alignment_60deg.yaml",
     "thesis_rician_nearfar5_csit02.yaml", "thesis_final_evaluation.yaml",
     "thesis_spatial_5deg_evaluation.yaml", "thesis_spatial_20deg_evaluation.yaml",
     "thesis_spatial_60deg_evaluation.yaml", "thesis_nearfar5_evaluation.yaml",
     "thesis_rician_mixed_geometry_csit02.yaml",
     "thesis_rician_mixed_geometry_snr_csit02.yaml"],
)
def test_shipped_env_configs_load_and_validate(name):
    cfg = EnvConfig.from_yaml(CONFIG_DIR / name)
    cfg.validate()
    assert cfg.num_users >= 1 and cfg.num_antennas >= 1
    assert cfg.total_power > 0.0
    assert len(cfg.payload_vector) == cfg.num_users
    assert len(cfg.weight_vector) == cfg.num_users


def test_shipped_ppo_config_loads():
    cfg = PPOConfig.from_yaml(CONFIG_DIR / "ppo_default.yaml")
    assert cfg.batch_size % cfg.minibatch_size == 0


def test_load_run_configs_merges_the_scenario_ppo_section():
    env_cfg, ppo_cfg = load_run_configs(
        CONFIG_DIR / "m0_debug.yaml", CONFIG_DIR / "ppo_default.yaml"
    )
    assert env_cfg.link_model == "fixed"
    # the scenario file overrides the defaults
    assert ppo_cfg.total_timesteps == 8192
    assert ppo_cfg.num_envs == 4
    # keys only present in the defaults survive
    assert ppo_cfg.gamma == 0.99


def test_milestone_configs_encode_their_milestone():
    m1 = EnvConfig.from_yaml(CONFIG_DIR / "m1_rsma.yaml")
    assert m1.scheduling_enabled is False, "M1 is continuous-allocation only"
    assert m1.link_model == "fbl"
    m2 = EnvConfig.from_yaml(CONFIG_DIR / "m2_hybrid.yaml")
    assert m2.scheduling_enabled is True, "M2 uses the hybrid action space"
    assert m2.num_users > m2.num_antennas, "M2 is deliberately overloaded"


def test_continuous_ablation_differs_only_in_the_scheduling_flag():
    """The hybrid-vs-continuous ablation must be apples-to-apples."""
    hybrid = EnvConfig.from_yaml(CONFIG_DIR / "m2_hybrid.yaml").to_dict()
    continuous = EnvConfig.from_yaml(CONFIG_DIR / "m2_continuous.yaml").to_dict()
    differences = {k for k in hybrid if hybrid[k] != continuous[k]}
    assert differences == {"scheduling_enabled"}
    assert continuous["scheduling_enabled"] is False


def test_snr_db_defines_the_total_power():
    assert EnvConfig(snr_db=10.0, noise_power=1.0).total_power == pytest.approx(10.0)
    assert EnvConfig(snr_db=0.0, noise_power=2.0).total_power == pytest.approx(2.0)
    assert EnvConfig(snr_db=None, p_max=7.0).total_power == pytest.approx(7.0)


def test_random_snr_requires_an_observed_finite_nonempty_set():
    cfg = EnvConfig(snr_db_values=[5.0, 10.0, 15.0], obs_include_snr=True)
    assert cfg.snr_db_values == [5.0, 10.0, 15.0]
    for values, observed in (([], True), ([5.0, float("nan")], True), ([5.0, 5.0], True), ([5.0], False)):
        with pytest.raises(ConfigError):
            EnvConfig(snr_db_values=values, obs_include_snr=observed)


def test_default_reward_scale_normalises_the_objective():
    cfg = EnvConfig(num_users=4, aoi_max=50)
    assert cfg.effective_reward_scale == pytest.approx(1.0 / 200.0)
    assert EnvConfig(reward_scale=0.5).effective_reward_scale == 0.5


def test_path_gain_conversion():
    cfg = EnvConfig(num_users=2, path_gains_db=[0.0, -10.0])
    assert cfg.path_gain_linear == pytest.approx([1.0, 0.1])


def test_unknown_keys_are_rejected():
    with pytest.raises(ConfigError, match="unknown keys"):
        EnvConfig.from_dict({"num_users": 2, "typo_key": 1})
    with pytest.raises(ConfigError, match="unknown keys"):
        PPOConfig.from_dict({"gamma": 0.9, "not_a_key": 1})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_users": 0},
        {"num_antennas": 0},
        {"alpha": 1.5},
        {"csit_error_std": -0.1},
        {"noise_power": 0.0},
        {"snr_db": None, "p_max": None},
        {"blocklength": 0},
        {"blocklength_values": []},
        {"blocklength_values": [64, 0], "obs_include_blocklength": True},
        {"blocklength_values": [64, 64], "obs_include_blocklength": True},
        {"blocklength_values": [64, 128]},
        {"private_beam": "bogus"},
        {"common_beam": "bogus"},
        {"small_scale_model": "bogus"},
        {"small_scale_model": "rician_ula"},
        {"small_scale_model": "rician_ula", "user_angles_deg": [0.0]},
        {"small_scale_model": "rician_ula", "user_angles_deg": [0.0, 91.0]},
        {"small_scale_model": "rician_ula", "user_angles_deg": [0.0, 20.0],
         "csit_error_mode": "absolute"},
        {"user_angle_sets_deg": [[-2.5, 2.5]]},
        {"small_scale_model": "rician_ula", "user_angles_deg": [-10.0, 10.0],
         "user_angle_sets_deg": []},
        {"small_scale_model": "rician_ula", "user_angles_deg": [-10.0, 10.0],
         "user_angle_sets_deg": [[-2.5]]},
        {"small_scale_model": "rician_ula", "user_angles_deg": [-10.0, 10.0],
         "user_angle_sets_deg": [[-2.5, 95.0]]},
        {"rician_k_db": float("nan")},
        {"rician_k_db": 101.0},
        {"antenna_spacing_wavelength": 0.0},
        {"antenna_spacing_wavelength": 0.6},
        {"multiple_access": "bogus"},
        {"link_model": "bogus"},
        {"ack_mode": "bogus"},
        {"obs_channel_mode": "bogus"},
        {"aoi_max": 0},
        {"episode_length": 0},
        {"payload_bits": [1.0]},
        {"payload_bits": -5.0},
        {"aoi_weights": [1.0]},
        {"aoi_weights": [-1.0, 1.0]},
        {"path_gains_db": [0.0]},
        {"lambda_power": -1.0},
        {"fixed_success_prob": 2.0},
        {"rzf_regularization": -1.0},
        {"common_beam_iters": -1},
        {"common_beam_damping": 0.0},
        {"common_beam_damping": 1.1},
        {"feasibility_tol": 0.0},
    ],
)
def test_invalid_env_configs_raise(kwargs):
    with pytest.raises(ConfigError):
        EnvConfig(**{"num_users": 2, **kwargs})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_envs": 0},
        {"rollout_steps": 0},
        {"minibatch_size": 7},   # does not divide the batch
        {"gamma": 1.5},
        {"gae_lambda": 2.0},
        {"clip_coef": 0.0},
        {"learning_rate": 0.0},
        {"max_grad_norm": 0.0},
        {"target_kl": -1.0},
        {"scalar_dist": "bogus"},
        {"activation": "bogus"},
        {"hidden_sizes": []},
    ],
)
def test_invalid_ppo_configs_raise(kwargs):
    with pytest.raises(ConfigError):
        PPOConfig(**{"num_envs": 2, "rollout_steps": 8, "minibatch_size": 4, **kwargs})


def test_missing_file_raises():
    with pytest.raises(ConfigError, match="not found"):
        load_yaml("does/not/exist.yaml")


def test_snapshot_roundtrip(tmp_path):
    cfg = EnvConfig(num_users=3)
    path = tmp_path / "snap.yaml"
    dump_config_snapshot(path, env=cfg, experiment={"seeds": [0, 1]})
    data = load_yaml(path)
    assert data["experiment"]["seeds"] == [0, 1]
    assert EnvConfig(**data["env"]).num_users == 3


def test_offpolicy_factory_reports_implementation_status(tmp_path):
    """Implemented algorithms build; unimplemented ones fail loudly."""
    from rsma_aoi.agents.offpolicy import (
        IMPLEMENTED_ALGORITHMS, SUPPORTED_ALGORITHMS, build_offpolicy_agent,
    )
    from rsma_aoi.agents.td3 import TD3Trainer
    from rsma_aoi.utils.config import TD3Config

    # TD3 is a legacy simplex+rho comparator; the thesis PPO default is the
    # gate-free direct-fraction action.
    cfg = EnvConfig(
        num_users=2, num_antennas=2, scheduling_enabled=False, episode_length=8,
        rate_split_mode="simplex_scaled",
    )
    td3_cfg = TD3Config(num_envs=2, buffer_size=256, batch_size=32, learning_starts=0,
                        hidden_sizes=[16, 16])
    trainer = build_offpolicy_agent("td3", cfg, td3_cfg, run_dir=tmp_path / "td3", seed=0)
    assert isinstance(trainer, TD3Trainer)
    trainer.close()

    for algorithm in set(SUPPORTED_ALGORITHMS) - set(IMPLEMENTED_ALGORITHMS):
        with pytest.raises(NotImplementedError):
            build_offpolicy_agent(algorithm, cfg)
    with pytest.raises(ValueError):
        build_offpolicy_agent("ppo", cfg)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_envs": 0},
        {"buffer_size": 0},
        {"gamma": 1.5},
        {"tau": 0.0},
        {"policy_delay": 0},
        {"exploration_noise": -0.1},
        {"actor_lr": 0.0},
        {"latent_scale": 0.0},
        {"activation": "bogus"},
        {"hidden_sizes": []},
        {"batch_size": 10_000},
        {"learning_starts": -1},
    ],
)
def test_invalid_td3_configs_raise(kwargs):
    from rsma_aoi.utils.config import TD3Config

    with pytest.raises(ConfigError):
        TD3Config(**{"buffer_size": 1000, "batch_size": 32, **kwargs})


def test_td3_config_yaml_loads():
    from rsma_aoi.utils.config import TD3Config, load_td3_configs

    cfg = TD3Config.from_yaml(CONFIG_DIR / "td3_default.yaml")
    assert cfg.policy_delay == 2 and cfg.buffer_size > 0
    env_cfg, td3_cfg = load_td3_configs(
        CONFIG_DIR / "m2_continuous.yaml", CONFIG_DIR / "td3_default.yaml"
    )
    assert env_cfg.scheduling_enabled is False
    assert td3_cfg.total_timesteps == 400000
