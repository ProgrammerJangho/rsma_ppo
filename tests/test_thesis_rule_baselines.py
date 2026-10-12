import numpy as np

from rsma_aoi.baselines.thesis_rules import (
    MaxAoISingleUserPolicy, OldestCommonFixedAlphaPolicy,
)


def _state(aoi):
    return {"aoi": np.asarray(aoi, float), "weights": np.ones(3)}


def test_max_aoi_single_user_is_exact_private_mrt_action():
    action = MaxAoISingleUserPolicy(3).act(np.zeros((1, 1)), [_state([2, 7, 4])])[0]
    np.testing.assert_array_equal(action["schedule"], [0, 1, 0])
    np.testing.assert_array_equal(action["common_power_fraction"], [0.0])
    np.testing.assert_array_equal(action["private_power_fractions"], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(action["common_payload_fractions"], [0.0, 0.0, 0.0])


def test_oldest_common_rule_does_not_waste_private_power_on_common_user():
    action = OldestCommonFixedAlphaPolicy(3, 0.65).act(
        np.zeros((1, 1)), [_state([2, 7, 4])]
    )[0]
    np.testing.assert_array_equal(action["schedule"], [1, 1, 1])
    np.testing.assert_allclose(action["common_power_fraction"], [0.65])
    np.testing.assert_allclose(action["private_power_fractions"], [0.5, 0.0, 0.5])
    np.testing.assert_array_equal(action["common_payload_fractions"], [0.0, 1.0, 0.0])
