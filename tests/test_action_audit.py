import numpy as np

from scripts.audit_action_state_dependence import swap_users


def test_swap_users_gain_features_preserves_global_tail_and_remaps_pairs():
    # K=2 layout: AoI | gains | corr(01) | ACK |
    # prev common/private power | prev schedule | global SNR.
    obs = np.arange(1, 15, dtype=np.float32)[None, :]
    swapped = swap_users(
        obs, num_users=2, num_antennas=2,
        include_prev_action=True, channel_mode="gain_features",
    )
    expected = np.asarray([[
        2, 1,                  # AoI
        4, 3,                  # gains
        5,                     # correlation (01)
        7, 6,                  # previous ACK
        8, 10, 9,              # common, private 1, private 0
        12, 11,                # previous schedule
        13, 14,                # arbitrary trailing global features
    ]], dtype=np.float32)
    np.testing.assert_array_equal(swapped, expected)
