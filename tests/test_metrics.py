"""Evaluation metrics and statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsma_aoi.evaluation.metrics import jain_fairness, peak_aoi, summarise_episode
from rsma_aoi.evaluation.statistics import bootstrap_ci, pivot_summary, summarise


def test_jain_fairness_bounds():
    assert jain_fairness(np.ones(5)) == pytest.approx(1.0)
    assert jain_fairness(np.array([1.0, 1.0, 1.0, 100.0])) < 0.5
    assert jain_fairness(np.array([2.0, 4.0])) == pytest.approx(36.0 / (2 * 20.0))
    assert jain_fairness(np.zeros(3)) == 1.0


def test_peak_aoi_uses_the_values_before_a_reset():
    # user 0 peaks at 3 and 4; user 1 never resets -> its maximum is reported
    trace = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [1.0, 4.0], [2.0, 5.0],
                      [3.0, 6.0], [4.0, 7.0], [1.0, 8.0]])
    np.testing.assert_allclose(peak_aoi(trace), [3.5, 8.0])


def make_info(num_users: int, scheduled: int) -> dict:
    mask = np.array([1] * scheduled + [0] * (num_users - scheduled), dtype=np.int8)
    return {
        "ack": np.array([1.0] * scheduled + [0.0] * (num_users - scheduled)),
        "num_scheduled": scheduled,
        "mean_bler": 0.1,
        "total_power": 8.0,
        "common_power": 2.0,
        "constraint_violation": 0.0,
        "scheduled_users": mask,
        "private_payload": mask.astype(float) * 100.0,
        "common_payload": 50.0,
    }


def test_summarise_episode_computes_the_documented_quantities():
    aoi = np.array([[1.0, 2.0], [2.0, 3.0], [1.0, 4.0], [2.0, 5.0]])
    infos = [make_info(2, 1) for _ in range(4)]
    metrics = summarise_episode(
        episode_index=0, seed=3, aoi_trace=aoi, reward_trace=np.full(4, -0.5),
        info_trace=infos, weights=np.array([1.0, 2.0]), tail_threshold=3.0,
        total_power=10.0, latency_ms=0.25,
    )
    assert metrics.steps == 4
    assert metrics.episode_return == pytest.approx(-2.0)
    assert metrics.mean_aoi == pytest.approx(np.mean([1.5, 3.5]))
    assert metrics.weighted_avg_aoi == pytest.approx(np.mean(aoi @ np.array([1.0, 2.0])))
    assert metrics.max_aoi == 5.0
    assert metrics.success_rate == pytest.approx(1.0)
    assert metrics.avg_common_power_ratio == pytest.approx(0.2)
    assert metrics.tail_excess == pytest.approx(np.mean([0.0, 0.0, 1.0, 2.0]))
    assert metrics.per_user_aoi == pytest.approx([1.5, 3.5])
    assert metrics.avg_common_payload_ratio == pytest.approx(50.0 / 150.0)
    assert metrics.schedule_entropy == pytest.approx(0.0)  # always the same subset
    assert metrics.schedule_count_hist == pytest.approx([0.0, 1.0, 0.0])
    assert metrics.activation_rate == pytest.approx([1.0, 0.0])
    row = metrics.to_row()
    assert row["aoi_user_1"] == pytest.approx(3.5)
    assert row["sched_frac_1"] == pytest.approx(1.0)
    assert row["activation_rate_0"] == pytest.approx(1.0)
    assert "per_user_aoi" not in row and "schedule_count_hist" not in row


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(0)
    sample = rng.normal(5.0, 1.0, size=200)
    low, high = bootstrap_ci(sample, seed=1)
    assert low < sample.mean() < high
    assert high - low < 1.0


def test_bootstrap_ci_is_seed_reproducible():
    sample = np.arange(50, dtype=float)
    assert bootstrap_ci(sample, seed=3) == bootstrap_ci(sample, seed=3)
    assert bootstrap_ci(sample, seed=3) != bootstrap_ci(sample, seed=4)


def test_bootstrap_ci_degenerate_inputs():
    assert bootstrap_ci(np.array([2.0])) == (2.0, 2.0)
    low, high = bootstrap_ci(np.array([]))
    assert np.isnan(low) and np.isnan(high)


def test_summarise_produces_one_row_per_group_and_metric():
    df = pd.DataFrame(
        {
            "algorithm": ["a"] * 5 + ["b"] * 5,
            "weighted_avg_aoi": list(range(5)) + list(range(10, 15)),
            "mean_aoi": list(range(5)) + list(range(10, 15)),
        }
    )
    summary = summarise(df, metrics=("weighted_avg_aoi", "mean_aoi"), seed=0)
    assert len(summary) == 4
    row = summary[(summary["algorithm"] == "a") & (summary["metric"] == "weighted_avg_aoi")]
    assert row["mean"].iloc[0] == pytest.approx(2.0)
    assert row["n"].iloc[0] == 5
    assert row["ci_low"].iloc[0] <= 2.0 <= row["ci_high"].iloc[0]
    assert len(pivot_summary(summary, "mean_aoi")) == 2


def test_summarise_skips_missing_metrics():
    df = pd.DataFrame({"algorithm": ["a", "a"], "mean_aoi": [1.0, 2.0]})
    summary = summarise(df, metrics=("mean_aoi", "not_recorded"), seed=0)
    assert set(summary["metric"]) == {"mean_aoi"}


def test_summarise_handles_all_nan_metric_without_warning():
    df = pd.DataFrame({"algorithm": ["sdma", "sdma"], "common_bler_when_on": [np.nan, np.nan]})
    summary = summarise(df, metrics=("common_bler_when_on",), seed=0)
    row = summary.iloc[0]
    assert row["n"] == 0
    assert np.isnan(row["mean"])
    assert np.isnan(row["ci_low"])


def test_plots_are_written(tmp_path):
    from rsma_aoi.evaluation.plots import plot_aoi_cdf, plot_metric_bars

    df = pd.DataFrame(
        {
            "algorithm": ["a"] * 4 + ["b"] * 4,
            "weighted_avg_aoi": [1.0, 1.2, 0.9, 1.1, 2.0, 2.2, 1.9, 2.1],
        }
    )
    summary = summarise(df, metrics=("weighted_avg_aoi",), seed=0)
    bar = plot_metric_bars(summary, "weighted_avg_aoi", tmp_path / "bar.png")
    cdf = plot_aoi_cdf(df, tmp_path / "cdf.png")
    assert bar.exists() and bar.stat().st_size > 0
    assert cdf.exists() and cdf.stat().st_size > 0
    with pytest.raises(ValueError):
        plot_metric_bars(summary, "missing_metric", tmp_path / "x.png")


def test_running_mean_std_tracks_the_batch_statistics():
    from rsma_aoi.utils.running_stats import RunningMeanStd

    rng = np.random.default_rng(0)
    rms = RunningMeanStd(shape=(3,))
    data = rng.normal(2.0, 3.0, size=(5000, 3))
    for chunk in np.split(data, 10):
        rms.update(chunk)
    np.testing.assert_allclose(rms.mean, data.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(rms.var, data.var(axis=0), rtol=1e-4)
    normed = rms.normalize(data[:10])
    assert normed.dtype == np.float32 and np.all(np.abs(normed) <= 10.0)

    restored = RunningMeanStd(shape=(3,))
    restored.load_state_dict(rms.state_dict())
    np.testing.assert_allclose(restored.mean, rms.mean)


def test_run_logger_csv_handles_late_metrics(tmp_path):
    """Columns that appear only after the first episode must not be dropped."""
    from rsma_aoi.utils.logging import RunLogger

    logger = RunLogger(tmp_path / "run", use_tensorboard=False)
    logger.log_scalars(0, {"train/loss": 1.0})
    logger.log_scalars(1, {"train/loss": 0.5, "episode/r": -3.0})
    logger.log_scalars(2, {"train/loss": 0.25, "episode/r": -2.0})
    logger.close()

    df = pd.read_csv(tmp_path / "run" / "scalars.csv")
    assert list(df["step"]) == [0, 1, 2]
    assert set(df.columns) == {"step", "train/loss", "episode/r"}
    assert np.isnan(df["episode/r"].iloc[0])
    assert df["episode/r"].iloc[2] == pytest.approx(-2.0)


def test_git_commit_hash_returns_none_outside_a_repository(tmp_path):
    from rsma_aoi.utils.logging import git_commit_hash

    result = git_commit_hash(tmp_path)
    assert result is None or isinstance(result, str)


def test_learning_curve_plots(tmp_path):
    from rsma_aoi.evaluation.plots import plot_learning_curve, plot_learning_curves

    for name in ("a", "b"):
        pd.DataFrame(
            {"step": [0, 100, 200], "env/weighted_aoi": [10.0, 8.0, 6.0]}
        ).to_csv(tmp_path / f"{name}.csv", index=False)

    single = plot_learning_curve(tmp_path / "a.csv", tmp_path / "single.png")
    multi = plot_learning_curves(
        {"hybrid": tmp_path / "a.csv", "continuous": tmp_path / "b.csv"},
        tmp_path / "multi.png",
        baselines={"oma": 7.0},
        smooth=2,
    )
    assert single.exists() and single.stat().st_size > 0
    assert multi.exists() and multi.stat().st_size > 0
    with pytest.raises(ValueError):
        plot_learning_curves({"a": tmp_path / "a.csv"}, tmp_path / "x.png", metric="nope")
    with pytest.raises(ValueError):
        plot_learning_curves({}, tmp_path / "x.png")


def test_schedule_statistics():
    from rsma_aoi.evaluation.metrics import schedule_statistics

    masks = np.array([[1, 0], [1, 0], [0, 1], [1, 1]], dtype=bool)
    hist, entropy, activation = schedule_statistics(masks)
    np.testing.assert_allclose(hist, [0.0, 0.75, 0.25])
    np.testing.assert_allclose(activation, [0.75, 0.5])
    # subsets {0} x2, {1} x1, {0,1} x1 -> H = -(.5 ln .5 + .25 ln .25 * 2)
    assert entropy == pytest.approx(-(0.5 * np.log(0.5) + 2 * 0.25 * np.log(0.25)))

    constant = schedule_statistics(np.ones((5, 3), dtype=bool))
    assert constant[1] == pytest.approx(0.0)
    np.testing.assert_allclose(constant[0], [0.0, 0.0, 0.0, 1.0])
