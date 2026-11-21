import csv

import numpy as np
import pytest
import torch

from src.analysis.electricity_window_acf import (
    EXPECTED_TRAINING_POINTS,
    EXPECTED_WINDOWS,
    WindowACFAccumulator,
    _write_outputs,
    expected_window_count,
)


def _direct_window_acf(sequence, max_lag):
    centered = np.asarray(sequence, dtype=np.float64) - np.mean(sequence)
    variance = np.dot(centered, centered) / len(centered)
    return np.asarray(
        [np.dot(centered[: len(centered) - lag], centered[lag:]) / len(centered) / variance for lag in range(max_lag + 1)]
    )


def test_fft_acf_matches_direct_traditional_window_acf_and_excludes_constants():
    values = np.asarray(
        [
            [[1.0, 5.0], [2.0, 5.0], [4.0, 5.0], [8.0, 5.0]],
            [[3.0, 5.0], [5.0, 5.0], [9.0, 5.0], [17.0, 5.0]],
        ]
    )
    accumulator = WindowACFAccumulator(max_lag=3)
    accumulator.update(values)
    result = accumulator.finalize()

    expected = np.mean(
        [_direct_window_acf(values[window, :, 0], 3) for window in range(2)], axis=0
    )
    assert result["valid_window_channel_count"] == 2
    assert result["total_window_channel_count"] == 4
    assert result["mean_acf"] == pytest.approx(expected)
    assert result["mean_acf"][0] == pytest.approx(1.0)


def test_acf_accumulation_is_independent_of_batch_boundaries():
    values = np.asarray(
        [
            [[1.0], [2.0], [4.0], [8.0]],
            [[3.0], [5.0], [9.0], [17.0]],
            [[4.0], [7.0], [11.0], [16.0]],
        ]
    )
    one_batch = WindowACFAccumulator(max_lag=2)
    one_batch.update(values)
    split_batches = WindowACFAccumulator(max_lag=2)
    split_batches.update(values[:1])
    split_batches.update(values[1:])
    assert split_batches.finalize()["mean_acf"] == pytest.approx(one_batch.finalize()["mean_acf"])


def test_torch_fft_acf_matches_numpy_fft_acf():
    values = np.asarray(
        [
            [[1.0, 2.0], [2.0, 5.0], [4.0, 3.0], [8.0, 7.0]],
            [[3.0, 1.0], [5.0, 2.0], [9.0, 6.0], [17.0, 14.0]],
        ],
        dtype=np.float32,
    )
    numpy_accumulator = WindowACFAccumulator(max_lag=2)
    numpy_accumulator.update(values)
    torch_accumulator = WindowACFAccumulator(max_lag=2)
    torch_accumulator.update_torch(torch.from_numpy(values))
    assert torch_accumulator.finalize()["mean_acf"] == pytest.approx(
        numpy_accumulator.finalize()["mean_acf"], abs=1e-6
    )


def test_expected_96_to_192_training_window_count_is_fixed():
    assert expected_window_count(EXPECTED_TRAINING_POINTS) == EXPECTED_WINDOWS


def test_csv_and_png_output_include_lags_zero_to_48(tmp_path):
    target = {
        "mean_acf": np.linspace(1.0, 0.0, 49),
        "valid_window_channel_count": 10,
        "total_window_channel_count": 12,
    }
    residual = {
        "mean_acf": np.linspace(1.0, -0.1, 49),
        "valid_window_channel_count": 9,
        "total_window_channel_count": 12,
    }
    _write_outputs(tmp_path, target, residual)
    with (tmp_path / "electricity_training_window_acf.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["lag"]) for row in rows] == list(range(49))
    assert rows[0]["mean_acf_raw_point_residual"] == "1"
    assert (tmp_path / "electricity_training_window_acf.png").stat().st_size > 0
