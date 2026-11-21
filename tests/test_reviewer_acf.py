import csv
import json

import numpy as np
import pytest
import torch

from src.analysis.reviewer_acf import (
    EXPECTED_DIMENSIONS,
    EXPECTED_HORIZON,
    EXPECTED_TRAINING_WINDOWS,
    WithinSequenceACF,
    _expected_pooled_pair_counts,
    _pooled_y_e_comparison,
    _validate_reproduction_constraints,
    _write_report,
)


def test_pooled_acf_matches_only_matched_within_sequence_pairs():
    values = torch.tensor(
        [
            [[1.0, 9.0], [2.0, 7.0], [4.0, 4.0], [8.0, 2.0]],
            [[3.0, 1.0], [5.0, 2.0], [9.0, 6.0], [17.0, 14.0]],
        ]
    )
    estimator = WithinSequenceACF(max_lag=2)
    estimator.update(values)
    result = estimator.finalize()

    for lag in (1, 2):
        left = values[:, :-lag, :].numpy().reshape(-1)
        right = values[:, lag:, :].numpy().reshape(-1)
        expected = np.corrcoef(left, right)[0, 1]
        assert result["pooled_acf"][lag - 1] == pytest.approx(expected)
        assert result["pooled_pair_count"][lag - 1] == 2 * 2 * (4 - lag)


def test_sequence_mean_excludes_constant_sequences_and_preserves_boundaries():
    # Channel 0 is constant and invalid; channel 1 is increasing and has ACF 1.
    values = torch.tensor([[[2.0, 1.0], [2.0, 3.0], [2.0, 7.0], [2.0, 15.0]]])
    estimator = WithinSequenceACF(max_lag=1)
    estimator.update(values)
    result = estimator.finalize()

    assert result["valid_sequence_count"][0] == 1
    assert result["mean_sequence_acf"][0] == pytest.approx(1.0)
    assert result["window_count"] == 1
    assert result["scalar_count"] == 8


def test_channel_mean_acf_correlates_each_channel_across_windows_before_averaging():
    # The constant first channel is excluded.  The second and third channels
    # receive equal weight even though their values have very different scales.
    values = torch.tensor(
        [
            [[2.0, 1.0, 8.0], [2.0, 2.0, 5.0], [2.0, 4.0, 3.0], [2.0, 8.0, 2.0]],
            [[2.0, 3.0, 9.0], [2.0, 5.0, 6.0], [2.0, 9.0, 4.0], [2.0, 17.0, 3.0]],
        ]
    )
    estimator = WithinSequenceACF(max_lag=1)
    estimator.update(values[:1])
    estimator.update(values[1:])
    result = estimator.finalize()

    per_channel = []
    for channel in (1, 2):
        left = values[:, :-1, channel].numpy().reshape(-1)
        right = values[:, 1:, channel].numpy().reshape(-1)
        per_channel.append(np.corrcoef(left, right)[0, 1])
    assert result["valid_channel_count"][0] == 2
    assert result["mean_channel_acf"][0] == pytest.approx(np.mean(per_channel))


def test_rejects_cross_boundary_prone_short_horizon_and_nonfinite_input():
    estimator = WithinSequenceACF(max_lag=3)
    with pytest.raises(ValueError, match="must exceed"):
        estimator.update(torch.ones(1, 3, 1))
    with pytest.raises(ValueError, match="non-finite"):
        estimator.update(torch.tensor([[[1.0], [float("nan")], [3.0], [4.0]]]))


def test_reproduction_constraints_and_pair_count_formula_are_fixed():
    _validate_reproduction_constraints(
        windows=EXPECTED_TRAINING_WINDOWS,
        horizon=EXPECTED_HORIZON,
        dimensions=EXPECTED_DIMENSIONS,
    )
    with pytest.raises(ValueError, match=r"N=18124 .*expected 18125"):
        _validate_reproduction_constraints(
            windows=EXPECTED_TRAINING_WINDOWS - 1,
            horizon=EXPECTED_HORIZON,
            dimensions=EXPECTED_DIMENSIONS,
        )

    pair_counts = _expected_pooled_pair_counts(
        EXPECTED_TRAINING_WINDOWS,
        EXPECTED_HORIZON,
        EXPECTED_DIMENSIONS,
        max_lag=20,
    )
    assert pair_counts[0] == EXPECTED_TRAINING_WINDOWS * EXPECTED_DIMENSIONS * (EXPECTED_HORIZON - 1)
    assert pair_counts[-1] == EXPECTED_TRAINING_WINDOWS * EXPECTED_DIMENSIONS * (EXPECTED_HORIZON - 20)


def test_y_e_comparison_writes_pooled_matched_pair_acfs_and_relative_decline(tmp_path):
    y = torch.tensor([[[1.0], [2.0], [4.0], [8.0]], [[3.0], [5.0], [9.0], [17.0]]])
    e = torch.tensor([[[2.0], [1.0], [5.0], [7.0]], [[1.0], [6.0], [8.0], [19.0]]])
    estimators = {name: WithinSequenceACF(max_lag=2) for name in ("Y", "E", "Z")}
    estimators["Y"].update(y)
    estimators["E"].update(e)
    estimators["Z"].update(e)
    results = {name: estimator.finalize() for name, estimator in estimators.items()}

    comparison = _pooled_y_e_comparison(results)
    expected_y = np.corrcoef(y[:, :-1, :].numpy().reshape(-1), y[:, 1:, :].numpy().reshape(-1))[0, 1]
    expected_e = np.corrcoef(e[:, :-1, :].numpy().reshape(-1), e[:, 1:, :].numpy().reshape(-1))[0, 1]
    assert comparison[0]["acf_y"] == pytest.approx(expected_y)
    assert comparison[0]["acf_e"] == pytest.approx(expected_e)
    assert comparison[0]["relative_decline"] == pytest.approx(1.0 - expected_e / expected_y)

    metadata = {
        "N": 2,
        "H": 4,
        "D": 1,
        "NHD": 8,
        "sigma_eps": 0.1,
        "sigma_clipped_scalar_count": 0,
        "sigma_clipped_scalar_fraction": 0.0,
        "checkpoint_path": "/tmp/best_model.pth",
        "checkpoint_sha256": "checkpoint",
        "stage1_checkpoint_path": "/tmp/stage1.pth",
        "stage1_checkpoint_sha256": "stage1",
        "sigma_summary": {
            "mean": 1.0,
            "min": 1.0,
            "max": 1.0,
            "max_abs_from_one": 0.0,
            "z_equals_e_exactly": True,
        },
    }
    _write_report(tmp_path, results, metadata)

    with (tmp_path / "electricity_training_y_e_acf.csv").open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["lag"] == "1"
    assert float(csv_rows[0]["relative_decline"]) == pytest.approx(comparison[0]["relative_decline"])
    report_json = json.loads((tmp_path / "electricity_training_residual_acf.json").read_text(encoding="utf-8"))
    assert report_json["acf"]["Y"]["pooled_acf"][0] == pytest.approx(expected_y)
    assert report_json["acf"]["Y"]["mean_channel_acf"][0] == pytest.approx(expected_y)
    assert report_json["pooled_y_e_comparison"] == comparison
    report_markdown = (tmp_path / "electricity_training_residual_acf.md").read_text(encoding="utf-8")
    assert "| Lag | ACF_Y(k) | ACF_E(k) | 1 - ACF_E(k) / ACF_Y(k) |" in report_markdown


def test_y_e_relative_decline_is_missing_when_y_acf_is_zero_or_nonfinite():
    comparison = _pooled_y_e_comparison(
        {
            "Y": {"pooled_acf": np.asarray([0.0, np.nan])},
            "E": {"pooled_acf": np.asarray([0.5, 0.25])},
        }
    )
    assert [item["relative_decline"] for item in comparison] == [None, None]
