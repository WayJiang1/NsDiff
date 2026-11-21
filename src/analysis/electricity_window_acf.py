#!/usr/bin/env python3
"""Compute window-level traditional ACFs for Electricity training targets.

The analysis uses the first 70% of ``electricity.csv`` and reproduces the
96-to-192 iTransformer point-prediction interface.  For every target window and
channel, it centers the complete 192-point series, calculates its linear
autocovariance with an FFT, divides by the lag-zero autocovariance, and then
equally averages valid window-channel ACFs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_timeseries.utils.timefeatures import time_features


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.iTransformer import Model as ITransformer  # noqa: E402


TRAIN_RATIO = 0.70
INPUT_LENGTH = 96
TARGET_LENGTH = 192
MAX_LAG = 48
EXPECTED_TRAINING_POINTS = 18_412
EXPECTED_WINDOWS = 18_125
EXPECTED_CHANNELS = 321
DEFAULT_DATA_PATH = REPO_ROOT / "data/electricity/electricity.csv"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / (
    "results/runs/iTransformer/"
    "electricity_96_192_Electricity_M_ft96_sl48_ll192_pl512_dm8_nh3_el1_dl512_df1_"
    "fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/analysis/electricity_window_acf"


def expected_window_count(training_points: int) -> int:
    """Return the number of stride-one 96-to-192 training windows."""
    return training_points - INPUT_LENGTH - TARGET_LENGTH + 1


class WindowACFAccumulator:
    """Accumulate equal-weight traditional ACFs for ``[B, H, D]`` values."""

    def __init__(self, max_lag: int = MAX_LAG, eps: float = 1e-12) -> None:
        if max_lag < 0:
            raise ValueError("max_lag must be non-negative.")
        self.max_lag = int(max_lag)
        self.eps = float(eps)
        self.acf_sum = np.zeros(self.max_lag + 1, dtype=np.float64)
        self.valid_count = 0
        self.total_count = 0

    def update(self, values: np.ndarray) -> None:
        """Add a batch of target or residual windows without crossing boundaries."""
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 3:
            raise ValueError("Expected values shaped [B, H, D], got %s." % (values.shape,))
        batch_size, horizon, channels = values.shape
        if horizon <= self.max_lag:
            raise ValueError("horizon must exceed max_lag.")
        if not np.isfinite(values).all():
            raise ValueError("ACF input contains non-finite values.")

        # FFT requires a length that prevents circular wraparound in the first
        # ``horizon`` autocovariances.  The power-of-two choice is faster while
        # still representing the linear autocorrelation exactly.
        sequences = np.moveaxis(values, 2, 1)  # [B, D, H]
        centered = sequences - sequences.mean(axis=-1, keepdims=True)
        n_fft = 1 << (2 * horizon - 1).bit_length()
        spectrum = np.fft.rfft(centered, n=n_fft, axis=-1)
        autocovariance = np.fft.irfft(
            spectrum * spectrum.conjugate(), n=n_fft, axis=-1
        )[..., : self.max_lag + 1] / horizon

        variance = autocovariance[..., 0]
        valid = np.isfinite(variance) & (variance > self.eps)
        acf = np.divide(
            autocovariance,
            variance[..., None],
            out=np.full_like(autocovariance, np.nan),
            where=valid[..., None],
        )
        valid &= np.isfinite(acf).all(axis=-1)
        self.total_count += batch_size * channels
        if valid.any():
            self.acf_sum += acf[valid].sum(axis=0)
            self.valid_count += int(valid.sum())

    def update_torch(self, values: torch.Tensor) -> None:
        """Add a tensor batch, keeping the FFT on its current device.

        CUDA runs use this path so only the 49 accumulated ACF sums are copied
        to the CPU per batch.  The numerical definition matches :meth:`update`.
        """
        if values.ndim != 3:
            raise ValueError("Expected values shaped [B, H, D], got %s." % (tuple(values.shape),))
        batch_size, horizon, channels = values.shape
        if horizon <= self.max_lag:
            raise ValueError("horizon must exceed max_lag.")
        if not torch.isfinite(values).all():
            raise ValueError("ACF input contains non-finite values.")

        sequences = values.permute(0, 2, 1)
        centered = sequences - sequences.mean(dim=-1, keepdim=True)
        n_fft = 1 << (2 * horizon - 1).bit_length()
        spectrum = torch.fft.rfft(centered, n=n_fft, dim=-1)
        autocovariance = torch.fft.irfft(
            spectrum * spectrum.conj(), n=n_fft, dim=-1
        )[..., : self.max_lag + 1] / horizon

        variance = autocovariance[..., 0]
        valid = torch.isfinite(variance) & (variance > self.eps)
        acf = torch.where(
            valid[..., None],
            autocovariance / variance[..., None],
            torch.full_like(autocovariance, float("nan")),
        )
        valid &= torch.isfinite(acf).all(dim=-1)
        self.total_count += int(batch_size * channels)
        if valid.any():
            self.acf_sum += acf[valid].sum(dim=0).double().cpu().numpy()
            self.valid_count += int(valid.sum().item())

    def finalize(self) -> Dict[str, Any]:
        """Return the mean ACF and sequence accounting for all processed batches."""
        mean_acf = np.full(self.max_lag + 1, np.nan, dtype=np.float64)
        if self.valid_count:
            mean_acf = self.acf_sum / self.valid_count
        return {
            "mean_acf": mean_acf,
            "valid_window_channel_count": self.valid_count,
            "total_window_channel_count": self.total_count,
        }


def _load_training_data(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load the first 70% of Electricity and apply train-only channel scaling."""
    frame = pd.read_csv(data_path)
    if "date" not in frame.columns:
        raise ValueError("Electricity CSV must contain a 'date' column.")
    values = frame.drop(columns="date").to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != EXPECTED_CHANNELS:
        raise ValueError(
            "Expected %d numeric Electricity channels, got shape %s."
            % (EXPECTED_CHANNELS, values.shape)
        )
    if not np.isfinite(values).all():
        raise ValueError("Electricity CSV contains non-finite channel values.")

    training_points = int(TRAIN_RATIO * len(frame))
    if training_points != EXPECTED_TRAINING_POINTS:
        raise ValueError(
            "Expected %d training points from the 70%% split, got %d."
            % (EXPECTED_TRAINING_POINTS, training_points)
        )
    train_values = values[:training_points]
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0, ddof=0)
    if np.any(~np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("A training channel has a non-positive or non-finite standard deviation.")
    scaled = (train_values - mean) / std

    dates = pd.DataFrame({"date": pd.to_datetime(frame.loc[: training_points - 1, "date"])})
    # Match SlidingWindowTS(time_enc=3, freq='h'), which was used to train the
    # supplied iTransformer checkpoint.
    marks = time_features(dates, timeenc=3, freq="h").astype(np.float32, copy=False)
    return scaled, marks


def _build_itransformer(checkpoint_path: Path, device: torch.device) -> ITransformer:
    """Construct the exact 96-to-192 architecture encoded by the checkpoint."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Missing iTransformer checkpoint: %s" % checkpoint_path)
    config = SimpleNamespace(
        seq_len=INPUT_LENGTH,
        pred_len=TARGET_LENGTH,
        output_attention=False,
        use_norm=True,
        d_model=512,
        embed="timeF",
        freq="h",
        dropout=0.1,
        class_strategy="projection",
        factor=1,
        n_heads=8,
        d_ff=512,
        e_layers=3,
        activation="gelu",
    )
    model = ITransformer(config)
    state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint must contain an iTransformer state dictionary.")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def _write_outputs(
    output_dir: Path,
    target_result: Dict[str, Any],
    residual_result: Dict[str, Any],
) -> None:
    """Write the tabular and plotted average ACF results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "electricity_training_window_acf.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "lag",
                "mean_acf_y",
                "mean_acf_raw_point_residual",
                "valid_window_channel_count_y",
                "valid_window_channel_count_raw_point_residual",
                "total_window_channel_count",
            ),
        )
        writer.writeheader()
        for lag, (target_acf, residual_acf) in enumerate(
            zip(target_result["mean_acf"], residual_result["mean_acf"])
        ):
            writer.writerow(
                {
                    "lag": lag,
                    "mean_acf_y": "%.12g" % target_acf,
                    "mean_acf_raw_point_residual": "%.12g" % residual_acf,
                    "valid_window_channel_count_y": target_result[
                        "valid_window_channel_count"
                    ],
                    "valid_window_channel_count_raw_point_residual": residual_result[
                        "valid_window_channel_count"
                    ],
                    "total_window_channel_count": target_result["total_window_channel_count"],
                }
            )

    lags = np.arange(len(target_result["mean_acf"]))
    fig, axis = plt.subplots(figsize=(7.5, 4.5), dpi=200)
    axis.plot(lags, target_result["mean_acf"], label="Target Y", linewidth=2)
    axis.plot(
        lags,
        residual_result["mean_acf"],
        label=r"Raw point residual $E=Y-\hat{\mu}$",
        linewidth=2,
    )
    axis.set(xlabel="Lag", ylabel="Mean traditional ACF", xlim=(0, MAX_LAG))
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "electricity_training_window_acf.png", bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def run_analysis(
    data_path: Path = DEFAULT_DATA_PATH,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    batch_size: int = 32,
    device: str = "auto",
) -> Dict[str, Any]:
    """Calculate and save target and raw point-residual average ACF curves."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    device_obj = _select_device(device)
    scaled, marks = _load_training_data(Path(data_path))
    windows = expected_window_count(len(scaled))
    if windows != EXPECTED_WINDOWS:
        raise RuntimeError("Expected %d target windows, got %d." % (EXPECTED_WINDOWS, windows))
    model = _build_itransformer(Path(checkpoint_path), device_obj)
    target_acf = WindowACFAccumulator(MAX_LAG)
    residual_acf = WindowACFAccumulator(MAX_LAG)

    for first in range(0, windows, batch_size):
        last = min(first + batch_size, windows)
        starts = range(first, last)
        history = np.stack([scaled[index : index + INPUT_LENGTH] for index in starts])
        target = np.stack(
            [
                scaled[index + INPUT_LENGTH : index + INPUT_LENGTH + TARGET_LENGTH]
                for index in range(first, last)
            ]
        )
        history_marks = np.stack([marks[index : index + INPUT_LENGTH] for index in range(first, last)])

        history_tensor = torch.from_numpy(history.astype(np.float32, copy=False)).to(device_obj)
        target_tensor = torch.from_numpy(target.astype(np.float32, copy=False)).to(device_obj)
        marks_tensor = torch.from_numpy(history_marks).to(device_obj)
        prediction = model(history_tensor, marks_tensor, None, None)
        if tuple(prediction.shape) != tuple(target_tensor.shape):
            raise RuntimeError(
                "iTransformer prediction shape %s does not match target shape %s."
                % (tuple(prediction.shape), tuple(target_tensor.shape))
            )
        if device_obj.type == "cuda":
            target_acf.update_torch(target_tensor)
            residual_acf.update_torch(target_tensor - prediction)
        else:
            target_acf.update(target)
            residual_acf.update(target - prediction.cpu().numpy())

    target_result = target_acf.finalize()
    residual_result = residual_acf.finalize()
    expected_total = EXPECTED_WINDOWS * EXPECTED_CHANNELS
    if (
        target_result["total_window_channel_count"] != expected_total
        or residual_result["total_window_channel_count"] != expected_total
    ):
        raise RuntimeError("Window-channel accounting does not match the expected training split.")
    _write_outputs(Path(output_dir), target_result, residual_result)
    return {
        "data_path": str(Path(data_path).resolve()),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "training_points": len(scaled),
        "windows": windows,
        "channels": scaled.shape[1],
        "target_valid_window_channel_count": target_result["valid_window_channel_count"],
        "residual_valid_window_channel_count": residual_result["valid_window_channel_count"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="Torch device or 'auto'.")
    return parser.parse_args()


if __name__ == "__main__":
    report = run_analysis(**vars(parse_args()))
    print(
        "Wrote Electricity window ACF: N={windows}, D={channels}, "
        "valid Y={target_valid_window_channel_count}, "
        "valid E={residual_valid_window_channel_count}".format(**report)
    )
