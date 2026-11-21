#!/usr/bin/env python3
"""Report strict within-forecast-window target and residual ACFs for Electricity.

The analysis deliberately operates on the *training* loader.  For every lag
``k`` it pairs only ``V[n, t, d]`` with ``V[n, t + k, d]``; it never turns the
``[window, horizon, channel]`` tensor into one artificial time series.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Type

import numpy as np
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SIGMA_EPS = 0.01
MAX_LAG = 20
EXPECTED_TRAINING_WINDOWS = 18_125
EXPECTED_HORIZON = 192
EXPECTED_DIMENSIONS = 321


def sha256(path: Path) -> str:
    """Return a file digest without loading a checkpoint into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class WithinSequenceACF:
    """Streaming lagged Pearson correlations for values shaped ``[B,H,D]``.

    Besides a single pooled correlation and an equal-weight mean over individual
    ``(window, channel)`` sequences, this keeps sufficient statistics for every
    channel.  The latter permits the reviewer-requested equal-weight mean of
    channel-level ACFs, where each channel is correlated across all windows and
    valid horizon positions before the channel ACFs are averaged.
    """

    def __init__(self, max_lag: int, eps: float = 1e-12) -> None:
        if max_lag < 1:
            raise ValueError("max_lag must be at least 1.")
        self.max_lag = int(max_lag)
        self.eps = float(eps)
        # Count, sums of each paired side, squares, and cross-products.
        self.pooled = np.zeros((self.max_lag, 6), dtype=np.float64)
        self.channel: Optional[np.ndarray] = None
        self.sequence_sum = np.zeros(self.max_lag, dtype=np.float64)
        self.sequence_count = np.zeros(self.max_lag, dtype=np.int64)
        self.window_count = 0
        self.scalar_count = 0

    def update(self, residual: torch.Tensor) -> None:
        """Accumulate a batch, preserving its window and channel boundaries."""
        if residual.ndim != 3:
            raise ValueError("Expected residual shape [B, H, D], got %s." % (tuple(residual.shape),))
        batch_size, horizon, channels = residual.shape
        if horizon <= self.max_lag:
            raise ValueError(
                "horizon=%d must exceed max_lag=%d." % (horizon, self.max_lag)
            )
        if not torch.isfinite(residual).all():
            raise ValueError("Residual tensor contains non-finite values.")

        values = residual.detach().to(dtype=torch.float64)
        sequences = values.permute(0, 2, 1)  # [B, D, H]; each row is one valid series.
        if self.channel is None:
            self.channel = np.zeros((self.max_lag, channels, 6), dtype=np.float64)
        elif self.channel.shape[1] != channels:
            raise ValueError(
                "Channel count changed from %d to %d while accumulating ACF."
                % (self.channel.shape[1], channels)
            )
        self.window_count += int(batch_size)
        self.scalar_count += int(batch_size * horizon * channels)

        for lag in range(1, self.max_lag + 1):
            left = sequences[..., :-lag]
            right = sequences[..., lag:]
            pair_count = int(left.numel())
            row = self.pooled[lag - 1]
            row[0] += pair_count
            row[1] += float(left.sum().item())
            row[2] += float(right.sum().item())
            row[3] += float(left.square().sum().item())
            row[4] += float(right.square().sum().item())
            row[5] += float((left * right).sum().item())

            # One sufficient-statistics row per channel.  Reducing only across
            # batch and horizon means each channel is centered and normalized
            # independently in ``finalize``.
            channel_row = self.channel[lag - 1]
            channel_pair_count = batch_size * (horizon - lag)
            channel_row[:, 0] += channel_pair_count
            channel_row[:, 1] += left.sum(dim=(0, 2)).cpu().numpy()
            channel_row[:, 2] += right.sum(dim=(0, 2)).cpu().numpy()
            channel_row[:, 3] += left.square().sum(dim=(0, 2)).cpu().numpy()
            channel_row[:, 4] += right.square().sum(dim=(0, 2)).cpu().numpy()
            channel_row[:, 5] += (left * right).sum(dim=(0, 2)).cpu().numpy()

            left_centered = left - left.mean(dim=-1, keepdim=True)
            right_centered = right - right.mean(dim=-1, keepdim=True)
            denominator = left_centered.square().sum(dim=-1).sqrt()
            denominator *= right_centered.square().sum(dim=-1).sqrt()
            correlation = (left_centered * right_centered).sum(dim=-1) / denominator
            valid = torch.isfinite(correlation) & (denominator > self.eps)
            if valid.any():
                self.sequence_sum[lag - 1] += float(correlation[valid].sum().item())
                self.sequence_count[lag - 1] += int(valid.sum().item())

    def finalize(self) -> Dict[str, np.ndarray]:
        """Return pooled, equal-channel, and equal-window/channel ACF estimates."""
        pooled_acf = np.full(self.max_lag, np.nan, dtype=np.float64)
        for lag, (count, sx, sy, sx2, sy2, sxy) in enumerate(self.pooled, start=1):
            if count <= 1:
                continue
            numerator = sxy - sx * sy / count
            var_left = sx2 - sx * sx / count
            var_right = sy2 - sy * sy / count
            denominator = math.sqrt(max(var_left, 0.0) * max(var_right, 0.0))
            if denominator > self.eps:
                pooled_acf[lag - 1] = numerator / denominator

        channel_mean_acf = np.full(self.max_lag, np.nan, dtype=np.float64)
        valid_channel_count = np.zeros(self.max_lag, dtype=np.int64)
        if self.channel is not None:
            for lag, rows in enumerate(self.channel):
                counts, sx, sy, sx2, sy2, sxy = rows.T
                numerator = sxy - sx * sy / counts
                var_left = sx2 - sx * sx / counts
                var_right = sy2 - sy * sy / counts
                denominator = np.sqrt(
                    np.maximum(var_left, 0.0) * np.maximum(var_right, 0.0)
                )
                valid = (counts > 1) & (denominator > self.eps)
                valid_channel_count[lag] = int(valid.sum())
                if valid.any():
                    channel_mean_acf[lag] = float((numerator[valid] / denominator[valid]).mean())

        sequence_acf = np.full(self.max_lag, np.nan, dtype=np.float64)
        valid_counts = self.sequence_count > 0
        sequence_acf[valid_counts] = (
            self.sequence_sum[valid_counts] / self.sequence_count[valid_counts]
        )
        return {
            "pooled_acf": pooled_acf,
            "pooled_pair_count": self.pooled[:, 0].astype(np.int64),
            "mean_channel_acf": channel_mean_acf,
            "valid_channel_count": valid_channel_count,
            "mean_sequence_acf": sequence_acf,
            "valid_sequence_count": self.sequence_count.copy(),
            "window_count": np.asarray(self.window_count, dtype=np.int64),
            "scalar_count": np.asarray(self.scalar_count, dtype=np.int64),
        }


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_checkpoint_dir(run_dir: Path, checkpoint_mode: int) -> Path:
    mode_dir = "train_mode_%d" % checkpoint_mode
    if run_dir.name == mode_dir:
        candidate = run_dir
    else:
        candidate = run_dir / mode_dir
    checkpoint = candidate / "best_model.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError("Expected designated checkpoint at %s." % checkpoint)
    return candidate


def _filter_init_kwargs(cls: Type[Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    if not is_dataclass(cls):
        return dict(config)
    allowed = {entry.name for entry in fields(cls)}
    return {key: value for key, value in config.items() if key in allowed}


def _validate_stage1_provenance(
    checkpoint_dir: Path, stage1_checkpoint: Optional[Path] = None
) -> Dict[str, str]:
    if stage1_checkpoint is not None:
        path = stage1_checkpoint.resolve()
        if not path.is_file():
            raise FileNotFoundError("Specified stage-1 checkpoint is unavailable: %s" % path)
        return {"path": str(path), "sha256": sha256(path), "source": "explicit_cli"}

    provenance_path = checkpoint_dir / "stage1_checkpoint.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            "Missing stage-1 provenance file: %s. Pass --stage1-checkpoint to bind an "
            "older run to its designated local iTransformer checkpoint." % provenance_path
        )
    provenance = _read_json(provenance_path)
    path = Path(provenance["path"])
    expected_sha = str(provenance["sha256"]).lower()
    if not path.is_file():
        raise FileNotFoundError("Recorded stage-1 checkpoint is unavailable: %s" % path)
    actual_sha = sha256(path)
    if actual_sha != expected_sha:
        raise ValueError(
            "Stage-1 checkpoint SHA-256 mismatch: expected %s, got %s." % (expected_sha, actual_sha)
        )
    return {"path": str(path.resolve()), "sha256": actual_sha, "source": "run_provenance"}


def load_experiment(
    run_dir: Path,
    checkpoint_mode: int,
    device: str,
    num_worker: int,
    stage1_checkpoint: Optional[Path] = None,
) -> Tuple[Any, Dict[str, Any], Path, Dict[str, str]]:
    """Recreate an iReflow experiment and bind it to one explicitly selected run."""
    checkpoint_dir = _find_checkpoint_dir(run_dir, checkpoint_mode)
    args_path = checkpoint_dir / "args.json"
    if not args_path.is_file():
        raise FileNotFoundError("Missing saved run arguments: %s" % args_path)
    config = _read_json(args_path)
    model_name = str(config.get("model_type") or config.get("model") or "")
    if model_name not in {"iReflow", "LS-Flow", "PDN-Flow"}:
        raise ValueError("This analysis supports iReflow/LS-Flow runs, got %r." % model_name)

    from src.experiments.iReflow import iReflowExp

    kwargs = _filter_init_kwargs(iReflowExp, config)
    kwargs["device"] = device
    kwargs["num_worker"] = int(num_worker)
    kwargs["wandb_project"] = None
    experiment = iReflowExp(**kwargs)
    experiment.run_save_dir = str(checkpoint_dir)
    experiment.best_checkpoint_filepath = str(checkpoint_dir / "best_model.pth")
    experiment.run_checkpoint_filepath = str(checkpoint_dir / "run_checkpoint.pth")
    experiment._init_data_loader(shuffle=False, fast_test=False, fast_val=False)
    experiment._init_model()
    experiment._load_best_model()
    experiment.model.eval()
    stage1 = _validate_stage1_provenance(checkpoint_dir, stage1_checkpoint=stage1_checkpoint)
    return experiment, config, checkpoint_dir, stage1


def _json_number(value: float) -> Optional[float]:
    return float(value) if math.isfinite(float(value)) else None


def _validate_reproduction_constraints(windows: int, horizon: int, dimensions: int) -> None:
    """Reject runs that do not use the reviewer-specified training configuration."""
    expected = {
        "N": EXPECTED_TRAINING_WINDOWS,
        "H": EXPECTED_HORIZON,
        "D": EXPECTED_DIMENSIONS,
    }
    observed = {"N": int(windows), "H": int(horizon), "D": int(dimensions)}
    mismatches = [
        "%s=%d (expected %d)" % (name, observed[name], expected[name])
        for name in expected
        if observed[name] != expected[name]
    ]
    if mismatches:
        raise ValueError(
            "Electricity reviewer report requires %s; got %s."
            % (
                ", ".join("%s=%d" % (name, value) for name, value in expected.items()),
                ", ".join(mismatches),
            )
        )


def _expected_pooled_pair_counts(windows: int, horizon: int, dimensions: int, max_lag: int) -> np.ndarray:
    return np.asarray(
        [windows * dimensions * (horizon - lag) for lag in range(1, max_lag + 1)],
        dtype=np.int64,
    )


def _pooled_y_e_comparison(
    results: Mapping[str, Mapping[str, np.ndarray]],
) -> list[Dict[str, Optional[float]]]:
    """Return the requested matched-pair pooled Y/E ACF comparison."""
    y_acf = results["Y"]["pooled_acf"]
    e_acf = results["E"]["pooled_acf"]
    if len(y_acf) != len(e_acf):
        raise ValueError("Y and E ACF results must have the same number of lags.")

    comparison = []
    for index, (y_value, e_value) in enumerate(zip(y_acf, e_acf), start=1):
        y_number = _json_number(y_value)
        e_number = _json_number(e_value)
        relative_decline = None
        if y_number is not None and e_number is not None and y_number != 0.0:
            relative_decline = _json_number(1.0 - e_number / y_number)
        comparison.append(
            {
                "lag": index,
                "acf_y": y_number,
                "acf_e": e_number,
                "relative_decline": relative_decline,
            }
        )
    return comparison


def _markdown_number(value: Optional[float]) -> str:
    return "N/A" if value is None else "%.8f" % value


def _write_report(
    output_dir: Path,
    results: Mapping[str, Mapping[str, np.ndarray]],
    metadata: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = _pooled_y_e_comparison(results)
    csv_path = output_dir / "electricity_training_residual_acf.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "residual",
                "aggregation",
                "lag",
                "acf",
                "pair_count",
                "valid_channel_count",
                "valid_sequence_count",
            ),
        )
        writer.writeheader()
        for residual_name, payload in results.items():
            for index in range(len(payload["pooled_acf"])):
                common = {
                    "residual": residual_name,
                    "lag": index + 1,
                    "pair_count": int(payload["pooled_pair_count"][index]),
                    "valid_channel_count": int(payload["valid_channel_count"][index]),
                    "valid_sequence_count": int(payload["valid_sequence_count"][index]),
                }
                writer.writerow({**common, "aggregation": "pooled_pearson", "acf": _json_number(payload["pooled_acf"][index])})
                writer.writerow({**common, "aggregation": "mean_channel_pearson", "acf": _json_number(payload["mean_channel_acf"][index])})
                writer.writerow({**common, "aggregation": "mean_within_window_channel_pearson", "acf": _json_number(payload["mean_sequence_acf"][index])})

    comparison_csv_path = output_dir / "electricity_training_y_e_acf.csv"
    with comparison_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("lag", "acf_y", "acf_e", "relative_decline"),
        )
        writer.writeheader()
        writer.writerows(comparison)

    json_payload = dict(metadata)
    json_payload["acf"] = {
        name: {
            key: [int(item) if np.issubdtype(np.asarray(item).dtype, np.integer) else _json_number(item) for item in value]
            for key, value in payload.items()
            if key in {
                "pooled_acf",
                "pooled_pair_count",
                "mean_channel_acf",
                "valid_channel_count",
                "mean_sequence_acf",
                "valid_sequence_count",
            }
        }
        for name, payload in results.items()
    }
    json_payload["pooled_y_e_comparison"] = comparison
    with (output_dir / "electricity_training_residual_acf.json").open("w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, ensure_ascii=False, indent=2)

    lines = [
        "# Electricity training-loader residual ACF",
        "",
        "The unmodified LS-Flow/Flow Matching target is `Z = (Y - mu_hat) / sigma_hat`. "
        "This report uses the requested numerical sensitivity floor: "
        "`Z_sigma_eps = (Y - mu_hat) / max(sigma_hat, sigma_eps)`. "
        "The matched-pair comparison reports target `Y` and raw point residual `E = Y - mu_hat`.",
        "",
        "## Data and provenance",
        "",
        "- Training forecast windows: `N = {N}`".format(**metadata),
        "- Forecast horizon: `H = {H}`".format(**metadata),
        "- Variables: `D = {D}`".format(**metadata),
        "- Residual scalars: `NHD = {NHD}`".format(**metadata),
        "- Residual scale: training-loader StandardScaler scale.",
        "- Additional sigma floor for this sensitivity analysis: `sigma_eps = {sigma_eps:g}`; "
        "{sigma_clipped_scalar_count} / {NHD} scalars ({sigma_clipped_scalar_fraction:.4%}) were floored.".format(**metadata),
        "- LS-Flow checkpoint: `{checkpoint_path}` (SHA-256 `{checkpoint_sha256}`)".format(**metadata),
        "- Linked iTransformer checkpoint: `{stage1_checkpoint_path}` (SHA-256 `{stage1_checkpoint_sha256}`)".format(**metadata),
        "",
        "## Correlation rule",
        "",
        "For lag `k`, only pairs `(R[n,t,d], R[n,t+k,d])` with the same forecast window `n` "
        "and channel `d` are used. Windows/channels are pooled only as independent matched pairs; "
        "they are never flattened and connected into a time sequence.",
        "",
        "## Requested matched-pair pooled comparison",
        "",
        "| Lag | ACF_Y(k) | ACF_E(k) | 1 - ACF_E(k) / ACF_Y(k) |",
        "|---:|---:|---:|---:|",
    ]
    sigma_summary = metadata["sigma_summary"]
    lines.insert(
        13,
        "- Predicted sigma summary: mean `{mean:.8f}`, range `[{min:.8f}, {max:.8f}]`, "
        "max `|sigma_hat - 1|` `{max_abs_from_one:.8g}`.".format(**sigma_summary),
    )
    if sigma_summary["z_equals_e_exactly"]:
        lines.insert(
            14,
            "- This saved run uses unit sigma throughout (`ablation_mode=mean_only`), so its actual target `Z` is exactly equal to `E`.",
        )
    for item in comparison:
        lines.append(
            "| {lag} | {acf_y} | {acf_e} | {relative_decline} |".format(
                lag=item["lag"],
                acf_y=_markdown_number(item["acf_y"]),
                acf_e=_markdown_number(item["acf_e"]),
                relative_decline=_markdown_number(item["relative_decline"]),
            )
        )
    lines.append("")
    for residual_name, payload in results.items():
        label = (
            "training target Y"
            if residual_name == "Y"
            else "raw point residual E"
            if residual_name == "E"
            else "sigma-clipped standardized residual Z_sigma_eps"
        )
        lines.extend(
            [
                "## %s" % label,
                "",
                "| Lag | Pooled Pearson ACF | Mean channel ACF | Mean within-window/channel ACF |",
                "|---:|---:|---:|---:|",
            ]
        )
        for index, pooled in enumerate(payload["pooled_acf"]):
            channel = payload["mean_channel_acf"][index]
            sequence = payload["mean_sequence_acf"][index]
            lines.append("| %d | %.8f | %.8f | %.8f |" % (index + 1, pooled, channel, sequence))
        lines.append("")
    (output_dir / "electricity_training_residual_acf.md").write_text("\n".join(lines), encoding="utf-8")


@torch.no_grad()
def run_analysis(
    run_dir: Path,
    output_dir: Path,
    checkpoint_mode: int = 1,
    device: str = "auto",
    num_worker: int = 0,
    max_lag: int = MAX_LAG,
    stage1_checkpoint: Optional[Path] = None,
    sigma_eps: float = SIGMA_EPS,
) -> Dict[str, Any]:
    if sigma_eps <= 0:
        raise ValueError("sigma_eps must be positive.")
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    experiment, config, checkpoint_dir, stage1 = load_experiment(
        run_dir=run_dir,
        checkpoint_mode=checkpoint_mode,
        device=device,
        num_worker=num_worker,
        stage1_checkpoint=stage1_checkpoint,
    )
    if str(experiment.dataset_type) != "Electricity":
        raise ValueError("Reviewer analysis is scoped to Electricity, got %r." % experiment.dataset_type)
    if int(experiment.pred_len) <= max_lag:
        raise ValueError("H must exceed max_lag.")

    declared_windows = len(experiment.train_loader.dataset)
    _validate_reproduction_constraints(
        windows=declared_windows,
        horizon=int(experiment.pred_len),
        dimensions=int(experiment.enc_in),
    )
    estimators = {
        "Y": WithinSequenceACF(max_lag),
        "E": WithinSequenceACF(max_lag),
        "Z": WithinSequenceACF(max_lag),
    }
    sigma_count = 0
    sigma_sum = 0.0
    sigma_min = float("inf")
    sigma_max = float("-inf")
    sigma_max_abs_from_one = 0.0
    sigma_clipped_scalar_count = 0
    for batch in tqdm(experiment.train_loader, total=len(experiment.train_loader), desc="training residual ACF"):
        batch_x, batch_y, _, _, batch_x_date, _ = batch
        batch_x = batch_x.to(experiment.device).float()
        batch_y = batch_y.to(experiment.device).float()
        batch_x_date = batch_x_date.to(experiment.device).float()
        _, mu_hat, sigma_hat = experiment.model.get_encoder_features(batch_x, batch_x_date)
        if not torch.isfinite(sigma_hat).all() or torch.any(sigma_hat <= 0):
            raise ValueError("Model produced non-finite or non-positive sigma_hat.")
        sigma_values = sigma_hat.detach().to(dtype=torch.float64)
        sigma_count += int(sigma_values.numel())
        sigma_sum += float(sigma_values.sum().item())
        sigma_min = min(sigma_min, float(sigma_values.min().item()))
        sigma_max = max(sigma_max, float(sigma_values.max().item()))
        sigma_max_abs_from_one = max(
            sigma_max_abs_from_one, float((sigma_values - 1.0).abs().max().item())
        )
        sigma_clipped_scalar_count += int((sigma_hat < sigma_eps).sum().item())
        raw = batch_y - mu_hat
        estimators["Y"].update(batch_y)
        estimators["E"].update(raw)
        estimators["Z"].update(raw / sigma_hat.clamp_min(sigma_eps))

    results = {name: estimator.finalize() for name, estimator in estimators.items()}
    observed_windows = int(results["Y"]["window_count"])
    if (
        observed_windows != declared_windows
        or results["E"]["window_count"] != observed_windows
        or results["Z"]["window_count"] != observed_windows
    ):
        raise RuntimeError("Training loader window count changed while iterating.")
    horizon, dimensions = int(experiment.pred_len), int(experiment.enc_in)
    scalar_count = observed_windows * horizon * dimensions
    expected_pair_counts = _expected_pooled_pair_counts(
        observed_windows, horizon, dimensions, max_lag
    )
    for residual_name, payload in results.items():
        if int(payload["scalar_count"]) != scalar_count:
            raise RuntimeError("%s scalar count is inconsistent with N*H*D." % residual_name)
        if not np.array_equal(payload["pooled_pair_count"], expected_pair_counts):
            raise RuntimeError("%s ACF pairs do not match the strict within-window rule." % residual_name)

    checkpoint_path = checkpoint_dir / "best_model.pth"
    metadata = {
        "dataset": "Electricity",
        "loader": "training",
        "N": observed_windows,
        "H": horizon,
        "D": dimensions,
        "NHD": scalar_count,
        "checkpoint_mode": checkpoint_mode,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "stage1_checkpoint_path": stage1["path"],
        "stage1_checkpoint_sha256": stage1["sha256"],
        "stage1_checkpoint_provenance": stage1["source"],
        "model_scale": "training-loader StandardScaler scale",
        "sigma_eps": float(sigma_eps),
        "sigma_clipped_scalar_count": sigma_clipped_scalar_count,
        "sigma_clipped_scalar_fraction": sigma_clipped_scalar_count / scalar_count,
        "residuals": {
            "Y": "training target",
            "E": "Y - mu_hat",
            "Z": "(Y - mu_hat) / max(sigma_hat, sigma_eps)",
        },
        "sigma_summary": {
            "count": sigma_count,
            "mean": sigma_sum / sigma_count,
            "min": sigma_min,
            "max": sigma_max,
            "max_abs_from_one": sigma_max_abs_from_one,
            "z_equals_e_exactly": sigma_max_abs_from_one == 0.0,
        },
        "aggregation": {
            "pooled_pearson": "Pearson correlation across all valid matched (n,d,t,t+lag) pairs.",
            "mean_channel_pearson": "Equal-weight mean of Pearson correlations calculated separately for each channel across all windows and valid horizon pairs; channels with undefined correlation are excluded.",
            "mean_within_window_channel_pearson": "Equal-weight mean of Pearson correlations calculated separately for every (n,d) sequence; constant sequences are excluded.",
            "boundary_rule": "Only pairs with fixed window n and channel d are included; no concatenation across either boundary.",
        },
        "saved_args": config,
    }
    _write_report(output_dir=output_dir, results=results, metadata=metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reviewer-requested Electricity LS-Flow training residual ACF.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run root containing train_mode_1.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-mode", type=int, default=1, choices=(1, 2))
    parser.add_argument("--device", default="auto", help="Torch device, or auto (CUDA when available).")
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--max-lag", type=int, default=MAX_LAG)
    parser.add_argument(
        "--sigma-eps",
        type=float,
        default=SIGMA_EPS,
        help="Additional lower bound applied to sigma_hat when forming the standardized-residual sensitivity analysis.",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        default=None,
        help="Explicit designated iTransformer checkpoint for older runs without stage1_checkpoint.json.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = run_analysis(**vars(args))
    print("Wrote reviewer ACF report: N={N}, H={H}, D={D}, NHD={NHD}".format(**report))
