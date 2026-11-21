#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-NsDiff}"
RUN_DIR="${RUN_DIR:-results/runs/iReflow/Electricity/w96h1s192/f3c305bf57f778b8a7794b769027d4f4}"
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-results/runs/iTransformer/electricity_96_192_Electricity_M_ft96_sl48_ll192_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/checkpoint.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-results/analysis/electricity_reviewer_acf_sigma_eps_0p01_lag48}"
DEVICE="${DEVICE:-cuda:6}"
NUM_WORKER="${NUM_WORKER:-0}"
MAX_LAG="${MAX_LAG:-48}"
SIGMA_EPS="${SIGMA_EPS:-0.01}"

if [[ ! -f "${RUN_DIR}/train_mode_1/best_model.pth" ]]; then
    echo "Missing LS-Flow checkpoint: ${RUN_DIR}/train_mode_1/best_model.pth" >&2
    exit 1
fi
if [[ ! -f "${STAGE1_CHECKPOINT}" ]]; then
    echo "Missing designated stage-1 checkpoint: ${STAGE1_CHECKPOINT}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

conda run --no-capture-output -n "${CONDA_ENV}" python -u src/analysis/reviewer_acf.py \
    --run-dir "${RUN_DIR}" \
    --checkpoint-mode 1 \
    --stage1-checkpoint "${STAGE1_CHECKPOINT}" \
    --device "${DEVICE}" \
    --num-worker "${NUM_WORKER}" \
    --max-lag "${MAX_LAG}" \
    --sigma-eps "${SIGMA_EPS}" \
    --output-dir "${OUTPUT_DIR}"
