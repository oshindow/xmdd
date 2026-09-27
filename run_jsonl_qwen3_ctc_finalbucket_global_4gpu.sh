#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-/home/xintong/miniconda3/envs/qwen3_omni/bin/python}"
export EXP_DIR="${EXP_DIR:-/data2/xintong/checkpoints/qwen3_asr_ctc_actual_tonecontrast_finalbucket_global}"
exec "$SCRIPT_DIR/run_jsonl_qwen3_ctc_finalbucket_4gpu.sh" \
  --tone-global-candidates "$@"
