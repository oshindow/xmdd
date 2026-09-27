#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-/home/xintong/miniconda3/envs/qwen3_omni/bin/python}"
export EXP_DIR="${EXP_DIR:-/data2/xintong/checkpoints/qwen3_asr_otc_finalbucket}"
NVIDIA_LIB="/home/xintong/miniconda3/envs/qwen3_omni/lib/python3.11/site-packages/nvidia"
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_nvrtc/lib:$NVIDIA_LIB/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
exec "$SCRIPT_DIR/run_jsonl_qwen3_ctc_finalbucket_4gpu.sh" --otc "$@"
