#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.cache-qwen3/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/qwen3-zh-en-error-aware-lm-mpl}"
export TOKENIZERS_PARALLELISM=false

exec /home/xintong/miniconda3/envs/qwen3_omni/bin/python \
  "$SCRIPT_DIR/scripts/train_qwen3_multilingual_bpe_icl.py" \
  --config "$SCRIPT_DIR/config/qwen3_zh_en_bpe_icl12_error_aware_lm_5k_2gpu.yaml" "$@"
