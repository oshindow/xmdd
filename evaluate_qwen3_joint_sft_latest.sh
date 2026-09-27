#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/xintong/miniconda3/envs/qwen3_omni/bin/python}"
EXP_DIR="${EXP_DIR:-/data2/xintong/checkpoints/qwen3_asr_0.6b_actual_ctc_canonical_lm_full_sft}"
CHECKPOINT="${CHECKPOINT:-$EXP_DIR/checkpoints/last.ckpt}"
CONFIG="${CONFIG:-$EXP_DIR/resolved_config.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-results/qwen3_joint_sft_step004000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.cache-qwen3/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/qwen3-joint-eval-matplotlib}"
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" scripts/evaluate_qwen3_joint_sft.py \
  --checkpoint "$CHECKPOINT" \
  --config "$CONFIG" \
  --manifests \
    data/mdd_paired_manifest_expanded/test_seen.jsonl \
    data/mdd_paired_manifest_expanded/test_unseen.jsonl \
  --output-dir "$OUTPUT_DIR" \
  --batch-size 16 \
  --workers 4 \
  "$@" 2>&1 | tee "$OUTPUT_DIR.log"
