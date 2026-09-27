#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/xintong/miniconda3/envs/qwen3_omni/bin/python}"
DATA_ROOT=/data2/xintong/mandarin_accent
SPLIT_DIR=data/jsonl_actual_split
MFA_TEXTGRIDS=/data2/xintong/checkpoints/mfa/target_alignments
MFA_INDEX=/data2/xintong/checkpoints/mfa/target_alignment_index.jsonl
EXP_DIR=/data2/xintong/checkpoints/qwen3_asr_ctc_actual_tonecontrast_mfa
export PYTHONUNBUFFERED=1

if [[ ! -s "$MFA_INDEX" ]]; then
  "$PYTHON_BIN" scripts/mfa/prepare_alignment_index.py \
    --manifest "$SPLIT_DIR/train.jsonl" \
    --textgrid-root "$MFA_TEXTGRIDS" \
    --output "$MFA_INDEX"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
"$PYTHON_BIN" scripts/baseline/finetuning_pinyin_ctc_qwen3.py \
  --data-root "$DATA_ROOT" \
  --train-path "$SPLIT_DIR/train.jsonl" \
  --val-path "$SPLIT_DIR/dev.jsonl" \
  --token-table-path data/lang_jsonl/tokens.txt \
  --model-name Qwen/Qwen3-ASR-0.6B-hf \
  --ctc-vocab 188 \
  --exp-dir "$EXP_DIR" \
  --batch-size 4 \
  --gradient-accumulation-steps 3 \
  --precision 16-mixed \
  --learning-rate 5e-5 \
  --adam-epsilon 1e-8 \
  --max-steps 30000 \
  --freeze-backbone-steps 10000 \
  --backbone-ramp-steps 3000 \
  --head-warmup-steps 3000 \
  --eval-steps 1000 \
  --mask-time-prob 0.65 \
  --mask-time-length 10 \
  --mask-feature-prob 0.25 \
  --mask-feature-length 64 \
  --tone-contrastive \
  --tone-alignment mfa \
  --mfa-alignment-index "$MFA_INDEX" \
  --tone-start-step 10000 \
  --tone-ramp-steps 2000 \
  --tone-loss-weight 0.05 \
  --tone-temperature 0.1 \
  --seed 42 \
  --devices 4 \
  --strategy ddp
