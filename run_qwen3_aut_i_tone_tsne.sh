#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME=${HF_HOME:-/home/xintong/whisper_pinyin/.cache-qwen3/huggingface}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/qwen3-aut-i-tone-mpl}
export PYTHONUNBUFFERED=1

BEST_ACTUAL_LM=${BEST_ACTUAL_LM:-/data2/xintong/checkpoints/qwen3_asr_0.6b_actual_lm_freeze_conv_5k/checkpoints/step=004501-per=0.0329.ckpt}
CONTRAST_CKPT=${CONTRAST_CKPT:-/data2/xintong/checkpoints/qwen3_aut_kmeans_infonce_30k/checkpoints/last.ckpt}

/home/xintong/miniconda3/envs/qwen3_omni/bin/python \
  scripts/analyze_qwen3_aut_i_tone_tsne.py \
  --model-name Qwen/Qwen3-ASR-0.6B-hf \
  --local-files-only \
  --checkpoint "$BEST_ACTUAL_LM" \
  --checkpoint-format actual_lm \
  --manifest data/jsonl_actual_split/train.jsonl \
  --data-root /data2/xintong/mandarin_accent \
  --phone-field actual_phones \
  --final i \
  --samples-per-tone 500 \
  --output-dir results/qwen3_aut_i_tone_tsne_best_actual_lm

/home/xintong/miniconda3/envs/qwen3_omni/bin/python \
  scripts/analyze_qwen3_aut_i_tone_tsne.py \
  --model-name Qwen/Qwen3-ASR-0.6B-hf \
  --local-files-only \
  --checkpoint "$CONTRAST_CKPT" \
  --checkpoint-format contrast \
  --manifest data/jsonl_actual_split/train.jsonl \
  --data-root /data2/xintong/mandarin_accent \
  --phone-field actual_phones \
  --final i \
  --samples-per-tone 500 \
  --output-dir results/qwen3_aut_i_tone_tsne_contrast
