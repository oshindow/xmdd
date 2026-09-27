#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=${HF_HOME:-/home/xintong/whisper_pinyin/.cache-qwen3/huggingface}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/qwen3-aut-kmeans-mpl}
export PYTHONUNBUFFERED=1

/home/xintong/miniconda3/envs/qwen3_omni/bin/python \
  scripts/train_qwen3_aut_kmeans_infonce.py \
  --config config/qwen3_aut_kmeans_infonce_30k.yaml
