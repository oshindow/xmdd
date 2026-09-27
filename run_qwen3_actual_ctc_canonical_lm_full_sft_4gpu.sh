#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/xintong/miniconda3/envs/qwen3_omni/bin/python}"
CONFIG="${CONFIG:-config/qwen3_actual_ctc_canonical_lm_full_sft.yaml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.cache-qwen3/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/qwen3-joint-sft-matplotlib}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python interpreter: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config: $CONFIG" >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import torch
import flash_attn
from transformers.utils import is_flash_attn_2_available
if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
    raise SystemExit(f"Expected 4 visible CUDA GPUs, got {torch.cuda.device_count()}")
if not is_flash_attn_2_available():
    raise SystemExit("Transformers cannot use FlashAttention 2 in this environment")
print(f"torch={torch.__version__}, CUDA={torch.version.cuda}, "
      f"flash-attn={flash_attn.__version__}, GPUs={torch.cuda.device_count()}")
PY

exec "$PYTHON_BIN" scripts/train_qwen3_actual_ctc_canonical_lm.py \
  --config "$CONFIG" "$@"
