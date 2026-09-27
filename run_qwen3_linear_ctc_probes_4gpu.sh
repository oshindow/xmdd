#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/xintong/miniconda3/envs/qwen3_omni/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data2/xintong/mandarin_accent}"
SPLIT_DIR="${SPLIT_DIR:-data/jsonl_actual_split}"
EXP_ROOT="${EXP_ROOT:-/data2/xintong/checkpoints/qwen3_asr_0.6b_linear_ctc_probes_lr5e5_bs4_acc3}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.cache-qwen3/huggingface}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/qwen3-linear-probe-matplotlib}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python: $PYTHON_BIN (set PYTHON_BIN to the Qwen3 environment)" >&2
  exit 1
fi
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
if [[ "${#GPUS[@]}" -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four comma-separated GPU IDs; got: $GPU_IDS" >&2
  exit 1
fi
mkdir -p "$EXP_ROOT"

pids=()
cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup INT TERM

run_worker() {
  local worker="$1" gpu="$2" layer exp_dir
  shift 2
  for ((layer=worker + 1; layer<=18; layer+=4)); do
    exp_dir="$EXP_ROOT/layer_$(printf '%02d' "$layer")"
    mkdir -p "$exp_dir"
    if [[ -f "$exp_dir/.done" ]]; then
      echo "[GPU $gpu] layer $layer already complete; skipping"
      continue
    fi
    echo "[GPU $gpu] starting layer $layer at $(date -Is)"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" \
      scripts/baseline/linear_ctc_probe_qwen3.py \
      --layer "$layer" \
      --train-path "$SPLIT_DIR/train.jsonl" \
      --val-path "$SPLIT_DIR/dev.jsonl" \
      --data-root "$DATA_ROOT" \
      --token-table-path data/lang_jsonl/tokens.txt \
      --model-name Qwen/Qwen3-ASR-0.6B-hf \
      --ctc-vocab 188 \
      --exp-dir "$exp_dir" \
      --batch-size 4 \
      --gradient-accumulation-steps 3 \
      --learning-rate 5e-5 \
      --warmup-steps 0 \
      --max-steps 10000 \
      --eval-steps 2000 \
      "$@" 2>&1 | tee "$exp_dir/train.log"
    touch "$exp_dir/.done"
    echo "[GPU $gpu] finished layer $layer at $(date -Is)"
  done
}

for worker in 0 1 2 3; do
  run_worker "$worker" "${GPUS[$worker]}" "$@" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -eq 0 ]]; then
  "$PYTHON_BIN" scripts/plot_qwen3_linear_ctc_probes.py \
    --exp-root "$EXP_ROOT" \
    --output "$EXP_ROOT/per_all_layers_2k_4k_6k_8k_10k.png"
fi
exit "$status"
