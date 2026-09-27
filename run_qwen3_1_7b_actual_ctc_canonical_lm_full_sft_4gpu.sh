#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG="${CONFIG:-$SCRIPT_DIR/config/qwen3_1.7b_actual_ctc_canonical_lm_full_sft.yaml}"
exec "$SCRIPT_DIR/run_qwen3_actual_ctc_canonical_lm_full_sft_4gpu.sh" "$@"
