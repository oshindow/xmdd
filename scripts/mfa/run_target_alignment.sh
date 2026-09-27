#!/usr/bin/env bash
set -euo pipefail

MFA_DIR=/home/xintong/mfa
MFA_ENV="$MFA_DIR/env"
OUTPUT_DIR=/data2/xintong/checkpoints/mfa
export MFA_ROOT_DIR="$MFA_DIR/root"
export PATH="$MFA_ENV/bin:$PATH"
JOBS="${MFA_JOBS:-32}"

mkdir -p "$OUTPUT_DIR/target_alignments" "$MFA_DIR/status"
touch "$MFA_DIR/status/stage2_train_complete"

"$MFA_ENV/bin/mfa" align \
  "$MFA_DIR/corpus_target" "$MFA_DIR/phone_dictionary.txt" \
  "$OUTPUT_DIR/models/mandarin_custom_phones.zip" \
  "$OUTPUT_DIR/target_alignments" \
  --num_jobs "$JOBS" --no_clean

touch "$MFA_DIR/status/stage3_align_complete"
