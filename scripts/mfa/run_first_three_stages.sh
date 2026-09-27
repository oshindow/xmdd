#!/usr/bin/env bash
set -euo pipefail

MFA_DIR=/home/xintong/mfa
MFA_ENV="$MFA_DIR/env"
OUTPUT_DIR=/data2/xintong/checkpoints/mfa
MFA_ROOT_DIR="$MFA_DIR/root"
export MFA_ROOT_DIR
export PATH="$MFA_ENV/bin:$PATH"

REPO=/home/xintong/whisper_pinyin
JOBS="${MFA_JOBS:-32}"

"$MFA_ENV/bin/python" "$REPO/scripts/mfa/prepare_mfa_corpora.py" \
  --tokens "$REPO/data/lang_jsonl/tokens.txt" \
  --output-root "$MFA_DIR" \
  --aishell1-root /data2/xiaolong/AISHELL-1/data_aishell \
  --aishell3-root /data2/xintong/mandarin_accent/aishell3 \
  --target-manifest "$REPO/data/jsonl_actual_split/train.jsonl"

"$MFA_ENV/bin/mfa" validate \
  "$MFA_DIR/corpus_train" "$MFA_DIR/phone_dictionary.txt" \
  --output_directory "$MFA_DIR/validation" --num_jobs "$JOBS"

"$MFA_ENV/bin/mfa" train \
  "$MFA_DIR/corpus_train" "$MFA_DIR/phone_dictionary.txt" \
  "$OUTPUT_DIR/models/mandarin_custom_phones.zip" \
  --output_directory "$OUTPUT_DIR/train_alignments" --num_jobs "$JOBS"

"$MFA_ENV/bin/mfa" align \
  "$MFA_DIR/corpus_target" "$MFA_DIR/phone_dictionary.txt" \
  "$OUTPUT_DIR/models/mandarin_custom_phones.zip" "$OUTPUT_DIR/target_alignments" \
  --num_jobs "$JOBS"
