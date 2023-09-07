#!/bin/bash
# ZSCL-S: Zero-shot Cross-lingual Single-transfer
# Fine-tune on one source language; evaluate on all target languages.
#
# Usage:
#   bash scripts/train_zscls.sh [minion|smiler] [source_lang] [model_size]
#
# Examples:
#   bash scripts/train_zscls.sh minion eng base
#   bash scripts/train_zscls.sh smiler ita large

set -e

DATASET=${1:-minion}
SOURCE=${2:-eng}
MODEL_SIZE=${3:-base}

case $MODEL_SIZE in
  small) MODEL="nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large" ;;
  base)  MODEL="xlm-roberta-base" ;;
  large) MODEL="xlm-roberta-large" ;;
  *) echo "Unknown model size: $MODEL_SIZE (use small|base|large)"; exit 1 ;;
esac

CONFIG="configs/${DATASET}_zscls.yaml"
OUTPUT_DIR="checkpoints/${DATASET}_zscls/${SOURCE}_${MODEL_SIZE}"
LOG_FILE="logs/${DATASET}_zscls/${SOURCE}_${MODEL_SIZE}.log"
mkdir -p "$(dirname $LOG_FILE)"

echo "=== ZSCL-S: dataset=${DATASET} source=${SOURCE} model=${MODEL_SIZE} ==="

python train.py \
  --config "$CONFIG" \
  --source_lang "$SOURCE" \
  --bert_model_name "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_FILE"

echo "Done. Results in $OUTPUT_DIR/results.json"
