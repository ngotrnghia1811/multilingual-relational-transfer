#!/bin/bash
# ZSCL-R: Zero-shot Cross-lingual Relational-transfer
# GrDA adversarial training with multi-lingual unlabeled data + language graph.
#
# Usage:
#   bash scripts/train_zsclr.sh [minion|smiler] [model_size]
#
# Examples:
#   bash scripts/train_zsclr.sh minion base
#   bash scripts/train_zsclr.sh smiler large

set -e

DATASET=${1:-minion}
MODEL_SIZE=${2:-base}

case $MODEL_SIZE in
  small) MODEL="nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large" ;;
  base)  MODEL="xlm-roberta-base" ;;
  large) MODEL="xlm-roberta-large" ;;
  *) echo "Unknown model size: $MODEL_SIZE (use small|base|large)"; exit 1 ;;
esac

CONFIG="configs/${DATASET}_zsclr.yaml"
OUTPUT_DIR="checkpoints/${DATASET}_zsclr/${MODEL_SIZE}"
LOG_FILE="logs/${DATASET}_zsclr/${MODEL_SIZE}.log"
mkdir -p "$(dirname $LOG_FILE)"

echo "=== ZSCL-R: dataset=${DATASET} model=${MODEL_SIZE} ==="

python train.py \
  --config "$CONFIG" \
  --transfer_mode zsclr \
  --bert_model_name "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_FILE"

echo "Done. Results in $OUTPUT_DIR/results.json"
