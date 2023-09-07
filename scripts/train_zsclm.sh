#!/bin/bash
# ZSCL-M: Zero-shot Cross-lingual Multi-transfer
# Fine-tune on cluster-selected source languages; evaluate on all target languages.
#
# Usage:
#   bash scripts/train_zsclm.sh [minion|smiler] [inter|intra|random] [model_size]
#
# Examples:
#   bash scripts/train_zsclm.sh minion inter base
#   bash scripts/train_zsclm.sh smiler inter large

set -e

DATASET=${1:-minion}
CONFIG_TYPE=${2:-inter}  # inter = medoids*, intra = within-cluster, random = baseline
MODEL_SIZE=${3:-base}

case $MODEL_SIZE in
  small) MODEL="nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large" ;;
  base)  MODEL="xlm-roberta-base" ;;
  large) MODEL="xlm-roberta-large" ;;
  *) echo "Unknown model size: $MODEL_SIZE (use small|base|large)"; exit 1 ;;
esac

CONFIG="configs/${DATASET}_zsclm.yaml"
OUTPUT_DIR="checkpoints/${DATASET}_zsclm/${CONFIG_TYPE}_${MODEL_SIZE}"
LOG_FILE="logs/${DATASET}_zsclm/${CONFIG_TYPE}_${MODEL_SIZE}.log"
mkdir -p "$(dirname $LOG_FILE)"

echo "=== ZSCL-M: dataset=${DATASET} config=${CONFIG_TYPE} model=${MODEL_SIZE} ==="

python train.py \
  --config "$CONFIG" \
  --transfer_mode zsclm \
  --selection_mode "$CONFIG_TYPE" \
  --bert_model_name "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_FILE"

echo "Done. Results in $OUTPUT_DIR/results.json"
