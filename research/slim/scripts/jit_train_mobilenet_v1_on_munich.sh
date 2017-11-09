#!/usr/bin/env bash
set -e

die() { echo "$@" 1>&2 ; exit 1; }

if [[ -z $1 ]]; then
    die "Usage: $0 TEST_IMAGE_ID (such as 4K0G0150)"
fi

PICK=$1

echo ""
echo "PICK=${PICK}"
echo ""

DIR=$(dirname $0)

source ${DIR}/munichrc.sh

# replace train directories
PRETRAINED_CHECKPOINT_DIR=${ALL_LAYER_TRAIN_DIR}
TRAIN_DIR="/home/zf/opt/drone-scalable-search/processed_dataset/munich/mobilenet_jit_train_${PICK}/logs"
ALL_LAYER_TRAIN_DIR="${TRAIN_DIR}/logs_finetune_all_layers"
LAST_LAYER_TRAIN_DIR="${TRAIN_DIR}/logs_finetune_last_layer_only"

echo ""
echo "JIT fine tune last layer"
echo ""

FINETUNE_LAST_LAYER_STEPS=5000
source ${DIR}/finetune_mobilenet_v1_on_munich_last_layer.sh
