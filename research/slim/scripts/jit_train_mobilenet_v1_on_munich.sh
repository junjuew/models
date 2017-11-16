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

JIT_DATASET_DIR="/home/zf/opt/drone-scalable-search/processed_dataset/munich/mobilenet_jit_train_${PICK}"
JIT_TRAIN_DIR="${JIT_DATASET_DIR}/logs"
JIT_LAST_LAYER_TRAIN_DIR="${JIT_TRAIN_DIR}/logs_finetune_last_layer_only"

# switch to LAST_LAYER_TRAIN_DIR if wanna start from last-layer-trained model
PRETRAINED_CHECKPOINT_DIR=${ALL_LAYER_TRAIN_DIR}

if [[ -d "$JIT_LAST_LAYER_TRAIN_DIR" ]]; then
    die "$JIT_LAST_LAYER_TRAIN_DIR already exists."
fi
mkdir -p ${JIT_LAST_LAYER_TRAIN_DIR}

echo ""
echo "JIT fine tune last layer"
echo ""
echo "Finetuning last layer for ${JIT_FINETUNE_LAST_LAYER_STEPS:=5000} steps."
echo "starting from model at ${PRETRAINED_CHECKPOINT_DIR}"
echo "using dataset at ${JIT_DATASET_DIR}"
echo ""

# IMPORTANT: remove --checkpoint_exclude_scopes=MobilenetV1/Logits to retain pre-trained last layer

python finetune_image_classifier.py \
  --train_dir=${JIT_LAST_LAYER_TRAIN_DIR} \
  --dataset_name=twoclass \
  --dataset_split_name=train \
  --dataset_dir=${JIT_DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR} \
  --restore_global_step=False \
  --trainable_scopes=MobilenetV1/Logits \
  --max_number_of_steps=${JIT_FINETUNE_LAST_LAYER_STEPS} \
  --batch_size=32 \
  --max_gpu_memory_fraction=0.6 \
  --learning_rate=0.00001 \
  --learning_rate_decay_type=exponential \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004 2>&1 | tee ${JIT_LAST_LAYER_TRAIN_DIR}/train_last_layer.log