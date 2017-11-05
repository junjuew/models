#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# Usage:
# cd slim
# ./slim/scripts/finetune_mobilenet_v1_on_munich.sh
set -e

die() { echo "$@" 1>&2 ; exit 1; }

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR="/home/junjuew/mobisys18/pretrained_models/mobilenet_ckpt"

MODEL_NAME=mobilenet_v1
BATCH_SIZE=32

TRAIN_LAST_LAYER_STEPS=20000
TRAIN_ALL_LAYER_STEPS=5000

# Where the training (fine-tuned) checkpoint and logs will be saved to.
# XXX Mobilenet training doesn't update global step
TRAIN_DIR="/home/zf/opt/drone-scalable-search/processed_dataset/munich/mobilenet_train/logs"

# Where the dataset is saved to.
DATASET_DIR="/home/junjuew/mobisys18/processed_dataset/munich/mobilenet_train"

if [[ -d "$TRAIN_DIR" ]]; then
    die "$TRAIN_DIR already exists."
fi
mkdir $TRAIN_DIR

if [[ -n "`pgrep -u $USER -f 'python train_image_classifier'`" ]]; then
    die "User $USR is already running python train_image_classifier.py. YOU SHALL NOT PASS."
fi

echo ""
echo "Start training laster layer in background: ${DATASET_DIR}"
echo ""

# Fine-tune only the new layers for TRAIN_LAST_LAYER_STEPS steps.
(python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=munich \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/mobilenet_v1_1.0_224.ckpt \
  --checkpoint_exclude_scopes=MobilenetV1/Logits \
  --trainable_scopes=MobilenetV1/Logits \
  --max_number_of_steps=${TRAIN_LAST_LAYER_STEPS} \
  --batch_size=${BATCH_SIZE} \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004 2>&1| tee ${TRAIN_DIR}/train_last_layer.log) &

echo ""
echo "Start evaluation loop on ${DATASET_DIR}"
echo ""

sleep 10s; python eval_image_classifier_loop.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=munich \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --batch_size=4 \
  --eval_interval_secs=60 \
  --timeout=15 \
  --model_name=${MODEL_NAME} 2>&1| tee ${TRAIN_DIR}/eval_last_layer.log


if [[ -n "`pgrep -u $USER -f 'python train_image_classifier'`" ]]; then
    die "User $USR is already running python train_image_classifier.py. YOU SHALL NOT PASS."
fi

echo ""
echo "Start training all layers in background: ${DATASET_DIR}"
echo ""

# Fine-tune all layers for TRAIN_ALL_LAYER_STEPS steps.
(python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=munich \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=${TRAIN_ALL_LAYER_STEPS} \
  --batch_size=16 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004 2>&1| tee ${TRAIN_DIR}/train_all_layer.log) &

echo ""
echo "Start evaluation loop on ${DATASET_DIR}"
echo ""

sleep 10s; python eval_image_classifier_loop.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=munich \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --batch_size=4 \
  --eval_interval_secs=60 \
  --timeout=30 \
  --model_name=${MODEL_NAME} 2>&1| tee ${TRAIN_DIR}/eval_all_layer.log


sleep 10

# Run evaluation on test data only
TEST_DIR="/home/zf/opt/drone-scalable-search/processed_dataset/munich/mobilenet_test/logs"
# Where the dataset is saved to.
TEST_DATASET_DIR="/home/junjuew/mobisys18/processed_dataset/munich/mobilenet_test"

echo ""
echo "Evaluating on test set at ${TEST_DATASET_DIR}"
echo ""

if [[ -d "$TEST_DIR" ]]; then
    die "$TEST_DIR already exists."
fi
mkdir ${TEST_DIR}
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TEST_DIR} \
  --dataset_name=munich \
  --dataset_split_name=test \
  --dataset_dir=${TEST_DATASET_DIR} \
  --model_name=${MODEL_NAME} 2>&1| tee ${TEST_DIR}/log.txt
