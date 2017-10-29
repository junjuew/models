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

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR="/home/junjuew/mobisys18/processed_dataset/munich/mobilenet_train/mobilenet_ckpt"

MODEL_NAME=mobilenet_v1

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR="/home/junjuew/mobisys18/processed_dataset/munich/mobilenet_train/train_logs"

# Where the dataset is saved to.
DATASET_DIR="/home/junjuew/mobisys18/processed_dataset/munich/mobilenet_train"

# Fine-tune only the new layers for 2000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=munich \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/mobilenet_v1_1.0_224.ckpt \
  --checkpoint_exclude_scopes=MobilenetV1/Logits \
  --trainable_scopes=MobilenetV1/Logits \
  --max_number_of_steps=2000 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=munich \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME}

# Fine-tune all the new layers for 1000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=munich \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=1000 \
  --batch_size=32 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=munich \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME}


# Run evaluation on test data only
MODEL_NAME=mobilenet_v1
# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR="/home/junjuew/mobisys18/processed_dataset/munich/mobilenet_train/train_logs"
TEST_DIR="/home/junjuew/mobisys18/processed_dataset/munich/mobilenet_test/test_logs"
# Where the dataset is saved to.
DATASET_DIR="/home/junjuew/mobisys18/processed_dataset/munich/mobilenet_test"
if [[ ! -d "$TEST_DIR" ]]; then
  mkdir ${TEST_DIR}
fi
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TEST_DIR} \
  --dataset_name=munich \
  --dataset_split_name=test \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} 2>&1| tee ${TEST_DIR}/log.txt
