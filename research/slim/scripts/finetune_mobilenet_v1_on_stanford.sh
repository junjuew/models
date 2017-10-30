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

# Where the pre-trained mobilenet checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR="/home/junjuew/mobisys18/pretrained_models/mobilenet_ckpt"

MODEL_NAME=mobilenet_v1

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR="/home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/mobilenet_classification/train/logs"

# Where the dataset is saved to.
DATASET_DIR="/home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/mobilenet_classification/train"

TEST_DIR="/home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/mobilenet_classification/test/logs"

TEST_DATASET_DIR="/home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/mobilenet_classification/test"

if [[ -d "$TRAIN_DIR" ]]; then
    die "$TRAIN_DIR already exists."
fi
if [[ -d "$TEST_DIR" ]]; then
    die "$TEST_DIR already exists."
fi

mkdir $TRAIN_DIR
mkdir $TEST_DIR

ckpt_step_size=2000
echo "ckpt_step_size: $ckpt_step_size"
# Fine-tune only the new layers for 2000 steps.
for ckpt_step in $(seq 1 10); do
    cur_step=$(($ckpt_step * $ckpt_step_size))
    prev_step=$((($ckpt_step-1) * $ckpt_step_size))
    echo "staring training for step $cur_step"
    echo "prev_step $prev_step"
    if [ "$ckpt_step" -eq "1" ]; then
        echo "picking up pretrained model first ..."
        python train_image_classifier.py \
               --train_dir=${TRAIN_DIR}/${cur_step} \
               --dataset_name=munich \
               --dataset_split_name=train \
               --dataset_dir=${DATASET_DIR} \
               --model_name=${MODEL_NAME} \
               --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/mobilenet_v1_1.0_224.ckpt \
               --checkpoint_exclude_scopes=MobilenetV1/Logits \
               --trainable_scopes=MobilenetV1/Logits \
               --max_number_of_steps=$ckpt_step_size \
               --batch_size=32 \
               --learning_rate=0.01 \
               --learning_rate_decay_type=fixed \
               --save_interval_secs=360000 \
               --save_summaries_secs=360000 \
a               --log_every_n_steps=10 \
               --optimizer=rmsprop \
               --weight_decay=0.00004 2>&1| tee ${TRAIN_DIR}/train_last_layer_${cur_step}.log
    else
        python train_image_classifier.py \
               --train_dir=${TRAIN_DIR}/${cur_step} \
               --dataset_name=munich \
               --dataset_split_name=train \
               --dataset_dir=${DATASET_DIR} \
               --model_name=${MODEL_NAME} \
               --checkpoint_path=${TRAIN_DIR}/${prev_step} \
               --checkpoint_exclude_scopes=MobilenetV1/Logits \
               --trainable_scopes=MobilenetV1/Logits \
               --max_number_of_steps=$ckpt_step_size \
               --batch_size=32 \
               --learning_rate=0.01 \
               --learning_rate_decay_type=fixed \
               --save_interval_secs=360000 \
               --save_summaries_secs=360000 \
               --log_every_n_steps=10 \
               --optimizer=rmsprop \
               --weight_decay=0.00004 2>&1| tee ${TRAIN_DIR}/train_last_layer_${cur_step}.log
    fi


    # # Run evaluation.
    python eval_image_classifier.py \
           --checkpoint_path=${TRAIN_DIR}/${cur_step} \
           --eval_dir=${TRAIN_DIR} \
           --dataset_name=munich \
           --dataset_split_name=validation \
           --dataset_dir=${DATASET_DIR} \
           --model_name=${MODEL_NAME} 2>&1| tee ${TRAIN_DIR}/eval_last_layer_${cur_step}.log

    # Run evaluation on test data only
    python eval_image_classifier.py \
           --checkpoint_path=${TRAIN_DIR}/${cur_step} \
           --eval_dir=${TEST_DIR} \
           --dataset_name=munich \
           --dataset_split_name=test \
           --dataset_dir=${TEST_DATASET_DIR} \
           --model_name=${MODEL_NAME} 2>&1| tee ${TEST_DIR}/test_${cur_step}.log
done

base_step=$cur_step
ckpt_step_size=1000
for ckpt_step in $(seq 1 10); do
    cur_step=$(($ckpt_step * $ckpt_step_size + $base_step))
    prev_step=$(($cur_step - $ckpt_step_size))

    echo "staring training for step $cur_step"
    if [ "$ckpt_step" -eq "1" ]; then
        echo "start fintuning the entire network..."
        python train_image_classifier.py \
               --train_dir=${TRAIN_DIR}/all_${cur_step} \
               --dataset_name=munich \
               --dataset_split_name=train \
               --dataset_dir=${DATASET_DIR} \
               --model_name=${MODEL_NAME} \
               --checkpoint_path=${TRAIN_DIR}/${prev_step} \
               --max_number_of_steps=$ckpt_step_size \
               --batch_size=32 \
               --learning_rate=0.0001 \
               --learning_rate_decay_type=fixed \
               --save_interval_secs=360000 \
               --save_summaries_secs=360000 \
               --log_every_n_steps=10 \
               --optimizer=rmsprop \
               --weight_decay=0.00004 2>&1| tee ${TRAIN_DIR}/train_all_layer_${cur_step}.log
    else
        python train_image_classifier.py \
               --train_dir=${TRAIN_DIR}/all_${cur_step} \
               --dataset_name=munich \
               --dataset_split_name=train \
               --dataset_dir=${DATASET_DIR} \
               --model_name=${MODEL_NAME} \
               --checkpoint_path=${TRAIN_DIR}/all_${prev_step} \
               --max_number_of_steps=$ckpt_step_size \
               --batch_size=32 \
               --learning_rate=0.0001 \
               --learning_rate_decay_type=fixed \
               --save_interval_secs=360000 \
               --save_summaries_secs=360000 \
               --log_every_n_steps=10 \
               --optimizer=rmsprop \
               --weight_decay=0.00004 2>&1| tee ${TRAIN_DIR}/train_all_layer_${cur_step}.log
    fi


    # Run evaluation.
    python eval_image_classifier.py \
           --checkpoint_path=${TRAIN_DIR}/all_${cur_step} \
           --eval_dir=${TRAIN_DIR}/all \
           --dataset_name=munich \
           --dataset_split_name=validation \
           --dataset_dir=${DATASET_DIR} \
           --model_name=${MODEL_NAME} 2>&1| tee ${TRAIN_DIR}/eval_all_layer_${cur_step}.log

    # Run evaluation on test data only
    python eval_image_classifier.py \
           --checkpoint_path=${TRAIN_DIR}/all_${cur_step} \
           --eval_dir=${TEST_DIR} \
           --dataset_name=munich \
           --dataset_split_name=test \
           --dataset_dir=${TEST_DATASET_DIR} \
           --model_name=${MODEL_NAME} 2>&1| tee ${TEST_DIR}/test_${cur_step}.log
done


# rename model file
    # if [ "$ckpt_step_size" != "$cur_step" ]; then
    #     for modelfile in `ls -1 ${TRAIN_DIR}/model.ckpt-${ckpt_step_size}*`; do
    #         suffix=`echo $modelfile | sed "s/^.*model.ckpt-${ckpt_step_size}\(.*\)$/\1/"`
    #         newname=${TRAIN_DIR}/model.ckpt-${cur_step}${suffix}
    #         echo "rename $modelfile -> $newname"
    #         mv -vn "$modelfile" "$newname"
    #     done
    # fi
