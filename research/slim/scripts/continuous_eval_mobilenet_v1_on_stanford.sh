#!/bin/bash
set -e

die() { echo "$@" 1>&2 ; exit 1; }

MODEL_NAME=mobilenet_v1

# Where the dataset is saved to.
DATASET_DIR="/home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/train"

TEST_DATASET_DIR="/home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/test"

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR="/home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/train/logs"

echo "launching eval on training data"
# Run evaluation on training data
python continuous_eval_image_classifier.py \
       --batch_size=100 \
       --checkpoint_path=${TRAIN_DIR} \
       --eval_dir=${TRAIN_DIR}/eval_train \
       --eval_interval_secs=360 \
       --dataset_name=twoclass \
       --dataset_split_name=train \
       --dataset_dir=${DATASET_DIR} \
       --model_name=${MODEL_NAME} &> ${TRAIN_DIR}/continuous_eval_train.log &
BGPIDS=$!

echo "launching eval on validation data"
# Run evaluation on validation data
python continuous_eval_image_classifier.py \
       --batch_size=100 \
       --checkpoint_path=${TRAIN_DIR} \
       --eval_dir=${TRAIN_DIR}/eval_validation \
       --eval_interval_secs=180 \
       --dataset_name=twoclass \
       --dataset_split_name=validation \
       --dataset_dir=${DATASET_DIR} \
       --model_name=${MODEL_NAME} &> ${TRAIN_DIR}/continuous_eval_validation.log &
BGPIDS="$BGPIDS $!"

echo "launching eval on test data"
# Run evaluation on test data
python continuous_eval_image_classifier.py \
       --batch_size=100 \
       --checkpoint_path=${TRAIN_DIR} \
       --eval_dir=${TRAIN_DIR}/eval_test \
       --eval_interval_secs=180 \
       --dataset_name=twoclass \
       --dataset_split_name=test \
       --dataset_dir=${TEST_DATASET_DIR} \
       --model_name=${MODEL_NAME} &> ${TRAIN_DIR}/continuous_eval_test.log &
BGPIDS="$BGPIDS $!"

echo "background process ids: $BGPIDS"
trap 'kill $BGPIDS; exit' SIGINT
wait
