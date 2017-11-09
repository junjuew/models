#!/bin/bash
set -ex

die() { echo "$@" 1>&2 ; exit 1; }

MODEL_NAME=mobilenet_v1

# Where the dataset is saved to.
DATASET_DIR="/home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/train"

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR="/home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/train/logs_finetune_all_layers"

# test images
IMAGE_DIR="/home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/test_negative"

echo "launching inference using models at ${TRAIN_DIR}"
# Run evaluation on training data
python infer_image_classifier.py \
       --input_dir=${IMAGE_DIR} \
       --checkpoint_path=${TRAIN_DIR} \
       --eval_dir=/tmp/infer \
       --eval_interval_secs=360 \
       --dataset_name=twoclass \
       --dataset_split_name=train \
       --dataset_dir=${DATASET_DIR} \
       --model_name=${MODEL_NAME}
