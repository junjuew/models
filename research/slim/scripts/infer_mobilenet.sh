#!/bin/bash
set -ex

die() { echo "$@" 1>&2 ; exit 1; }

MODEL_NAME=mobilenet_v1

# Where the training (fine-tuned) checkpoint and logs will be saved to.
# train_dir is input from user
if [ $# -lt 4 ]; then
    die "Invoke the scripts as infer_mobilenet.sh dataset_dir checkpoint_dir test_image_dir redis_db"
fi

# Where the dataset is saved to.
DATASET_DIR=$1
CHECKPOINT_DIR=$2
IMAGE_DIR=$3
REDIS_DB=$4

if [ ! -d "$CHECKPOINT_DIR" ]; then
    die "directory $CHECKPOINT_DIR does not exist!"
fi

if [ ! -d "$IMAGE_DIR" ]; then
    die "directory $IMAGE_DIR does not exist!"
fi

echo "launching inference using models at ${CHECKPOINT_DIR}"
# Run evaluation on training data
python infer_image_classifier.py \
       --input_dir=${IMAGE_DIR} \
       --checkpoint_path=${CHECKPOINT_DIR} \
       --dataset_name=twoclass \
       --dataset_dir=${DATASET_DIR} \
       --model_name=${MODEL_NAME} \
       --redis_db=${REDIS_DB}
