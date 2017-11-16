#!/usr/bin/env bash
set -e

#DATASET_DIR=/home/zf/opt/drone-scalable-search/processed_dataset/munich/mobilenet_jit_train_4K0G0110
DATASET_DIR=/home/zf/opt/drone-scalable-search/processed_dataset/stanfard_campus/experiments/tiled_mobilenet_classification/train

run() {
    python extract_pre_logit_image_classifier.py \
        --model_name=mobilenet_v1 \
        --batch_size=64 \
        --checkpoint_path=/home/junjuew/mobisys18/pretrained_models/mobilenet_ckpt/mobilenet_v1_1.0_224.ckpt \
        --input_dir=$1 \
        --label=$2 \
        --output_file=$3
}

run ${DATASET_DIR}/photos/positive \
    1 \
    ${DATASET_DIR}/pre_logit_positive.p

run ${DATASET_DIR}/photos/negative \
    0 \
    ${DATASET_DIR}/pre_logit_negative.p