#!/bin/bash
set -ex
die() { echo "$@" 1>&2 ; exit 1; }

MODEL_NAME=mobilenet_v1
if [[ -f scripts/finetune_mobilenet_v1_on_twoclass.shrc ]]; then
    source scripts/finetune_mobilenet_v1_on_twoclass.shrc
fi

: "${TRAIN_LAST_LAYER:?Need to set TRAIN_LAST_LAYER non-empty}"
: "${TRAIN_ALL_LAYER:?Need to set TRAIN_ALL_LAYER non-empty}"
: "${PRETRAINED_CHECKPOINT_DIR:?Need to set PRETRAINED_CHECKPOINT_DIR non-empty}"
: "${LAST_LAYER_TRAIN_DIR:?Need to set LAST_LAYER_TRAIN_DIR non-empty}"
: "${ALL_LAYER_TRAIN_DIR:?Need to set ALL_LAYER_TRAIN_DIR non-empty}"
: "${DATASET_DIR:?Need to set DATASET_DIR non-empty}"
: "${LAST_LAYER_MAX_STEP:?Need to set LAST_LAYER_MAX_STEP non-empty}"
: "${ALL_LAYER_MAX_STEP:?Need to set ALL_LAYER_MAX_STEP non-empty}"
: "${MAX_GPU_MEMORY_USAGE:?Need to set MAX_GPU_MEMORY_USAGE non-empty}"

if [[ "$TRAIN_LAST_LAYER" == "true" ]]; then
    if [[ -d "$LAST_LAYER_TRAIN_DIR" ]]; then
        die "$LAST_LAYER_TRAIN_DIR already exists."
    fi
    mkdir -p $LAST_LAYER_TRAIN_DIR
fi

if [[ "$TRAIN_ALL_LAYER" == "true" ]]; then
    if [[ -d "$ALL_LAYER_TRAIN_DIR" ]]; then
        die "$ALL_LAYER_TRAIN_DIR already exists."
    fi
    mkdir -p $ALL_LAYER_TRAIN_DIR
fi

# should not use 'exponential' for learning_rate_decay_type, since exponential
# depends on global_step and # of samples
if [[ "$TRAIN_LAST_LAYER" == "true" ]]; then
    python finetune_image_classifier.py \
           --train_dir=${LAST_LAYER_TRAIN_DIR} \
           --dataset_name=twoclass \
           --dataset_split_name=train \
           --dataset_dir=${DATASET_DIR} \
           --model_name=${MODEL_NAME} \
           --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/mobilenet_v1_1.0_224.ckpt \
           --checkpoint_exclude_scopes=MobilenetV1/Logits \
           --restore_global_step=False \
           --trainable_scopes=MobilenetV1/Logits \
           --max_number_of_steps=${LAST_LAYER_MAX_STEP} \
           --batch_size=32 \
           --max_gpu_memory_fraction=${MAX_GPU_MEMORY_USAGE} \
           --learning_rate=0.001 \
           --learning_rate_decay_type=fixed \
           --save_interval_secs=60 \
           --save_summaries_secs=60 \
           --log_every_n_steps=10 \
           --optimizer=rmsprop \
           --weight_decay=0.00004 2>&1 | tee ${LAST_LAYER_TRAIN_DIR}/train_last_layer.log
fi

if [[ "$TRAIN_ALL_LAYER" == "true" ]]; then
    python finetune_image_classifier.py \
           --train_dir=${ALL_LAYER_TRAIN_DIR} \
           --dataset_name=twoclass \
           --dataset_split_name=train \
           --dataset_dir=${DATASET_DIR} \
           --model_name=${MODEL_NAME} \
           --checkpoint_path=${LAST_LAYER_TRAIN_DIR} \
           --restore_global_step=True \
           --max_number_of_steps=${ALL_LAYER_MAX_STEP} \
           --max_gpu_memory_fraction=${MAX_GPU_MEMORY_USAGE} \
           --batch_size=32 \
           --learning_rate=0.001 \
           --learning_rate_decay_type=fixed \
           --save_interval_secs=60 \
           --save_summaries_secs=60 \
           --log_every_n_steps=10 \
           --optimizer=rmsprop \
           --weight_decay=0.00004 2>&1 | tee ${ALL_LAYER_TRAIN_DIR}/train_all_layer.log
fi
