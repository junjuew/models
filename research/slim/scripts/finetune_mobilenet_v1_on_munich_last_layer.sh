#!/bin/bash
set -e

die() { echo "$@" 1>&2 ; exit 1; }

if [[ -d "$LAST_LAYER_TRAIN_DIR" ]]; then
    die "$LAST_LAYER_TRAIN_DIR already exists."
fi
mkdir -p $LAST_LAYER_TRAIN_DIR

echo "Finetuning last layer for ${FINETUNE_LAST_LAYER_STEPS:=15000} steps."

# should not use 'exponential' for learning_rate_decay_type, since exponential depends on
# global_step and # of samples
# Fine-tune only the new layers for 2000 steps.

python finetune_image_classifier.py \
  --train_dir=${LAST_LAYER_TRAIN_DIR} \
  --dataset_name=twoclass \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR} \
  --checkpoint_exclude_scopes=MobilenetV1/Logits \
  --restore_global_step=False \
  --trainable_scopes=MobilenetV1/Logits \
  --max_number_of_steps=${FINETUNE_LAST_LAYER_STEPS} \
  --batch_size=32 \
  --max_gpu_memory_fraction=0.6 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004 2>&1 | tee ${LAST_LAYER_TRAIN_DIR}/train_last_layer.log

