#!/bin/bash
set -e

die() { echo "$@" 1>&2 ; exit 1; }


FINETUNE_ALL_LAYERS_STEPS=5000

# should not use 'exponential' for learning_rate_decay_type, since exponential depends on
# global_step and # of samples
# Fine-tune only the new layers for 2000 steps.

if [[ -d "$ALL_LAYER_TRAIN_DIR" ]]; then
    die "$ALL_LAYER_TRAIN_DIR already exists."
fi
mkdir -p $ALL_LAYER_TRAIN_DIR

# start from the converged last layer model
python finetune_image_classifier.py \
       --train_dir=${ALL_LAYER_TRAIN_DIR} \
       --dataset_name=twoclass \
       --dataset_split_name=train \
       --dataset_dir=${DATASET_DIR} \
       --model_name=${MODEL_NAME} \
       --checkpoint_path=${LAST_LAYER_TRAIN_DIR} \
       --restore_global_step=False \
       --max_number_of_steps=${FINETUNE_ALL_LAYERS_STEPS} \
       --max_gpu_memory_fraction=0.6 \
       --batch_size=32 \
       --learning_rate=0.0001 \
       --learning_rate_decay_type=fixed \
       --save_interval_secs=60 \
       --save_summaries_secs=60 \
       --log_every_n_steps=10 \
       --optimizer=rmsprop \
       --weight_decay=0.00004 2>&1 | tee ${ALL_LAYER_TRAIN_DIR}/train_all_layer.log
