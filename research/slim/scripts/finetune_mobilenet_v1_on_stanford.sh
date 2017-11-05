#!/bin/bash
set -e

die() { echo "$@" 1>&2 ; exit 1; }

# Where the pre-trained mobilenet checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR="/home/junjuew/mobisys18/pretrained_models/mobilenet_ckpt"

MODEL_NAME=mobilenet_v1

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR="/home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/train/logs"

# Where the dataset is saved to.
DATASET_DIR="/home/junjuew/mobisys18/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/train"

if [[ -d "$TRAIN_DIR" ]]; then
    die "$TRAIN_DIR already exists."
fi

mkdir -p $TRAIN_DIR


# should not use 'exponential' for learning_rate_decay_type, since exponential depends on
# global_step and # of samples
# Fine-tune only the new layers for 2000 steps.
python finetune_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=twoclass \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/mobilenet_v1_1.0_224.ckpt \
  --checkpoint_exclude_scopes=MobilenetV1/Logits \
  --restore_global_step=False \  
  --trainable_scopes=MobilenetV1/Logits \
  --max_number_of_steps=20000 \
  --batch_size=32 \
  --max_gpu_memory_fraction=1 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004 2>&1 | tee ${TRAIN_DIR}/train_last_layer.log

python finetune_image_classifier.py \
       --train_dir=${TRAIN_DIR}/all \
       --dataset_name=twoclass \
       --dataset_split_name=train \
       --dataset_dir=${DATASET_DIR} \
       --model_name=${MODEL_NAME} \
       --checkpoint_path=${TRAIN_DIR} \
       --max_number_of_steps=20000 \
       --max_gpu_memory_fraction=1 \
       --batch_size=32 \
       --learning_rate=0.0001 \
       --learning_rate_decay_type=fixed \
       --save_interval_secs=60 \
       --save_summaries_secs=60 \
       --log_every_n_steps=10 \
       --optimizer=rmsprop \
       --weight_decay=0.00004 &> ${TRAIN_DIR}/train_all_layer_${cur_step}.log
