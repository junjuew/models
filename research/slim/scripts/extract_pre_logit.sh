#!/usr/bin/env bash

python extract_pre_logit_image_classifier.py \
    --model_name=mobilenet_v1 \
    --batch_size=32 \
    --checkpoint_path=/home/junjuew/mobisys18/pretrained_models/mobilenet_ckpt/mobilenet_v1_1.0_224.ckpt \
    --dataset_name=twoclass \
    --dataset_split=train \
    --dataset_dir=/home/zf/opt/drone-scalable-search/processed_dataset/munich/mobilenet_train \
    --input_dir=/home/zf/opt/drone-scalable-search/processed_dataset/munich/mobilenet_train/photos/positive