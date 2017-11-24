#!/usr/bin/env bash

REDIS_HOST=172.17.0.10
TEST_VIDEO=coupa_video1

python prepare_jit_train_data.py make_tp_fp_dataset \
    --redis-host ${REDIS_HOST} \
    --over_sample_ratio=200 \
    --file_glob "/home/zf/opt/drone-scalable-search/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/2_more_test/tile_test_by_label/*/${TEST_VIDEO}*" \
    --output_file /home/zf/opt/drone-scalable-search/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/2_more_test/${TEST_VIDEO}_tp_fp.p

python svm_on_pre_logit.py train \
    --pre_logit_files /home/zf/opt/drone-scalable-search/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/2_more_test/${TEST_VIDEO}_tp_fp.p \
    --test_ratio 0.5 \
    --eval_every_iters=10