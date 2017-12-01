#!/usr/bin/env bash

# also applicable on elephant, raft, okutama

BASE_DIR=/home/zf/opt/drone-scalable-search
DATASET=okutama
TEST_VIDEO=1.1.7



if true
then
    mkdir -p ${BASE_DIR}/processed_dataset/${DATASET}/jitl
    python prepare_jit_train_data.py make_tp_fp_dataset_2 \
        --tile_classification_annotation_file ${BASE_DIR}/processed_dataset/${DATASET}/classification_*_annotations/${TEST_VIDEO}.pkl \
        --tile_test_inference_file ${BASE_DIR}/experiments/jitl/${DATASET}.pkl \
        --output_file ${BASE_DIR}/processed_dataset/${DATASET}/jitl/${TEST_VIDEO}_tp_fp.p
fi

if true
then
    mkdir -p ${BASE_DIR}/processed_dataset/${DATASET}/jitl/log
    for tr in 0.05 0.1 0.2 0.5 1.0
    do
        echo ""
        echo "train ratio ${tr}"
        echo ""
        python jit_svm_on_pre_logit.py train \
            --pre_logit_files ${BASE_DIR}/processed_dataset/${DATASET}/jitl/${TEST_VIDEO}_tp_fp.p \
            --test_ratio 0.5 \
            --downsample_train=${tr} \
            --split_pos True \
            --save_model_path ${BASE_DIR}/processed_dataset/${DATASET}/jitl/${TEST_VIDEO}_svm.p 2>&1
    done | tee ${BASE_DIR}/processed_dataset/${DATASET}/jitl/log/${TEST_VIDEO}.log | grep "test accuracy"  | cut -d: -f2
fi