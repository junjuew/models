#!/usr/bin/env bash

BASE_DIR=/home/zf/opt/drone-scalable-search/processed_dataset/stanford_campus/
TEST_VIDEO=hyang_video4

if true
then
    python prepare_jit_train_data.py make_tp_fp_dataset_2 \
        --tile_classification_annotation_file ${BASE_DIR}/classification_448_224_224_224_annotations/${TEST_VIDEO}.pkl \
        --tile_test_inference_file ${BASE_DIR}/experiments/classification_448_224_224_224/${TEST_VIDEO}_inference_result.pkl \
        --output_file ${BASE_DIR}/experiments/classification_448_224_224_224/${TEST_VIDEO}_tp_fp.p
fi

if true
then
    for tr in 0.05 0.1 0.2 0.5 1.0
    do
        echo ""
        echo "train ratio ${tr}"
        echo ""
        python jit_svm_on_pre_logit.py train \
            --pre_logit_files ${BASE_DIR}/experiments/classification_448_224_224_224/${TEST_VIDEO}_tp_fp.p \
            --test_ratio 0.5 \
            --downsample_train=${tr} \
            --split_pos True \
            --save_model_path ${BASE_DIR}/experiments/classification_448_224_224_224/${TEST_VIDEO}_svm.p | grep "validation accuracy"
    done
fi