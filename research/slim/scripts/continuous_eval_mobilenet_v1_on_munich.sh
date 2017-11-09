#!/bin/bash
set -e

DIR=$(dirname $0)

source ${DIR}/munichrc.sh

### For evaluating JIT training
if [[ -n $1 ]]; then
    PICK=$1
    DATASET_DIR="/home/zf/opt/drone-scalable-search/processed_dataset/munich/mobilenet_jit_train_${PICK}"
    EVAL_FROM_DIR=${DATASET_DIR}/logs/logs_finetune_last_layer_only
fi
###

die() { echo "$@" 1>&2 ; exit 1; }

# Where the training (fine-tuned) checkpoint and logs will be saved to.
#EVAL_FROM_DIR="/home/zf/opt/drone-scalable-search/processed_dataset/munich/mobilenet_train/logs/logs_finetune_last_layer_only"
#EVAL_FROM_DIR="/home/zf/opt/drone-scalable-search/processed_dataset/munich/mobilenet_train/logs/logs_finetune_all_layers"

echo ""
echo "Launching three evaluation loops on models stored at ${EVAL_FROM_DIR}"
echo "using train/validation set at ${DATASET_DIR}"
echo "using test set at ${TEST_DATASET_DIR}"
echo ""

echo "launching eval on training data"
# Run evaluation on training data
python continuous_eval_image_classifier.py \
       --batch_size=100 \
       --checkpoint_path=${EVAL_FROM_DIR} \
       --eval_dir=${EVAL_FROM_DIR}/eval_train \
       --eval_interval_secs=60 \
       --dataset_name=twoclass \
       --dataset_split_name=train \
       --dataset_dir=${DATASET_DIR} \
       --max_gpu_memory_fraction=0.1 \
       --model_name=${MODEL_NAME} &> ${EVAL_FROM_DIR}/continuous_eval_train.log &
BGPIDS=$!

sleep 1

echo "launching eval on validation data"
# Run evaluation on validation data
python continuous_eval_image_classifier.py \
       --batch_size=100 \
       --checkpoint_path=${EVAL_FROM_DIR} \
       --eval_dir=${EVAL_FROM_DIR}/eval_validation \
       --eval_interval_secs=60 \
       --dataset_name=twoclass \
       --dataset_split_name=validation \
       --dataset_dir=${DATASET_DIR} \
       --max_gpu_memory_fraction=0.1 \
       --model_name=${MODEL_NAME} &> ${EVAL_FROM_DIR}/continuous_eval_validation.log &
BGPIDS="$BGPIDS $!"

sleep 1

echo "launching eval on test data"
# Run evaluation on test data
python continuous_eval_image_classifier.py \
       --batch_size=100 \
       --checkpoint_path=${EVAL_FROM_DIR} \
       --eval_dir=${EVAL_FROM_DIR}/eval_test \
       --eval_interval_secs=60 \
       --dataset_name=twoclass \
       --dataset_split_name=test \
       --dataset_dir=${TEST_DATASET_DIR} \
       --max_gpu_memory_fraction=0.1 \
       --model_name=${MODEL_NAME} &> ${EVAL_FROM_DIR}/continuous_eval_test.log &
BGPIDS="$BGPIDS $!"

echo "background process ids: $BGPIDS"
trap 'kill $BGPIDS; echo exiting; exit' SIGINT
wait
