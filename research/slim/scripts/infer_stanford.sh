#!/bin/bash
set -ex

die() { echo "$@" 1>&2 ; exit 1; }

MODEL_NAME=mobilenet_v1

# Where the training (fine-tuned) checkpoint and logs will be saved to.
# train_dir is input from user
if [ "$#" -ne "3" ]; then
    die "Invoke the scripts as infer_stanford.sh test_input_dir checkpoint_dir output_dir"
fi

TEST_INPUT_DIR=$1
CHECKPOINT_DIR=$2
OUTPUT_DIR=$3
OUTPUT_ENDPOINT_NAMES="Predictions,AvgPool_1a"
BATCH_SIZE=100

echo "Working on width > height videos"
video_ids=(
    "bookstore_video5"
    "bookstore_video4"
)
echo "${video_ids}"
IMAGE_W=448
IMAGE_H=224
GRID_W=2
GRID_H=1
for VIDEO_ID in "${video_ids[@]}"
do
    RESULT_FILE=${OUTPUT_DIR}/${VIDEO_ID}_inference_result.pkl
    python infer_tile_classifier.py --batch_size=${BATCH_SIZE} --checkpoint_path=${CHECKPOINT_DIR} --model_name=${MODEL_NAME} --input_dir=${TEST_INPUT_DIR} --video_ids=${VIDEO_ID} --output_endpoint_names=${OUTPUT_ENDPOINT_NAMES} --result_file=${RESULT_FILE} --image_w=${IMAGE_W} --image_h=${IMAGE_H} --grid_w=${GRID_W} --grid_h=${GRID_H} --max_gpu_memory_fraction=1
done

echo "Working on width < height videos"
video_ids=(
    "little_video2"
    "hyang_video4"
    "gates_video1"
)
echo "${video_ids}"
IMAGE_W=224
IMAGE_H=448
GRID_W=1
GRID_H=2
for VIDEO_ID in "${video_ids[@]}"
do
    RESULT_FILE=${OUTPUT_DIR}/${VIDEO_ID}_inference_result.pkl
    python infer_tile_classifier.py --batch_size=${BATCH_SIZE} --checkpoint_path=${CHECKPOINT_DIR} --model_name=${MODEL_NAME} --input_dir=${TEST_INPUT_DIR} --video_ids=${VIDEO_ID} --output_endpoint_names=${OUTPUT_ENDPOINT_NAMES} --result_file=${RESULT_FILE} --image_w=${IMAGE_W} --image_h=${IMAGE_H} --grid_w=${GRID_W} --grid_h=${GRID_H} --max_gpu_memory_fraction=1
done
