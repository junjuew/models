#!/bin/bash
set -e

die() { echo "$@" 1>&2 ; exit 1; }

# default values
BATCH_SIZE=40
MODEL_NAME=mobilenet_v1

while [[ $# -gt 1 ]]
do
key="$1"
case $key in
    -t|--train_dir) # Where the training (fine-tuned) checkpoint and logs are saved to
    TRAIN_DIR="$2"
    shift
    ;;
    -d|--dataset_dir)
    DATASET_DIR="$2" # Where the dataset is saved to.
    shift
    ;;
    -s|--split_name)
    DATA_SPLIT_NAME="$2" # train, validation, or test
    shift
    ;;
    -b|--batch_size)
    BATCH_SIZE="$2"
    shift
    ;;
    *)  # unknown option
    die "Unknown argument ${key}"
    ;;
esac
shift # past argument or value
done

echo "reading checkpoint from dir ${TRAIN_DIR}"
if [ ! -d "$TRAIN_DIR" ]; then
    die "directory $TRAIN_DIR does not exist!"
fi
echo "reading dataset from dir ${DATASET_DIR}"
if [ ! -d "$DATASET_DIR" ]; then
    die "directory $DATASET_DIR does not exist!"
fi
echo "using split $DATA_SPLIT_NAME with batch size $BATCH_SIZE"
echo "launching continous eval"

python continuous_eval_image_classifier.py \
       --batch_size=40 \
       --checkpoint_path=${TRAIN_DIR} \
       --eval_dir=${TRAIN_DIR}/eval_${DATA_SPLIT_NAME} \
       --eval_interval_secs=360 \
       --dataset_name=twoclass \
       --dataset_split_name=${DATA_SPLIT_NAME} \
       --dataset_dir=${DATASET_DIR} \
       --max_gpu_memory_fraction=0.2 \
       --model_name=${MODEL_NAME} &> ${TRAIN_DIR}/continuous_eval_${DATA_SPLIT_NAME}.log &
BGPIDS=$!

echo "background process ids: $BGPIDS"
trap 'kill $BGPIDS; exit' SIGINT
wait
