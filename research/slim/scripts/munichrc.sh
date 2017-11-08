# Where the pre-trained mobilenet checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR="/home/junjuew/mobisys18/pretrained_models/mobilenet_ckpt"

MODEL_NAME=mobilenet_v1

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR="/home/zf/opt/drone-scalable-search/processed_dataset/munich/mobilenet_train/logs"
ALL_LAYER_TRAIN_DIR="${TRAIN_DIR}/logs_finetune_all_layers"
LAST_LAYER_TRAIN_DIR="${TRAIN_DIR}/logs_finetune_last_layer_only"

# Where the dataset (TFrecords) was saved to.
DATASET_DIR="/home/zf/opt/drone-scalable-search/processed_dataset/munich/mobilenet_train"
TEST_DATASET_DIR="/home/zf/opt/drone-scalable-search/processed_dataset/munich/mobilenet_test"
