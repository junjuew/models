#!/bin/bash

set -e

DIR=$(dirname $0)

source ${DIR}/munichrc.sh

echo ""
echo "Tuning last layer"
echo ""
#source ${DIR}/finetune_mobilenet_v1_on_munich_last_layer.sh

echo ""
echo "Tuning all layers"
echo ""

source ${DIR}/finetune_mobilenet_v1_on_munich_all_layers.sh

