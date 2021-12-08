#!/bin/bash
# Set up the variables:
AT=6  # AT=12
OS=16 # OS=8

WORK_DIR="$(pwd)/deeplab"
DATASET="/home/any/processing/tf_dataset"
VIS_LOGDIR="/app/shared/segmentation"
CHKPT_DIR="/app/shared/deeplab/deeplab/data/model_checkpoint"

# Run deeplab on dataset
python3 "${WORK_DIR}"/vis.py \
    --logtostderr \
    --vis_split="DatasetToSegment" \
    --model_variant="xception_65" \
    --atrous_rates=${AT} \
    --atrous_rates=$((${AT} * 2)) \
    --atrous_rates=$((${AT} * 3)) \
    --output_stride=${OS} \
    --decoder_output_stride=4 \
    --vis_crop_size=513,513 \
    --checkpoint_dir="${CHKPT_DIR}" \
    --vis_logdir="${VIS_LOGDIR}" \
    --dataset_dir="${DATASET}" \
    --max_number_of_iterations=1 \
    --dataset="IPPGTOOLBOX" \
    --save_softmax_probabilities=True \
    --also_save_raw_predictions=False
