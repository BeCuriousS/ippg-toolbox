# Set up the variables:
DATA="ECU_SFA_SCH_HGR"
TRAIN_SPLIT="train_rot_gaussian"
PRETRAIN="VOC_trainval"
ATROUS_RATE_TRAIN=6
OUTPUT_STRIDE_TRAIN=16
ATROUS_RATE_EVAL=6
OUTPUT_STRIDE_EVAL=16
BATCH_SIZE=14
FINE_TUNE_BATCH_NORM="False"
BASE_LR=0.003
END_LR=0.0
NUM_ITERATIONS=41851
OPTIMIZER="momentum"
LABEL_WEIGHTS="None"
AUGMENTATION="rot_gaussian"
AT=6
# AT=12
OS=16
# OS=8


# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
DATASET_DIR="datasets"
# Set up the working directories.
DATASET_FOLDER="ECU_SFA_SCH_HGR"
EXP_FOLDER="exp/train_on_train_set"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/ECU/${EXP_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/train/${DATA},${TRAIN_SPLIT},${PRETRAIN},${BATCH_SIZE},${FINE_TUNE_BATCH_NORM},${BASE_LR},${END_LR},${NUM_ITERATIONS},${OPTIMIZER},${LABEL_WEIGHTS},${AUGMENTATION},${ATROUS_RATE_TRAIN},${ATROUS_RATE_EVAL}"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/eval/${DATA},${TRAIN_SPLIT},${PRETRAIN},${BATCH_SIZE},${FINE_TUNE_BATCH_NORM},${BASE_LR},${END_LR},${NUM_ITERATIONS},${OPTIMIZER},${LABEL_WEIGHTS},${AUGMENTATION},${ATROUS_RATE_TRAIN},${ATROUS_RATE_EVAL}"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/vis/${DATA},${TRAIN_SPLIT},${PRETRAIN},${BATCH_SIZE},${FINE_TUNE_BATCH_NORM},${BASE_LR},${END_LR},${NUM_ITERATIONS},${OPTIMIZER},${LABEL_WEIGHTS},${AUGMENTATION},${ATROUS_RATE_TRAIN},${ATROUS_RATE_EVAL}"
# DATASET="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/tf_dataset"
# DATASET="/app/shared/data/tf_dataset"
DATASET="/app/shared/data/tf_dataset"

mkdir -p "${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/vis/${DATA},${TRAIN_SPLIT},${PRETRAIN},${BATCH_SIZE},${FINE_TUNE_BATCH_NORM},${BASE_LR},${END_LR},${NUM_ITERATIONS},${OPTIMIZER},${LABEL_WEIGHTS},${AUGMENTATION},${ATROUS_RATE_TRAIN},${ATROUS_RATE_EVAL}"

python3 "${WORK_DIR}"/vis.py \
--logtostderr \
--vis_split="DatasetToSegment" \
--model_variant="xception_65" \
--atrous_rates=${AT} \
--atrous_rates=$((${AT}*2)) \
--atrous_rates=$((${AT}*3)) \
--output_stride=${OS} \
--decoder_output_stride=4 \
--vis_crop_size=513,513 \
--checkpoint_dir="${TRAIN_LOGDIR}" \
--vis_logdir="${VIS_LOGDIR}" \
--dataset_dir="${DATASET}" \
--max_number_of_iterations=1 \
--dataset=${DATA} \
--save_softmax_probabilities=True \
# --eval_scales=0.5 \
# --eval_scales=1 \
# --eval_scales=2 \
