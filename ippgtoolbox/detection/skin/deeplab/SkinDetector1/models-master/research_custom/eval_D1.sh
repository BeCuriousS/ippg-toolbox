# Set up the variables:
DATA="ECU_SFA_SCH_HGR"
TRAIN_SPLIT="train"
PRETRAIN="VOC_trainval"
ATROUS_RATE_TRAIN=6
OUTPUT_STRIDE_TRAIN=16
ATROUS_RATE_EVAL=6
OUTPUT_STRIDE_EVAL=16
BATCH_SIZE=14
FINE_TUNE_BATCH_NORM="True"
BASE_LR=0.007
END_LR=0.0
NUM_ITERATIONS=30000
OPTIMIZER="momentum"
LABEL_WEIGHTS="None"
AUGMENTATION="aug_0"

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
DATASET="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/tfrecord_tst_ign_aug"

mkdir -p "${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/exp"
mkdir -p "${TRAIN_LOGDIR}"



python3 "${WORK_DIR}"/eval.py \
--logtostderr \
--eval_split="val" \
--model_variant="xception_65" \
--atrous_rates=${ATROUS_RATE_EVAL} \
--atrous_rates=$((${ATROUS_RATE_EVAL}*2)) \
--atrous_rates=$((${ATROUS_RATE_EVAL}*3)) \
--output_stride=${OUTPUT_STRIDE_EVAL} \
--decoder_output_stride=4 \
--eval_crop_size=513,513 \
--checkpoint_dir="${TRAIN_LOGDIR}" \
--eval_logdir="${EVAL_LOGDIR}" \
--dataset_dir="${DATASET}" \
--dataset=${DATA} \
--eval_batch_size=1 \
--max_number_of_evaluations=0 \
--eval_interval_secs=5


/*
timeout 200 python3 "${WORK_DIR}"/eval.py \
--logtostderr \
--eval_split="val" \
--model_variant="xception_65" \
--atrous_rates=${ATROUS_RATE_EVAL} \
--atrous_rates=$((${ATROUS_RATE_EVAL}*2)) \
--atrous_rates=$((${ATROUS_RATE_EVAL}*3)) \
--output_stride=${OUTPUT_STRIDE_EVAL} \
--decoder_output_stride=4 \
--eval_crop_size=513,513 \
--checkpoint_dir="${TRAIN_LOGDIR}" \
--eval_logdir="${EVAL_LOGDIR}" \
--dataset_dir="${DATASET}" \
--dataset=${DATA} \
--eval_batch_size=1 \
--max_number_of_evaluations=0 \
--eval_interval_secs=5


timeout 400 python3 "${WORK_DIR}"/eval.py \
--logtostderr \
--eval_split="val_rot" \
--model_variant="xception_65" \
--atrous_rates=${ATROUS_RATE_EVAL} \
--atrous_rates=$((${ATROUS_RATE_EVAL}*2)) \
--atrous_rates=$((${ATROUS_RATE_EVAL}*3)) \
--output_stride=${OUTPUT_STRIDE_EVAL} \
--decoder_output_stride=4 \
--eval_crop_size=513,513 \
--checkpoint_dir="${TRAIN_LOGDIR}" \
--eval_logdir="${EVAL_LOGDIR}" \
--dataset_dir="${DATASET}" \
--dataset=${DATA} \
--eval_batch_size=1 \
--max_number_of_evaluations=0 \
--eval_interval_secs=5

timeout 200 python3 "${WORK_DIR}"/eval.py \
--logtostderr \
--eval_split="val_dark" \
--model_variant="xception_65" \
--atrous_rates=${ATROUS_RATE_EVAL} \
--atrous_rates=$((${ATROUS_RATE_EVAL}*2)) \
--atrous_rates=$((${ATROUS_RATE_EVAL}*3)) \
--output_stride=${OUTPUT_STRIDE_EVAL} \
--decoder_output_stride=4 \
--eval_crop_size=513,513 \
--checkpoint_dir="${TRAIN_LOGDIR}" \
--eval_logdir="${EVAL_LOGDIR}" \
--dataset_dir="${DATASET}" \
--dataset=${DATA} \
--eval_batch_size=1 \
--max_number_of_evaluations=0 \
--eval_interval_secs=5

timeout 200 python3 "${WORK_DIR}"/eval.py \
--logtostderr \
--eval_split="val_bright" \
--model_variant="xception_65" \
--atrous_rates=${ATROUS_RATE_EVAL} \
--atrous_rates=$((${ATROUS_RATE_EVAL}*2)) \
--atrous_rates=$((${ATROUS_RATE_EVAL}*3)) \
--output_stride=${OUTPUT_STRIDE_EVAL} \
--decoder_output_stride=4 \
--eval_crop_size=513,513 \
--checkpoint_dir="${TRAIN_LOGDIR}" \
--eval_logdir="${EVAL_LOGDIR}" \
--dataset_dir="${DATASET}" \
--dataset=${DATA} \
--eval_batch_size=1 \
--max_number_of_evaluations=0 \
--eval_interval_secs=5

timeout 200 python3 "${WORK_DIR}"/eval.py \
--logtostderr \
--eval_split="val_gaussian" \
--model_variant="xception_65" \
--atrous_rates=${ATROUS_RATE_EVAL} \
--atrous_rates=$((${ATROUS_RATE_EVAL}*2)) \
--atrous_rates=$((${ATROUS_RATE_EVAL}*3)) \
--output_stride=${OUTPUT_STRIDE_EVAL} \
--decoder_output_stride=4 \
--eval_crop_size=513,513 \
--checkpoint_dir="${TRAIN_LOGDIR}" \
--eval_logdir="${EVAL_LOGDIR}" \
--dataset_dir="${DATASET}" \
--dataset=${DATA} \
--eval_batch_size=1 \
--max_number_of_evaluations=0 \
--eval_interval_secs=5

timeout 200 python3 "${WORK_DIR}"/eval.py \
--logtostderr \
--eval_split="val_gaussian_dark" \
--model_variant="xception_65" \
--atrous_rates=${ATROUS_RATE_EVAL} \
--atrous_rates=$((${ATROUS_RATE_EVAL}*2)) \
--atrous_rates=$((${ATROUS_RATE_EVAL}*3)) \
--output_stride=${OUTPUT_STRIDE_EVAL} \
--decoder_output_stride=4 \
--eval_crop_size=513,513 \
--checkpoint_dir="${TRAIN_LOGDIR}" \
--eval_logdir="${EVAL_LOGDIR}" \
--dataset_dir="${DATASET}" \
--dataset=${DATA} \
--eval_batch_size=1 \
--max_number_of_evaluations=0 \
--eval_interval_secs=5

timeout 200 python3 "${WORK_DIR}"/eval.py \
--logtostderr \
--eval_split="val_gaussian_bright" \
--model_variant="xception_65" \
--atrous_rates=${ATROUS_RATE_EVAL} \
--atrous_rates=$((${ATROUS_RATE_EVAL}*2)) \
--atrous_rates=$((${ATROUS_RATE_EVAL}*3)) \
--output_stride=${OUTPUT_STRIDE_EVAL} \
--decoder_output_stride=4 \
--eval_crop_size=513,513 \
--checkpoint_dir="${TRAIN_LOGDIR}" \
--eval_logdir="${EVAL_LOGDIR}" \
--dataset_dir="${DATASET}" \
--dataset=${DATA} \
--eval_batch_size=1 \
--max_number_of_evaluations=0 \
--eval_interval_secs=5 
*/
