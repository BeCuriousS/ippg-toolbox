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


CUDA_VISIBLE_DEVICES=0 python3 "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split=${TRAIN_SPLIT} \
  --model_variant="xception_65" \
  --atrous_rates=${ATROUS_RATE_TRAIN} \
  --atrous_rates=$((${ATROUS_RATE_TRAIN}*2)) \
  --atrous_rates=$((${ATROUS_RATE_TRAIN}*3)) \
  --output_stride=${OUTPUT_STRIDE_TRAIN} \
  --decoder_output_stride=4 \
  --train_crop_size=513,513 \
  --train_batch_size=${BATCH_SIZE} \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=${FINE_TUNE_BATCH_NORM} \
  --tf_initial_checkpoint="${INIT_FOLDER}/${PRETRAIN}/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${DATASET}" \
  --dataset=${DATA} \
  --initialize_last_layer=False \
  --last_layers_contain_logits_only=True \
  --save_summaries_images=False \
  --save_interval_secs=180 \
  --save_summaries_secs=180 \
  --max_to_keep=500 \
  --base_learning_rate=${BASE_LR} \
  --end_learning_rate=${END_LR} \
  --optimizer=${OPTIMIZER} \
  --adam_learning_rate=${BASE_LR} &

sleep 30s

CUDA_VISIBLE_DEVICES=1 python3 "${WORK_DIR}"/eval.py \
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
--max_number_of_evaluations=-1 \
--eval_interval_secs=5
