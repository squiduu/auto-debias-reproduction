TESTS=seat3,seat3b,seat4,seat5,seat5b,seat6,seat6b,seat7,seat7b,seat8,seat8b,seat9,seat10
MODEL_NAME=bert
OUTPUT_DIR=./seat/out/
SEED=42
DATA_DIR=./data/tests/
NUM_SAMPLES=100000
RUN_NAME=run01
CKPT_DIR=./aa_debias/out/bert_run00_gender_epoch:10/
VERSION=bert-base-uncased

# run this file at my_xai directory
python ./seat/run_seat.py \
    --tests ${TESTS} \
    --model_name ${MODEL_NAME} \
    --seed ${SEED} \
    --log_dir ${OUTPUT_DIR} \
    --results_path ${OUTPUT_DIR}${MODEL_NAME}_seat_${RUN_NO}.csv \
    --ignore_cached_encs \
    --data_dir ${DATA_DIR} \
    --exp_dir ${OUTPUT_DIR} \
    --num_samples ${NUM_SAMPLES} \
    --run_name ${RUN_NAME} \
    --use_ckpt \
    --ckpt_dir ${CKPT_DIR} \
    --version ${VERSION}