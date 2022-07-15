MODEL_VERSION=bert-base-uncased
MODEL_NAME=bert
SEED=42
BATCH_SIZE=1000
DEBIAS_TYPE=gender
RUN_NO=run00
DATA_DIR=./data/
MAX_PROMPT_LEN=5
TOP_K=100
SAVE_DIR=./prompts/
NUM_WORKERS=4
LR=2e-5
ACCUMULATE_GRAD_BATCHES=1
GRADIENT_CLIP_VAL=1.0
MAX_EPOCHS=100
OUTPUT_DIR=./out/
GPUS=2
PATIENCE=10
CKPT_PATH=./ckpts/
PRECISION=16

python generate_prompts.py \
    --model_version ${MODEL_VERSION} \
    --model_name ${MODEL_NAME} \
    --seed ${SEED} \
    --batch_size ${BATCH_SIZE} \
    --debias_type ${DEBIAS_TYPE} \
    --run_no ${RUN_NO} \
    --data_dir ${DATA_DIR} \
    --max_prompt_len ${MAX_PROMPT_LEN} \
    --top_k ${TOP_K} \
    --save_dir ${SAVE_DIR} \
    --num_workers ${NUM_WORKERS} \
    --lr ${LR} \
    --accumulate_grad_batches ${ACCUMULATE_GRAD_BATCHES} \
    --gradient_clip_val ${GRADIENT_CLIP_VAL} \
    --max_epochs ${MAX_EPOCHS} \
    --output_dir ${OUTPUT_DIR} \
    --gpus ${GPUS} \
    --patience ${PATIENCE} \
    --ckpt_path ${CKPT_PATH} \
    --precision ${PRECISION}
