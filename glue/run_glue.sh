export MODEL_NAME_OR_PATH=bert-base-uncased
export TASK_NAME=mrpc
export MAX_SEQ_LEN=128
export BATCH_SIZE=32
export LR=2e-5
export MAX_EPOCHS=3
export OUTPUT_DIR=./out/
export RUN_NAME=run00
export CKPT_DIR=../aa_debias/out/bert_run00_gender_epoch:1

python run_glue.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length $MAX_SEQ_LEN \
    --per_device_train_batch_size $BATCH_SIZE \
    --learning_rate $LR \
    --num_train_epochs $MAX_EPOCHS \
    --output_dir $OUTPUT_DIR \
    --run_name $RUN_NAME \
    --overwrite_output_dir \
    --resume_from_checkpoint $CKPT_DIR