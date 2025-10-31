#!/bin/bash

MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20000-29999 -n 1)
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)

MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR="./output"

DATASETS="watermarks"

export HF_USE_FLASH_ATTENTION_2=0
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwen-vl-finetune/qwenvl/train/train_qwen.py \
         --model_name_or_path $MODEL_PATH \
         --tune_mm_vision False \
         --tune_mm_mlp True \
         --tune_mm_llm True \
         --dataset_use $DATASETS \
         --output_dir $OUTPUT_DIR \
         --bf16 \
         --per_device_train_batch_size 4 \
         --gradient_accumulation_steps 4 \
         --learning_rate 2e-7 \
         --mm_projector_lr 1e-5 \
         --vision_tower_lr 1e-6 \
         --optim adamw_torch \
         --model_max_length 2048 \
         --max_pixels 112896 \
         --min_pixels 3136 \
         --num_train_epochs 10 \
         --warmup_ratio 0.03 \
         --lr_scheduler_type "cosine" \
         --weight_decay 0.01 \
         --logging_steps 10 \
         --save_steps 1000 \
         --save_total_limit 4 \
         --lora_enable True \
         --lora_r 4 \
         --lora_alpha 16 \
         --lora_dropout 0.05 \
         --deepspeed ./qwen-vl-finetune/scripts/zero3.json
