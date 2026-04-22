#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
# GRPO training script for OPT-1.3B on single A100 (80GB)

ACTOR_MODEL_PATH=${1:-"facebook/opt-1.3b"}
ACTOR_ZERO_STAGE=${2:-2}
OUTPUT=${3:-"./grpo_output_opt1.3b"}
NUM_GPUS=${4:-1}

mkdir -p $OUTPUT

deepspeed --num_gpus $NUM_GPUS main_grpo.py \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --num_padding_at_beginning 1 \
   --per_device_generation_batch_size 4 \
   --per_device_training_batch_size 4 \
   --generation_batches 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate 9.65e-6 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --num_warmup_steps 100 \
   --deepspeed \
   --seed 1234 \
   --enable_hybrid_engine \
   --grpo_num_generations 4 \
   --kl_ctl 0.1 \
   --cliprange 0.2 \
   --print_answers \
   --print_answers_interval 10 \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
