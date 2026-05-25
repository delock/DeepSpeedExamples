#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
# GRPO + HumanEval reward training script.
#
# The model learns to write correct Python functions judged by unit-test
# execution on the HumanEval benchmark.
#
# Usage:
#   bash training_scripts/grpo/run_humaneval_qwen0.5b.sh
#   bash training_scripts/grpo/run_humaneval_qwen0.5b.sh Qwen/Qwen2.5-0.5B 2 ./my_output 1

ACTOR_MODEL_PATH=${1:-"Qwen/Qwen2.5-0.5B"}
ACTOR_ZERO_STAGE=${2:-0}
OUTPUT=${3:-"./grpo_output_humaneval_qwen0.5b"}
NUM_GPUS=${4:-1}

mkdir -p $OUTPUT

deepspeed --num_gpus $NUM_GPUS main_grpo.py \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --data_path openai/openai_humaneval \
   --data_split 9,1,0 \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --num_padding_at_beginning 0 \
   --per_device_generation_batch_size 2 \
   --per_device_training_batch_size 2 \
   --generation_batches 1 \
   --max_answer_seq_len 512 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate 1e-5 \
   --num_train_epochs 3 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 4 \
   --num_warmup_steps 10 \
   --deepspeed \
   --seed 1234 \
   --grpo_num_generations 4 \
   --kl_ctl 0.05 \
   --cliprange 0.2 \
   --reward_type humaneval \
   --humaneval_timeout 10.0 \
   --print_answers \
   --print_answers_interval 20 \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
