#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""

engine = DeepSpeedGRPOEngine(actor_model_name_or_path=...,
                             tokenizer=tokenizer,
                             args=args)
trainer = DeepSpeedGRPOTrainer(rlhf_engine=engine, args=args, reward_fn=reward_fn)

for prompt_batch in prompt_train_dataloader:
    out = trainer.generate_experience(prompt_batch)
    actor_loss = trainer.train_grpo(out)

"""
import argparse
import os
import random
import time
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from torch.utils.tensorboard import SummaryWriter

from transformers import (
    SchedulerType,
    default_data_collator,
)

import deepspeed

from dschat.rlhf.grpo_trainer import DeepSpeedGRPOTrainer, DeepSpeedGRPOTrainerUnsupervised
from dschat.rlhf.grpo_engine import DeepSpeedGRPOEngine
from dschat.utils.data.data_utils import create_prompt_dataset, MiniDataset, DataCollatorRLHF, get_unsupervised_data
from dschat.utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, moving_average, save_zero_three_model, load_hf_tokenizer
from dschat.utils.module.lora import convert_lora_to_linear_layer
from dschat.utils.perf import print_throughput_step3
from deepspeed.accelerator import get_accelerator

writer = None


def parse_args():
    global writer
    parser = argparse.ArgumentParser(
        description="(Step 3) GRPO training arguments")

    parser.add_argument(
        '--data_path',
        nargs='*',
        default=['Dahoas/rm-static'],
        help=
        'Path to the training dataset. Accepted format: 1) a single data path, 2) multiple datasets in the form: dataset1-path dataset2-path ...'
    )
    parser.add_argument(
        '--data_split',
        type=str,
        default='2,4,4',
        help=
        'Comma-separated list of proportions for training phase 1, 2, and 3 data.'
    )
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files',
        help=
        'Where to store the data-related files such as shuffle index.'
    )
    parser.add_argument(
        "--unsupervised_dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).")
    parser.add_argument(
        "--unsupervised_dataset_config_name",
        type=str,
        default=None,
        help=
        "The configuration name of the dataset to use (via the datasets library)."
    )
    parser.add_argument("--unsup_coef",
                        type=float,
                        default=27.8,
                        help='''gamma in Equation 2 from InstructGPT paper''')
    parser.add_argument(
        "--actor_model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True)
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input."
    )
    parser.add_argument(
        "--per_device_generation_batch_size",
        type=int,
        default=16,
        help=
        "Batch size (per device) for the training dataloader and generation purpose."
    )
    parser.add_argument(
        "--per_device_training_batch_size",
        type=int,
        default=16,
        help=
        "Mini Batch size (per device) for the training dataloader and training purpose."
    )
    parser.add_argument("--generation_batches",
                        type=int,
                        default=1,
                        help="Generate x batches to go to training mode.")
    parser.add_argument(
        "--max_prompt_seq_len",
        type=int,
        default=256,
        help="The maximum prompt sequence length.")
    parser.add_argument("--max_answer_seq_len",
                        type=int,
                        default=256,
                        help="The maximum answer sequence length.")
    parser.add_argument(
        "--actor_learning_rate",
        type=float,
        default=9.65e-6,
        help="Initial learning rate for actor.")
    parser.add_argument("--actor_weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.")
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=100,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    # DeepSpeed
    parser.add_argument(
        "--enable_hybrid_engine",
        action='store_true',
        help=
        "Enable hybrid engine for actor model to optimize both inference and training through DeepSpeed."
    )
    parser.add_argument(
        "--unpin_actor_parameters",
        action='store_true',
        help=
        "Unpin actor's parameters during generation."
    )
    parser.add_argument(
        "--release_inference_cache",
        action='store_true',
        help=
        "Release the memory cache used for inference."
    )
    parser.add_argument(
        "--inference_tp_size",
        type=int,
        default=1,
        help=
        "Tensor-parallelism degree used for inference."
    )
    parser.add_argument(
        "--tp_gather_partition_size",
        type=int,
        default=8,
        help=
        "Granularity to bring in layers for TP sharding inside the hybrid engine."
    )
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument('--dtype',
                        type=str,
                        default='fp16',
                        choices=['fp16', 'bf16'],
                        help='Training data type')
    parser.add_argument(
        '--offload_reference_model',
        action='store_true',
        help='Enable ZeRO Offload techniques for reference model')
    parser.add_argument(
        '--actor_zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and ref).')
    parser.add_argument(
        '--actor_gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for Actor model.')
    parser.add_argument(
        "--actor_dropout",
        type=float,
        default=None,
        help="Override actor dropout."
    )
    parser.add_argument("--actor_lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--actor_lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--actor_lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial actor LoRA learning rate."
    )
    parser.add_argument(
        '--enable_mixed_precision_lora',
        action='store_true',
        help='Enable Mixed Precision ZeRO++ for training and generation.')
    parser.add_argument(
        '--compute_fp32_loss',
        action='store_true',
        help='Compute loss in fp32.')
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step3_tensorboard")
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add </s> as additional special token to tokenizer")
    parser.add_argument('--print_answers',
                        action='store_true',
                        help='Print prompt and answers during training')
    parser.add_argument(
        "--print_answers_interval",
        type=int,
        default=1,
        help="If --print_answers enabled, controls the printing interval.")
    parser.add_argument(
        '--enable_test_mode',
        action='store_true',
        help=
        'Enable a testing mode that terminates training based on args.test_stop_step'
    )
    parser.add_argument(
        "--test_stop_step",
        type=int,
        default=0,
        help=
        "Training non-overflow step at which to terminate training during testing."
    )

    # GRPO specific
    parser.add_argument("--grpo_num_generations",
                        type=int,
                        default=4,
                        help="Number of generations per prompt for GRPO.")
    parser.add_argument("--kl_ctl",
                        type=float,
                        default=0.1,
                        help="KL divergence coefficient.")
    parser.add_argument("--cliprange",
                        type=float,
                        default=0.2,
                        help="PPO clip range for GRPO actor loss.")
    parser.add_argument(
        "--reward_type",
        type=str,
        default="mock",
        choices=["mock", "humaneval", "mbpp", "mixed"],
        help=(
            "Reward function to use.  "
            "'mock' returns 1.0 for every response (for debugging). "
            "'humaneval' executes generated code against HumanEval unit tests. "
            "'mbpp' executes generated code against MBPP unit tests."
        ),
    )
    parser.add_argument(
        "--humaneval_timeout",
        type=float,
        default=5.0,
        help="Per-sample subprocess timeout (seconds) for humaneval reward.",
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    if args.enable_tensorboard:
        print(
            f"Tensorboard logs going to: {args.tensorboard_path}/grpo_tensorboard_logs"
        )
        writer = SummaryWriter(
            f"{args.tensorboard_path}/grpo_tensorboard_logs")

    if args.inference_tp_size > 1:
        assert (
            args.actor_zero_stage == 3
        ), "Zero stage 3 must be used to do Tensor sharding in the hybrid engine"

    return args


def create_datasets(args, tokenizer, train_phase=3):
    unsupervised_training_enabled = args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    prompt_train_dataset, _ = create_prompt_dataset(
        args.local_rank, args.data_path, args.data_split,
        args.data_output_path, train_phase, args.seed, tokenizer,
        args.max_prompt_seq_len)
    if unsupervised_training_enabled:
        unsupervised_train_dataset = get_unsupervised_data(args, tokenizer)
    else:
        unsupervised_train_dataset = None

    data_collator = DataCollatorRLHF(args.max_prompt_seq_len,
                                      args.inference_tp_size)
    if args.local_rank == -1:
        prompt_train_sampler = RandomSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = RandomSampler(
                unsupervised_train_dataset)
    else:
        prompt_train_sampler = DistributedSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = DistributedSampler(
                unsupervised_train_dataset)
    prompt_train_dataloader = DataLoader(
        prompt_train_dataset,
        collate_fn=data_collator,
        sampler=prompt_train_sampler,
        batch_size=args.per_device_generation_batch_size)
    if unsupervised_training_enabled:
        unsupervised_train_dataloader = DataLoader(
            unsupervised_train_dataset,
            collate_fn=default_data_collator,
            sampler=unsupervised_train_sampler,
            batch_size=args.per_device_generation_batch_size)
    else:
        unsupervised_train_dataloader = [None] * len(
            prompt_train_dataloader)

    num_update_steps_per_epoch = len(prompt_train_dataloader) * \
        (args.per_device_generation_batch_size / args.per_device_training_batch_size) / \
        args.gradient_accumulation_steps
    num_total_iters = int(args.num_train_epochs * num_update_steps_per_epoch)

    return prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    unsupervised_training_enabled = args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    if unsupervised_training_enabled:
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps * 2
    else:
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps

    set_random_seed(args.seed)
    torch.distributed.barrier()

    args.end_of_conversation_token = ""
    additional_special_tokens = args.end_of_conversation_token if args.add_eot_token else None
    tokenizer = load_hf_tokenizer(args.actor_model_name_or_path,
                                  fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)

    prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters = create_datasets(
        args=args, tokenizer=tokenizer, train_phase=3)

    rlhf_engine = DeepSpeedGRPOEngine(
        actor_model_name_or_path=args.actor_model_name_or_path,
        tokenizer=tokenizer,
        num_total_iters=num_total_iters,
        args=args)

    if args.enable_mixed_precision_lora:
        assert args.actor_lora_dim > 0, "Mixed Precision LoRA requires LoRA to be enabled"
        assert args.actor_zero_stage == 3, "Mixed Precision LoRA requires Zero stage 3"
        rlhf_engine.actor.optimizer.quantize_nontrainable_params()
        print_rank_0("Mixed Precision ZeRO++ enabled")

    grpo_trainer_cls = DeepSpeedGRPOTrainerUnsupervised if unsupervised_training_enabled else DeepSpeedGRPOTrainer

    # ------------------------------------------------------------------ #
    # Reward function selection                                            #
    # ------------------------------------------------------------------ #
    if args.reward_type == "humaneval":
        from dschat.utils.reward.humaneval_reward import HumanEvalRewardFn
        reward_fn = HumanEvalRewardFn(timeout=args.humaneval_timeout)
        print_rank_0(
            f"Using HumanEval reward (timeout={args.humaneval_timeout}s, "
            f"{reward_fn.task_count} tasks loaded)",
            args.global_rank,
        )
    elif args.reward_type == "mbpp":
        from dschat.utils.reward.mbpp_reward import MBPPRewardFn
        reward_fn = MBPPRewardFn(timeout=args.humaneval_timeout, split="train")
        print_rank_0(
            f"Using MBPP reward (timeout={args.humaneval_timeout}s, "
            f"{reward_fn.task_count} tasks loaded)",
            args.global_rank,
        )
    elif args.reward_type == "mixed":
        from dschat.utils.reward.mixed_reward import MixedRewardFn
        reward_fn = MixedRewardFn(timeout=args.humaneval_timeout)
        print_rank_0(
            f"Using Mixed reward (HumanEval + MBPP, timeout={args.humaneval_timeout}s)",
            args.global_rank,
        )
    else:
        reward_fn = None   # trainer uses default mock (always 1.0)
        print_rank_0("Using mock reward (all 1.0)", args.global_rank)

    trainer = grpo_trainer_cls(rlhf_engine, args, reward_fn=reward_fn)

    exp_mini_dataset = MiniDataset(args.generation_batches,
                                   args.per_device_training_batch_size)
    unsup_mini_dataset = MiniDataset(args.generation_batches,
                                     args.per_device_training_batch_size)

    print_rank_0(
        f"***** Running GRPO training (total_iters={num_total_iters}) *****",
        args.global_rank)
    print_rank_0(
        f"***** GRPO config: num_generations={args.grpo_num_generations}, kl_ctl={args.kl_ctl}, cliprange={args.cliprange} *****",
        args.global_rank)

    non_overflow_step_count = 0

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Generation Batches {min(len(prompt_train_dataloader), len(unsupervised_train_dataloader))}",
            args.global_rank)
        for step, (batch_prompt, batch_unsupervised) in enumerate(
                zip(prompt_train_dataloader, unsupervised_train_dataloader)):

            batch_prompt = to_device(batch_prompt, device)

            out = trainer.generate_experience(batch_prompt['prompt'],
                                              batch_prompt['prompt_att_mask'],
                                              step)

            training_start = time.time()
            if batch_unsupervised is not None:
                batch_unsupervised = to_device(batch_unsupervised, device)
                unsup_dataset = unsup_mini_dataset.add(batch_unsupervised)
            else:
                unsup_dataset = unsup_mini_dataset.add(
                    [[None] * args.per_device_generation_batch_size])

            exp_dataset = exp_mini_dataset.add(out)

            if exp_dataset is not None:
                inner_iter = 0
                actor_loss_sum, unsup_loss_sum = 0, 0
                average_reward = 0

                if args.actor_gradient_checkpointing:
                    rlhf_engine.actor.gradient_checkpointing_enable()

                for i, (exp_data, unsup_data) in enumerate(
                        zip(exp_dataset, unsup_dataset)):
                    actor_loss = trainer.train_grpo(exp_data)
                    actor_loss_sum += actor_loss.item()
                    average_reward += exp_data["reward_scores"].mean()

                    if unsupervised_training_enabled:
                        unsup_loss = trainer.train_unsupervised(
                            unsup_data, args.unsup_coef)
                        unsup_loss_sum += unsup_loss.item()

                    inner_iter += 1

                end = time.time()
                training_time = end - training_start
                e2e_time = training_time + trainer.generate_time * args.generation_batches

                average_reward = get_all_reduce_mean(average_reward).item()

                print_rank_0(
                    f'Epoch: {epoch} | Step: {step} | Actor Loss: {actor_loss_sum/inner_iter} | Unsupervised Loss: {unsup_loss_sum/inner_iter}',
                    args.global_rank)
                print_rank_0(
                    f"Average reward score: {average_reward/inner_iter}",
                    args.global_rank)
                print_rank_0(
                    f"[Timing] generate={trainer.generate_time:.2f}s | logprob={getattr(trainer,'logprob_time',0):.2f}s | reward={getattr(trainer,'reward_time',0):.2f}s | train={training_time:.2f}s | e2e={e2e_time:.2f}s",
                    args.global_rank)
                print_rank_0(
                    "-------------------------------------------------------------------------------------",
                    args.global_rank)

                if args.enable_tensorboard and torch.distributed.get_rank(
                ) == 0:
                    writer.add_scalar('reward',
                                      average_reward / inner_iter,
                                      global_step=step)
                    writer.add_scalar('actor_loss',
                                      actor_loss.item(),
                                      global_step=step)
                    writer.flush()

            if args.actor_gradient_checkpointing:
                rlhf_engine.actor.gradient_checkpointing_disable()

            if args.dtype != "bf16":
                actor_overflow = rlhf_engine.actor.optimizer.overflow
            else:
                actor_overflow = False

            if not actor_overflow:
                non_overflow_step_count += 1

            if args.enable_test_mode and non_overflow_step_count == args.test_stop_step:
                break

        if args.enable_test_mode:
            break

    if args.output_dir is not None:
        print_rank_0('saving model ...')
        rlhf_engine.actor = convert_lora_to_linear_layer(rlhf_engine.actor)

        if torch.distributed.get_rank() == 0:
            save_hf_format(rlhf_engine.actor,
                           tokenizer,
                           args,
                           sub_folder='actor')

        if args.actor_zero_stage == 3:
            save_zero_three_model(rlhf_engine.actor,
                                  global_rank=args.global_rank,
                                  save_dir=os.path.join(
                                      args.output_dir, 'actor'),
                                  zero_stage=args.actor_zero_stage)


if __name__ == "__main__":
    main()
