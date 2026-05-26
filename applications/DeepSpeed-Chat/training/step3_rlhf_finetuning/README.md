# GRPO Training (Step 3)

## Basic Usage

```bash
deepspeed --include localhost:1,2 \
    main_grpo.py \
    --data_path openai/openai_humaneval \
    --data_split 0,0,10 \
    --actor_model_name_or_path <path_to_sft_checkpoint> \
    --per_device_generation_batch_size 2 \
    --per_device_training_batch_size 2 \
    --generation_batches 1 \
    --max_answer_seq_len 512 \
    --max_prompt_seq_len 512 \
    --actor_learning_rate 5e-6 \
    --num_train_epochs 1 \
    --actor_gradient_checkpointing \
    --actor_zero_stage 2 \
    --kl_ctl 0.05 \
    --grpo_num_generations 8 \
    --dtype bf16 \
    --seed 42 \
    --reward_type humaneval \
    --humaneval_timeout 10 \
    --output_dir ./output
```

## Generate Optimization Flags

### `--shared_prefix_generate`

Prefills each unique prompt once, then copies the KV cache G times using
`batch_repeat_interleave`. Eliminates (G-1)/G redundant prefill computation.
Decode still uses HF `model.generate()`.

```bash
deepspeed --include localhost:1,2 main_grpo.py \
    ... \
    --shared_prefix_generate
```

**Speedup**: 2.4x (generate: 19.2s -> 7.47s)

### `--early_exit_generate`

Custom token-by-token decode loop with batch compaction. **Subsumes
`--shared_prefix_generate`** (prefills each prompt once and copies the KV cache
G times), so you do not need both flags. Additionally, when a sequence hits EOS,
it is physically removed from the batch via `reorder_cache`, reducing compute
for subsequent decode steps.

```bash
deepspeed --include localhost:1,2 main_grpo.py \
    ... \
    --early_exit_generate
```

**Speedup**: 3.3x (generate: 19.2s -> 4.51s, epoch: 18.8min -> 5.7min)

This is the recommended flag for best performance.

### `--async_reward`

Overlaps reward computation (CPU, subprocess code execution) with the next
step's generation (GPU). Introduces one-step policy staleness which is
negligible due to small updates + clipping.

```bash
deepspeed --include localhost:1,2 main_grpo.py \
    ... \
    --early_exit_generate \
    --async_reward
```

## Reward Types

- `--reward_type humaneval` -- executes generated code against HumanEval test cases
- `--reward_type mbpp` -- executes against MBPP test cases
- `--reward_type mixed` -- routes to HumanEval or MBPP based on prompt format

All reward functions use `ProcessPoolExecutor` for parallel execution with
`--humaneval_timeout <seconds>` as the per-test timeout.

## Performance Summary (2xA100-40G, Qwen2.5-0.5B, G=8, HumanEval 1 epoch)

| Config | avg generate | avg e2e | total epoch |
|--------|-------------|---------|-------------|
| baseline (repeat + generate) | 19.2s | 27.5s | 18.8 min |
| `--shared_prefix_generate` | 7.47s | 11.4s | 7.8 min |
| `--early_exit_generate` | 4.51s | 8.38s | 5.7 min |
