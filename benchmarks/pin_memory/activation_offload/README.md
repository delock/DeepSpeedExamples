# Activation / checkpoint hidden-state offload

Compares `use_pin_memory` True vs False on `CheckpointHiddenStatesOffload`
with **async / side streams held on**. This is not the async-vs-blocking
table from DeepSpeed #8282.

See the [parent README](../README.md).

Published rows use ``--model`` (config + random weights, bf16, non-reentrant
checkpointing). Synthetic ``--hidden/--layers`` is smoke only. The HF timed
step is causal-LM forward loss, backward, and a full-parameter AdamW update.

H200 (async on, pin vs pageable):

```bash
ulimit -l unlimited
python bench.py --model Qwen/Qwen3-8B --batch 1 --seq 8192
python bench.py --model Qwen/Qwen3.5-9B --batch 1 --seq 8192
```

ZeRO-2 CPU optimizer offload (params stay on GPU; scale sequence length):

```bash
python bench.py --model Qwen/Qwen3-8B --batch 1 --seq 32768 --offload-optimizer
python bench.py --model Qwen/Qwen3.5-9B --batch 1 --seq 32768 --offload-optimizer
```

HF path defaults to Liger (RoPE, RMSNorm, SwiGLU, fused linear CE) plus DeepSpeed
TiledMLP. If Liger fused CE is missing, DeepSpeed ``TiledFusedLogitsLoss`` is used.
Disable with ``--no-act-mem-opt``.

## Environment setup (H200, CUDA)

```bash
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ulimit -l unlimited
```

One-click run — `cuda/run.sh` pins the H200 experiment parameters (Qwen3-8B,
batch 1 × seq 32768, ZeRO-2 CPU AdamW, Liger fused linear CE + TiledMLP,
10 steps / 1 warmup):

```bash
cd cuda && ./run.sh                      # model from the HF hub
cd cuda && MODEL=/path/to/Qwen3-8B ./run.sh   # local checkpoint
```

Manual equivalent of the pinned parameters:

```bash
python bench.py --model Qwen/Qwen3-8B --batch 1 \
    --seq 32768 --offload-optimizer --act-mem-opt --steps 10 --warmup 1
```

## Environment setup (Ascend 910B4, CANN 8.5.1 + torch 2.10)

```bash
conda activate <your-env>            # torch_npu env, e.g. torch 2.10.0+cpu + torch_npu 2.10.0.post7 + transformers 5.19.0
source <your-cann-install>/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
ulimit -l unlimited
```

One-click run — `npu/run.sh` pins the NPU experiment parameters (Qwen3-4B,
batch 1 × seq 8192, ZeRO-2 CPU AdamW, Liger fused linear CE + TiledMLP,
10 steps / 1 warmup):

```bash
cd npu && ./run.sh                      # model from the HF hub
cd npu && MODEL=/path/to/Qwen3-4B ./run.sh   # local checkpoint
```

Manual equivalent of the pinned parameters:

```bash
DS_PIN_MEMORY_BACKEND=torch python bench.py --model Qwen/Qwen3-4B --batch 1 \
    --seq 8192 --offload-optimizer --act-mem-opt --steps 10 --warmup 1
```
