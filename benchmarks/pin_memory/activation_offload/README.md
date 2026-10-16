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

## Environment setup (Ascend 910B4, CANN 8.5.1 + torch 2.10)

NPU runs use the same script (the peak-memory fallback now goes through the
accelerator abstraction, and the pin backend can be selected from the
environment). Setup:

```bash
conda activate cann   # torch 2.10.0+cpu + torch_npu 2.10.0.post7 + transformers 5.19.0
source ~/Ascend/cann-8.5.1/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
ulimit -l unlimited
```

One-click run — `npu/run.sh` pins the NPU experiment parameters (Qwen3-4B,
batch 1 × seq 8192, ZeRO-2 CPU AdamW, Liger FLCE + TiledMLP, 10 steps /
1 warmup; the H200 rows run Qwen3-8B, which does not fit this card):

```bash
cd npu && ./run.sh                      # model from the HF hub
cd npu && MODEL=/path/to/Qwen3-4B ./run.sh   # local checkpoint
```

Manual equivalent of the pinned parameters:

```bash
DS_PIN_MEMORY_BACKEND=torch python bench.py --model Qwen/Qwen3-4B --batch 1 \
    --seq 8192 --offload-optimizer --act-mem-opt --steps 10 --warmup 1
```

Notes:

- On this host the native pin backend's device registration collapses for the
  ~40 MB per-step hidden-state tensors at seq 8192 (pinned arm 2-4x slower
  than pageable), so the NPU run selects `DS_PIN_MEMORY_BACKEND=torch`
  (torch_npu's pin_memory path), which is healthy and beats pageable. H200
  behavior is unchanged; `bench.py` only prefers an externally set backend.
- transformers 5.19.0: the 4.57 default Qwen3 attention forces a hub `kernels`
  package that is unavailable on this host.
- Qwen3-8B ZeRO-2 static state (32.8 GB) exceeds the 29.5 GiB usable card, so
  NPU rows use Qwen3-4B. seq 8192 peaks at ~16.5 GiB with act-mem-opt.
