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
