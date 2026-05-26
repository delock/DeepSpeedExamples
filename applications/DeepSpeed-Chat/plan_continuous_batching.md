# Continuous Batching for GRPO Rollout — Implementation Plan

## Problem

Current rollout generates a fixed batch of sequences, padded to max length.
- Average utilization: 34%
- 66% of compute wasted on padding
- Batch stalls until the longest sequence finishes

## Design: Multi-prompt Interleaved Continuous Batching

Instead of generating all G responses for one prompt before moving to the next,
interleave multiple prompts' generations in the same batch. When a sequence finishes
(EOS), its slot is freed and filled with the next pending generation (possibly from
a different prompt).

## Implementation Layers

### Layer 1: Masked Early-Exit (minimal change)

**Where:** `dschat/rlhf/grpo_trainer.py`

**Approach:**
- Replace `model.generate()` with a custom token-by-token decode loop
- Use HF's `DynamicCache`, manually manage batch dimension
- Completed sequences get attention mask zeroed out (not removed from batch)
- Once all G responses for a prompt are done, immediately return them for reward
- No paged KV cache needed

**Pros:** Minimal code change, no DeepSpeed kernel modifications  
**Cons:** Memory not freed for completed sequences (padding in memory still exists)  
**Expected gain:** ~50% reduction in wasted FLOPs

---

### Layer 2: Dynamic Batch Scheduler (moderate change)

**Where:** `dschat/rlhf/grpo_trainer.py` + new `dschat/rlhf/generation_scheduler.py`

**Approach:**
- Implement a slot manager and prefill/decode state machine
- Dynamic batch size: completed sequences truly removed, new sequences added
- KV cache spliced/concatenated as sequences enter/leave
- Scheduler tracks per-prompt completion (all G done → ready for reward/train)
- Can overlap reward computation for completed prompts while others still generating

**Pros:** GPU utilization ~90%+, memory reclaimed  
**Cons:** Complex KV cache management, need to handle ragged batch shapes  
**Expected gain:** ~3x generate speedup (matches profiled potential of 3.26x)

---

### Layer 3: DeepSpeed Inference V2 Integration (large change)

**Where:** `deepspeed/inference/v2/` (upstream DeepSpeed)

**Approach:**
- Leverage existing ragged batching infrastructure in inference v2
- Implement paged KV cache (block-level allocation/reclaim)
- Full continuous batching scheduler with preemption support
- Expose async generate API for GRPO trainer to consume
- Shared prefix caching for same-prompt G generations (prefill once, decode G times)

**Pros:** Near-vLLM inference performance, in-process with training engine  
**Cons:** Large engineering effort, upstream DeepSpeed changes required  
**Expected gain:** ~5-10x generate speedup with prefix caching

---

## Dependency Chain

```
Layer 1 (days)  →  Layer 2 (1-2 weeks)  →  Layer 3 (months)
```

Layer 1 is independent and gives immediate benefit.
Layer 2 builds on Layer 1's custom decode loop.
Layer 3 is a strategic long-term investment.

## Relevant Files

- `dschat/rlhf/grpo_trainer.py` — current generate logic
- `deepspeed/inference/v2/scheduling/` — existing scheduler infrastructure
- `deepspeed/inference/v2/ragged/` — ragged batching primitives
- `deepspeed/ops/transformer/` — fused attention kernels
