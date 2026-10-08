<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- DeepSpeed Team -->

# Credit-bounded saved-activation offload

This benchmark compares pageable and CUDA-native pinned host buffers for saved
MLP intermediates. It is an experiment harness, not a DeepSpeed configuration
or production API.

The driver first performs a discovery subprocess and freezes the selected
saved-tensor manifest. Every timed subprocess repeats discovery only to reject
signature drift, then loads that same manifest. Selection is never changed
online. A forward credit is acquired in the target MLP's pre-hook, before its
targeted activations can be allocated. Layers the frozen manifest does not
offload never take a credit, so they cannot stall in one arm only. Forward
credits are returned only after D2H completion. A separate backward pool holds
restored GPU tensors until a compute-stream event records consumption.

`K` is a request, not a contract. A layer that exposes fewer offloadable
storages than `--offload-tensors` is offloaded with what it has, and the arm
reports `k_effective_min` and `k_effective_max`. Only a layer with nothing
offloadable is an error.

## Quick checks

The scheduler smoke is CPU-safe:

```bash
python benchmarks/pin_memory/saved_activation_offload/bench.py --smoke
```

The benchmark itself is CUDA-only. Synthetic runs are for correctness and
credit-pressure validation:

```bash
ulimit -l unlimited
python benchmarks/pin_memory/saved_activation_offload/bench.py \
  --hidden 1024 --layers 8 --batch 1 --seq 2048

python benchmarks/pin_memory/saved_activation_offload/bench.py \
  --hidden 1024 --layers 8 --batch 1 --seq 2048 --n-sweep
```

Published model runs use Qwen3 with checkpointing disabled:

```bash
python benchmarks/pin_memory/saved_activation_offload/bench.py \
  --model Qwen/Qwen3-8B --batch 1 --seq 4096 --offload-optimizer \
  --manifest-out qwen3-8b-saved-activation-manifest.json
```

Defaults are `N=2`, `M=2`, `K=3`, one skipped final layer, 5 warmup steps,
and 20 measured steps. The normal driver runs fresh subprocesses in A/B/B/A
order. `--n-sweep` runs `N={1,2,4}` with the same manifest. `ARMRESULT` and
`DRIVERRESULT` lines use the shared pin-memory benchmark JSON helpers.

## Arms and invariants

- Both arms use the same model, selector, credits, stream protocol, and native
  CUDA registration environment. The pinned arm alone calls
  `get_accelerator().pin_memory(..., make_copy=False)`.
- Host buffers are dense. Restore allocates with the saved shape, stride,
  dtype, and device.
- D2H and H2D use one copy stream, with an explicit per-group D2H-to-H2D
  event dependency. H2D waits on that event but deliberately not on the
  compute stream, which would collapse the prefetch window into just-in-time
  reloads. Restored buffers are allocated on the copy stream, so the unpack
  hook's `record_stream` is what governs their reuse.
- Every offloaded group must be reloaded and consumed before the context
  exits. A module backward hook that silently does not fire would leave the
  backward pool unbounded for the step while the timings still look
  plausible, so the arm fails rather than reporting that step.
- Repeated unpack of a slot during the first backward returns one cached GPU
  tensor. A second backward traversal of the same graph is rejected.
- Credits bound GPU source and destination groups, not host RAM. Host capacity
  must cover the sum of selected manifest bytes. Pinned capacity must also fit
  the process memlock limit.
- The context must enclose both forward and backward. Cleanup drains armed
  events, reclaims D2H-only host buffers, cancels unarmed credits, and unpins
  every pool-owned allocation on close, including exceptional paths.

The default optimizer is momentum-free SGD. `--offload-optimizer` is the
published Qwen3 comparison: ZeRO-2 CPU AdamW, with pin covering both the
activation buffers and the optimizer host buffers.

`--mode none` is the numerical baseline. It installs matching saved-tensor and
full-backward hook machinery and records target-MLP entry events, which makes
it the source of the uncontaminated forward period. `--mode remat` is optional
context only; it requires `--model`, enables non-reentrant model gradient
checkpointing, and runs `CheckpointHiddenStatesOffload` pinned, since that is
what a practitioner enabling checkpoint offload today would get.

## Current limits

Only single-GPU CUDA, dense non-overlapping offset-zero views, and one matched
MLP per decoder layer are supported. A storage saved through multiple different
shape/stride signatures is left resident; one dense slot cannot restore those
aliases safely. XPU, B70, MoE, Qwen3.5, distributed training, ZeRO-3,
compilation, and repeated backward are out of scope.

The JSON reports copy-stream event time and effective D2H/H2D bandwidth, step
percentiles, host enqueue time, selected and allocated host bytes,
forward/backward wait time, blocking acquisitions, and both credit high-water
marks. Every CUDA end event is synchronized before `elapsed_time` is read.

### Reading the timing fields

`t_d_layer_ms` sums each D2H copy bracketed on its own, and
`d2h_effective_gbps` derives from it. `d2h_group_span_ms_total` is the wider
first-save-to-scope-exit span; it is the right quantity for credit-hold
analysis but it also covers the compute this hook waits on between saves, so
it must not be used for bandwidth.

`inter_mlp_entry_mean_ms` is the mean interval between consecutive target-MLP
forward entries. Under offload this is the observed layer period, which is the
quantity the steady-state model predicts, but it absorbs credit stalls and, in
the pageable arm, the host-synchronous copy itself. It is therefore not a
forward-compute time, and a ratio built on it would fall precisely when
transfer is the binding constraint. Take the reference forward period from a
`--mode none` run with the same model, batch, sequence, and hook set.

`mlp_span_mean_ms` is measured from the post-acquire enqueue point to module
exit, so it excludes the host credit wait, but it can still include compute
already queued ahead of the module. `r_d2h_over_mlp_span` is the feasibility
ratio built from it.

`hbm_peak_allocated_gb` and `hbm_peak_reserved_gb` cover the measured steps
only. Peak stats are reset after discovery and again after warmup, because
discovery is a full forward with every activation resident; its own peak is
reported separately as `hbm_peak_discovery_gb`. Discovery is also the step
most likely to run out of memory at the configurations where transfer is most
constrained.

The driver reports `pin_gap_s`, `arm_spread_s`, and `pin_separated`. Treat a
`pin_speedup` with `pin_separated` false as noise.

Backward prefetch is issued without blocking, but the autograd thread still
calls the replacement issue from inside a module backward hook, so backward
overlap is weaker than a fully decoupled prefetcher would achieve.
