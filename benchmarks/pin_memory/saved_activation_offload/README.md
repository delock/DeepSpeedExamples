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

## Environment setup (H200, torch 2.14 + flash-attn 2.8.3)

The widened whole-decoder-layer scope and `--act-mem-opt` (Liger FLCE-only)
runs below need a C++20-built flash-attn against a `torch==2.14.0+cu130`
install; the stock pip wheel is C++17 and will not load against that torch
build.

```bash
# fresh conda env cloned from the pod's existing `dev` env
conda create -y -n torch214 --clone dev
conda activate torch214
PY=$(command -v python3)

"$PY" -m pip install --force-reinstall 'torch==2.14.0+cu130' \
  --index-url https://download.pytorch.org/whl/cu130
"$PY" -c 'import torch; assert torch.__version__ == "2.14.0+cu130", torch.__version__'

# remove the stale (C++17) flash-attn wheel
"$PY" -m pip uninstall -y flash-attn || true
SITE=$("$PY" -c 'import site; print(site.getsitepackages()[0])')
rm -f "$SITE"/flash_attn_2_cuda*.so
rm -rf "$SITE"/flash_attn "$SITE"/flash_attn-*.dist-info

# rebuild flash-attn 2.8.3 from source with -std=c++20
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export MAX_JOBS=8
export TORCH_CUDA_ARCH_LIST="9.0"   # H200
export CXXFLAGS="${CXXFLAGS:-} -std=c++20"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-} -std=c++20"

git clone --depth 1 --branch v2.8.3 \
  https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attn-c20-v283
sed -i 's/-std=c++17/-std=c++20/' /tmp/flash-attn-c20-v283/setup.py
"$PY" -m pip install -U /tmp/flash-attn-c20-v283 --no-build-isolation --no-deps

"$PY" -c 'import torch, flash_attn, flash_attn_2_cuda; print(torch.__version__, flash_attn.__version__)'
```

Runtime env for every sweep below:

```bash
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"   # $REPO = .../sfc_ds_pin_memory
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ulimit -l unlimited   # native pin mlocks host memory
```

## Environment setup (Ascend 910B4, CANN 8.5.1 + torch 2.10)

NPU runs use the same script (the CUDA-only guard, device API, sku field and
one message are generalized to the accelerator abstraction). Setup:

```bash
conda activate cann   # torch 2.10.0+cpu + torch_npu 2.10.0.post7 + transformers 5.19.0
source ~/Ascend/cann-8.5.1/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
ulimit -l unlimited
```

One-click run — `npu/run.sh` pins the NPU experiment parameters (Qwen3-4B,
batch 1 × seq 2048, K=3, `--credits 1`, ZeRO-2 CPU AdamW; the H200 rows run
Qwen3-8B × seq 65536, which does not fit this card):

```bash
cd npu && ./run.sh                      # model from the HF hub
cd npu && MODEL=/path/to/Qwen3-4B ./run.sh   # local checkpoint
```

Manual equivalent of the pinned parameters:

```bash
python bench.py --model Qwen/Qwen3-4B --batch 1 --seq 2048 \
    --offload-tensors 3 --offload-optimizer --act-mem-opt \
    --steps 20 --warmup 5 --credits 1 \
    --manifest-out qwen3-4b-seq2048-k3-manifest.json
```

Notes:

- transformers 5.19.0: the 4.57 default Qwen3 attention forces a hub `kernels`
  package that is unavailable on this host.
- The usable 29.5 GiB card OOMs the full-residency discovery pass at seq 4096
  (~27.1 GiB reserved); seq 2048 works. Qwen3-8B ZeRO-2 static state (32.8 GB)
  exceeds the card, so NPU rows use Qwen3-4B.

## Repro: widened-scope sweep, seq-extrapolated discovery, k/credits ablations

All runs below are Qwen3-8B, batch 1, ZeRO-2 CPU AdamW optimizer offload
(`--offload-optimizer`), Liger FLCE-only kernels (`--act-mem-opt`), 10 steps /
1 warmup, whole-decoder-layer scope (`--offload-modules` default). `--seq`
32768 and above needs `--discover-calib-seqs 4096 8192` — Discovery's own
full-residency forward pass does not fit in 139.8 GB at that scale, so its
manifest is instead produced from two cheap small-seq discovery runs and
linearly extrapolated; see `extrapolate_manifest()` in `bench.py` and verify
against a direct `--seq 8192 --discover-only` manifest before trusting a new
seqlen.

```bash
# main dataset: unlimited offload coverage per layer (k = -1)
for SEQ in 8192 16384 32768 65536; do
  CALIB=(); [ "$SEQ" -ge 32768 ] && CALIB=(--discover-calib-seqs 4096 8192)
  python benchmarks/pin_memory/saved_activation_offload/bench.py \
    --model Qwen/Qwen3-8B --batch 1 --seq "$SEQ" --offload-tensors -1 \
    --offload-optimizer --act-mem-opt --steps 10 --warmup 1 \
    --manifest-out "/tmp/saved-activation-widened-scope-seq${SEQ}.json" "${CALIB[@]}"
done

# k ablation: cap to top-1 tensor/layer by size (counterintuitively *raises*
# peak HBM -- the ~20 non-selected eligible tensors/layer fall back to
# always-resident instead of being offloaded)
for SEQ in 8192 16384 32768 65536; do
  CALIB=(); [ "$SEQ" -ge 32768 ] && CALIB=(--discover-calib-seqs 4096 8192)
  python benchmarks/pin_memory/saved_activation_offload/bench.py \
    --model Qwen/Qwen3-8B --batch 1 --seq "$SEQ" --offload-tensors 1 \
    --offload-optimizer --act-mem-opt --steps 10 --warmup 1 \
    --manifest-out "/tmp/saved-activation-widened-scope-k1-seq${SEQ}.json" "${CALIB[@]}"
done

# credits ablation: halve the forward credit-pool depth instead of capping
# coverage -- this is the knob that actually bounds the resident held-group
# staging buffer (fwd_max_held_bytes/peak HBM), independent of k
for SEQ in 8192 16384 32768 65536; do
  CALIB=(); [ "$SEQ" -ge 32768 ] && CALIB=(--discover-calib-seqs 4096 8192)
  python benchmarks/pin_memory/saved_activation_offload/bench.py \
    --model Qwen/Qwen3-8B --batch 1 --seq "$SEQ" --offload-tensors -1 --credits 1 \
    --offload-optimizer --act-mem-opt --steps 10 --warmup 1 \
    --manifest-out "/tmp/saved-activation-widened-scope-credits1-seq${SEQ}.json" "${CALIB[@]}"
done
```

Matched ckpt-activation (recipe C, `activation_offload/bench.py`) baseline for
the speed/offload-traffic/peak-HBM comparison, same seqlens:

```bash
for SEQ in 8192 16384 32768 65536; do
  for PIN in 0 1; do
    python benchmarks/pin_memory/activation_offload/bench.py \
      --model Qwen/Qwen3-8B --batch 1 --seq "$SEQ" --offload-optimizer \
      --act-mem-opt --pin "$PIN" --steps 10 --warmup 1
  done
done
```

Read `pin_speedup`, `host_offload_bytes`, `hbm_peak_allocated_gb`,
`hbm_peak_reserved_gb`, `fwd_max_held_bytes`/`bwd_max_held_bytes`,
`k_effective_max`, and `credits_n` off the `DRIVERRESULT=`/`ARMRESULT=` JSON
lines for the sao arms; `step_avg_s`, `gpu_peak_gb`, `offload_gb` off
ckpt-activation's.
