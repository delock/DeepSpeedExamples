# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Model-tensor CPU offload: pinned vs unpinned host memory (ZeRO stage 1/2/3).

Default is ZeRO-3 with offload_optimizer + offload_param. Stages 1 and 2
offload the optimizer only. Pinned arms use DS_PIN_MEMORY_BACKEND=native.
Pass --ablate-register for registered vs unregistered (CUDA).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_PIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PIN_ROOT not in sys.path:
    sys.path.insert(0, _PIN_ROOT)

from common import (_backbone_hidden, _enable_non_reentrant_checkpointing, _install_tiled_mlp,
                    _tiled_fused_ce_loss, apply_liger, build_hf_causal_lm, dist_env, param_numel,
                    patch_cpp_extension_drop_cxx17, print_arm_result, print_driver_result, run_arm_subprocess)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",
                        type=str,
                        default=None,
                        help="HF model id; architecture from config, random weights. Omit for synthetic.")
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--zero-stage", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--pin", type=int, default=None, help="internal: offload pin_memory 0/1")
    parser.add_argument("--register", type=int, default=None, help="internal: DS_PIN_MEMORY_REGISTER_DEVICE")
    parser.add_argument("--ablate-register", action="store_true")
    parser.add_argument("--act-mem-opt",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Liger FLCE-only (fused linear cross-entropy) when --model is set. "
                             "Default off: existing invocations are unaffected.")
    parser.add_argument("--tiled-mlp",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help="When --act-mem-opt: also wrap SwiGLU MLPs with TiledMLP. No effect "
                             "without --act-mem-opt.")
    parser.add_argument("--gradient-checkpointing",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Non-reentrant gradient checkpointing. Default off: existing "
                             "invocations are unaffected.")
    return parser.parse_args()


def _build_synthetic(hidden, layers):
    import torch

    class Block(torch.nn.Module):

        def __init__(self, h):
            super().__init__()
            self.fc1 = torch.nn.Linear(h, 4 * h)
            self.fc2 = torch.nn.Linear(4 * h, h)

        def forward(self, x):
            return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

    class Net(torch.nn.Module):

        def __init__(self, h, n):
            super().__init__()
            self.emb = torch.nn.Embedding(32000, h)
            self.blocks = torch.nn.ModuleList([Block(h) for _ in range(n)])
            self.head = torch.nn.Linear(h, 32000, bias=False)

        def forward(self, idx):
            x = self.emb(idx)
            for block in self.blocks:
                x = x + block(x)
            return self.head(x).sum()

    return Net(hidden, layers), 32000


def _cpu_bytes(tensor):
    import torch
    if tensor is None or not torch.is_tensor(tensor):
        return 0
    if tensor.device.type != "cpu":
        return 0
    return int(tensor.numel()) * int(tensor.element_size())


def _measure_offload_gb(engine):
    """CPU-resident param shards + optimizer/master-weight tensors after a step."""
    seen = set()
    param_bytes = 0
    optim_bytes = 0

    def add(bucket, tensor):
        nonlocal param_bytes, optim_bytes
        nbytes = _cpu_bytes(tensor)
        if nbytes == 0:
            return
        key = int(tensor.data_ptr())
        if key in seen:
            return
        seen.add(key)
        if bucket == "param":
            param_bytes += nbytes
        else:
            optim_bytes += nbytes

    for param in engine.module.parameters():
        add("param", getattr(param, "ds_tensor", param))
        add("param", param)

    optimizer = engine.optimizer
    for tensor in getattr(optimizer, "fp16_partitioned_groups_flat", []) or []:
        add("param", tensor)
    for tensor in getattr(optimizer, "fp32_partitioned_groups_flat", []) or []:
        add("optim", tensor)
        add("optim", getattr(tensor, "grad", None))
        add("optim", getattr(tensor, "exp_avg", None))
        add("optim", getattr(tensor, "exp_avg_sq", None))
        add("optim", getattr(tensor, "exp_avg_cpu_data", None))
        add("optim", getattr(tensor, "exp_avg_sq_cpu_data", None))

    inner = getattr(optimizer, "optimizer", optimizer)
    state = getattr(inner, "state", None)
    if isinstance(state, dict):
        for group_state in state.values():
            if not isinstance(group_state, dict):
                continue
            for value in group_state.values():
                add("optim", value)

    gb = 1024**3
    return {
        "offload_param_gb": round(param_bytes / gb, 2),
        "offload_optim_gb": round(optim_bytes / gb, 2),
        "offload_gb": round((param_bytes + optim_bytes) / gb, 2),
    }


def _zero_config(stage, pin):
    offload_optimizer = {"device": "cpu", "pin_memory": bool(pin)}
    cfg = {"stage": stage, "offload_optimizer": offload_optimizer}
    if stage == 3:
        cfg["offload_param"] = {"device": "cpu", "pin_memory": bool(pin)}
    return cfg


def run_arm(args):
    os.environ["DS_PIN_MEMORY_BACKEND"] = "native"
    os.environ["DS_PIN_MEMORY_REGISTER_DEVICE"] = str(args.register if args.pin else 0)
    dist_env()
    patch_cpp_extension_drop_cxx17()

    import torch
    import deepspeed

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch,
        "gradient_accumulation_steps": 1,
        "bf16": {
            "enabled": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4
            }
        },
        "zero_optimization": _zero_config(args.zero_stage, args.pin),
    }

    hf_transformers = None
    hf_model_type = None
    act_mem_opt = []
    use_tiled_ce = False
    if args.model:
        if args.act_mem_opt:
            try:
                act_mem_opt.extend(apply_liger(args.model))
            except Exception as exc:
                print(f"liger unavailable ({type(exc).__name__}: {exc})", flush=True)

        # Partition at construct time so H200-scale Qwen3/Qwen3.5 fit host RAM.
        if args.zero_stage == 3:
            with deepspeed.zero.Init(config_dict_or_path=ds_config,
                                     dtype=torch.bfloat16,
                                     remote_device="cpu",
                                     pin_memory=bool(args.pin)):
                model, vocab, hf_transformers, hf_model_type = build_hf_causal_lm(args.model, torch.bfloat16)
        else:
            model, vocab, hf_transformers, hf_model_type = build_hf_causal_lm(args.model, torch.bfloat16)

        if args.gradient_checkpointing:
            _enable_non_reentrant_checkpointing(model)
        if args.act_mem_opt:
            if args.tiled_mlp:
                act_mem_opt.extend(_install_tiled_mlp(model))
            use_tiled_ce = "liger_fused_linear_ce" not in act_mem_opt
            if use_tiled_ce:
                act_mem_opt.append("tiled_fused_ce")

        def forward_loss(engine, ids):
            if use_tiled_ce:
                hidden = _backbone_hidden(engine.module, ids)
                return _tiled_fused_ce_loss(engine.module, hidden, ids)
            return engine(input_ids=ids, labels=ids).loss
    else:
        model, vocab = _build_synthetic(args.hidden, args.layers)

        def forward_loss(engine, ids):
            return engine(ids)

    n_params = sum(param_numel(p) for p in model.parameters())

    engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
    dev = engine.device
    dev_api = getattr(torch, dev.type)
    ids = torch.randint(0, vocab, (args.batch, args.seq), dtype=torch.long, device=dev)

    step_times = []
    for step in range(args.warmup + args.steps):
        dev_api.synchronize()
        t0 = time.perf_counter()
        loss = forward_loss(engine, ids)
        engine.backward(loss)
        engine.step()
        dev_api.synchronize()
        if step >= args.warmup:
            step_times.append(time.perf_counter() - t0)

    def _mem(fn):
        try:
            return round(fn() / 1e9, 2)
        except Exception:
            return None

    offload = _measure_offload_gb(engine)
    print_arm_result({
        "experiment": "model_tensor_offload",
        "zero_stage": args.zero_stage,
        "pin_memory": bool(args.pin),
        "register": bool(args.register) if args.pin else None,
        "act_mem_opt": act_mem_opt,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "device": dev.type,
        "model": args.model or f"synthetic-h{args.hidden}-l{args.layers}",
        "hf_model_type": hf_model_type,
        "transformers": hf_transformers,
        "params_b": round(n_params / 1e9, 3),
        "batch": args.batch,
        "seq": args.seq,
        "tokens_per_step": args.batch * args.seq,
        "steps": len(step_times),
        "step_avg_s": sum(step_times) / len(step_times),
        "step_min_s": min(step_times),
        "gpu_peak_gb": _mem(dev_api.max_memory_allocated),
        **offload,
    })


def _passthrough(args):
    extra = [
        "--zero-stage",
        str(args.zero_stage),
        "--hidden",
        str(args.hidden),
        "--layers",
        str(args.layers),
        "--batch",
        str(args.batch),
        "--seq",
        str(args.seq),
        "--steps",
        str(args.steps),
        "--warmup",
        str(args.warmup),
    ]
    extra.append("--act-mem-opt" if args.act_mem_opt else "--no-act-mem-opt")
    extra.append("--tiled-mlp" if args.tiled_mlp else "--no-tiled-mlp")
    extra.append("--gradient-checkpointing" if args.gradient_checkpointing else "--no-gradient-checkpointing")
    if args.model:
        extra.extend(["--model", args.model])
    return extra


def run_driver(args):
    if args.ablate_register:
        arms = [("unpinned", 0, 0), ("pinned-unregistered", 1, 0), ("pinned-registered", 1, 1)]
    else:
        arms = [("unpinned", 0, 0), ("pinned", 1, 1)]
    script = os.path.abspath(__file__)
    results = {}
    for name, pin, reg in arms:
        extra = _passthrough(args) + ["--pin", str(pin), "--register", str(reg)]
        results[name] = run_arm_subprocess(script, extra)

    unpinned = results["unpinned"]
    pinned = results["pinned-registered"] if args.ablate_register else results["pinned"]
    print("\n================ Model-tensor CPU-offload step time ================")
    print(f"stage: {pinned['zero_stage']}  model: {pinned['model']}  params: {pinned['params_b']}B  "
          f"device: {pinned['device']}")
    print(f"batch: {pinned['batch']} x seq: {pinned['seq']}  ({pinned['tokens_per_step']} tokens/step)")
    print(f"offload: params {pinned.get('offload_param_gb', 0):.2f} GB  "
          f"optimizer {pinned.get('offload_optim_gb', 0):.2f} GB  "
          f"total {pinned.get('offload_gb', 0):.2f} GB")
    if pinned.get("act_mem_opt"):
        print(f"act mem opt: {', '.join(pinned['act_mem_opt'])}")
    if pinned.get("gradient_checkpointing"):
        print("gradient checkpointing: enabled")
    print()
    print(f"{'arm':<22}{'avg step (s)':>14}{'min step (s)':>14}{'GPU peak (GB)':>16}{'offload (GB)':>16}")
    for name, _, _ in arms:
        row = results[name]
        print(f"{name:<22}{row['step_avg_s']:>14.3f}{row['step_min_s']:>14.3f}"
              f"{(row['gpu_peak_gb'] or 0):>16.2f}{(row.get('offload_gb') or 0):>16.2f}")
    speedup = unpinned["step_avg_s"] / pinned["step_avg_s"]
    saved = unpinned["step_avg_s"] - pinned["step_avg_s"]
    tok_pin = pinned["tokens_per_step"] / pinned["step_avg_s"]
    tok_unpin = unpinned["tokens_per_step"] / unpinned["step_avg_s"]
    print()
    print(f"pinning speedup: {speedup:.2f}x   saved: {saved:.3f} s/step   "
          f"throughput: {tok_pin:.0f} vs {tok_unpin:.0f} tok/s")
    summary = {"unpinned": unpinned, "pinned": pinned, "speedup": speedup}
    if args.ablate_register:
        summary["pinned-unregistered"] = results["pinned-unregistered"]
    print_driver_result(summary)


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.pin is None:
        run_driver(parsed)
    else:
        run_arm(parsed)
