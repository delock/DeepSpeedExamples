# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Activation / checkpoint hidden-state CPU offload: pin vs pageable with async on.

Holds use_streams=True. Compares use_pin_memory True vs False. Do not treat
this as DeepSpeed #8282's async-vs-blocking table.
"""

from __future__ import annotations

import argparse
import contextlib
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
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--pin", type=int, default=None, help="internal: use_pin_memory 0/1")
    parser.add_argument("--offload-optimizer",
                        action="store_true",
                        help="ZeRO CPU AdamW offload. HF --model only. Stage controlled by --zero-stage.")
    parser.add_argument("--zero-stage",
                        type=int,
                        choices=(2, 3),
                        default=2,
                        help="ZeRO stage used with --offload-optimizer: 2 = optimizer-only CPU offload "
                             "(params stay on GPU, default); 3 = full param+optimizer CPU offload (ablation).")
    parser.add_argument("--act-mem-opt",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Liger + DeepSpeed tiled MLP / tiled fused CE when --model is set.")
    parser.add_argument("--tiled-mlp",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="When --act-mem-opt: also wrap SwiGLU MLPs with TiledMLP (a custom "
                             "autograd.Function). --no-tiled-mlp keeps Liger kernels/fused CE but skips the "
                             "TiledMLP wrap -- isolates whether TiledMLP's own saved-tensor handling is what "
                             "breaks ds_async / ds_ckpt_only's peak memory.")
    parser.add_argument("--no-act-offload",
                        action="store_true",
                        help="Skip the CheckpointHiddenStatesOffload CPU pool; plain gradient-checkpointing "
                             "(recompute only, nothing shipped to host). --pin is ignored in this mode.")
    parser.add_argument("--act-offload-mode",
                        choices=("hf", "ds_async", "ds_ckpt_only"),
                        default="hf",
                        help="hf: CheckpointHiddenStatesOffload context manager spanning the whole step "
                             "(today's default; unpack cache=True, so peak converges with plain gc). "
                             "ds_async: DeepSpeed's native non_reentrant_checkpoint + _ActivationOffloadEngine, "
                             "wired per-layer via checkpointing.configure(checkpoint_in_cpu=True) with no "
                             "step-spanning context manager. "
                             "ds_ckpt_only: isolate just the _gradient_checkpointing_func swap to "
                             "non_reentrant_checkpoint -- no offload engine installed, no "
                             "checkpointing.configure(checkpoint_in_cpu=True) call, --pin has no effect. "
                             "Tests whether DeepSpeed's own checkpoint function (pure recompute, no host "
                             "traffic) matches plain-gc peak on its own. "
                             "--model only; ignored when --no-act-offload is set.")
    parser.add_argument("--debug-mem",
                        action="store_true",
                        help="Print allocator snapshots (alloc/reserved/running-peak) at forward/backward/"
                             "optimizer-step boundaries on the last measured step. Diagnostic only.")
    return parser.parse_args()


def _build_synthetic(hidden, layers, device):
    import torch
    from torch.utils.checkpoint import checkpoint

    class Block(torch.nn.Module):

        def __init__(self, hidden):
            super().__init__()
            self.fc1 = torch.nn.Linear(hidden, 4 * hidden)
            self.fc2 = torch.nn.Linear(4 * hidden, hidden)

        def forward(self, hidden_states):
            return self.fc2(torch.nn.functional.gelu(self.fc1(hidden_states)))

    class Net(torch.nn.Module):

        def __init__(self, hidden, layers):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block(hidden) for _ in range(layers)])

        def forward(self, hidden_states, offload):
            x = hidden_states
            for block in self.blocks:
                offload.mark(x)
                x = x + checkpoint(block, x, use_reentrant=False)
            return x.sum()

    return Net(hidden, layers).to(device)


def run_arm(args):
    # Prefer an externally selected pin backend (e.g. NPU run wrapper exports
    # DS_PIN_MEMORY_BACKEND=torch); default remains the H200 native backend.
    os.environ.setdefault("DS_PIN_MEMORY_BACKEND", "native")
    os.environ["DS_PIN_MEMORY_REGISTER_DEVICE"] = "1" if args.pin else "0"
    patch_cpp_extension_drop_cxx17()

    import torch
    from deepspeed.accelerator import get_accelerator
    from deepspeed.runtime.activation_checkpointing.offload_activations import CheckpointHiddenStatesOffload

    accelerator = get_accelerator()
    if not accelerator.is_available():
        raise RuntimeError(f"No {accelerator.device_name()} device is available")
    accelerator.set_device(0)
    device = accelerator.current_device_name()

    hf_transformers = None
    hf_model_type = None
    engine = None
    act_mem_opt = []
    use_tiled_ce = False
    if args.offload_optimizer and not args.model:
        raise ValueError("--offload-optimizer requires --model")
    if args.no_act_offload and not args.model:
        raise ValueError("--no-act-offload requires --model (synthetic net always calls offload.mark)")
    if args.act_offload_mode in ("ds_async", "ds_ckpt_only") and not args.model:
        raise ValueError(f"--act-offload-mode {args.act_offload_mode} requires --model "
                          "(patches HF GradientCheckpointingLayer)")

    if args.model:
        if args.act_mem_opt or args.offload_optimizer:
            dist_env()
        if args.act_mem_opt:
            try:
                act_mem_opt.extend(apply_liger(args.model))
            except Exception as exc:
                print(f"liger unavailable ({type(exc).__name__}: {exc})", flush=True)

        ds_config = None
        if args.offload_optimizer:
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
                "zero_optimization": {
                    "stage": args.zero_stage,
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": bool(args.pin)
                    },
                },
            }
            if args.zero_stage == 3:
                ds_config["zero_optimization"]["offload_param"] = {
                    "device": "cpu",
                    "pin_memory": bool(args.pin),
                }

        if ds_config is not None and args.zero_stage == 3:
            # Partition params at construct time so H200-scale Qwen3 fits host RAM.
            with deepspeed.zero.Init(config_dict_or_path=ds_config,
                                     dtype=torch.bfloat16,
                                     remote_device="cpu",
                                     pin_memory=bool(args.pin)):
                model, vocab, hf_transformers, hf_model_type = build_hf_causal_lm(args.model, torch.bfloat16)
        else:
            model, vocab, hf_transformers, hf_model_type = build_hf_causal_lm(args.model, torch.bfloat16)

        _enable_non_reentrant_checkpointing(model)
        if args.act_mem_opt:
            if args.tiled_mlp:
                act_mem_opt.extend(_install_tiled_mlp(model))
            use_tiled_ce = "liger_fused_linear_ce" not in act_mem_opt
            if use_tiled_ce:
                act_mem_opt.append("tiled_fused_ce")
            act_mem_opt.append("flash_attn_2_if_available")
        if args.offload_optimizer:
            engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
            device = engine.device
            n_params = sum(param_numel(p) for p in engine.module.parameters())
            ids = torch.randint(0, vocab, (args.batch, args.seq), dtype=torch.long, device=device)
            module = engine.module

            def forward_loss(offload):
                if use_tiled_ce:
                    hidden = _backbone_hidden(module, ids)
                    return _tiled_fused_ce_loss(module, hidden, ids)
                return engine(input_ids=ids, labels=ids).loss

            def zero_grad():
                engine.zero_grad()

            def optimizer_step():
                engine.step()
        else:
            model = model.to(device=device, dtype=torch.bfloat16)
            n_params = sum(param_numel(p) for p in model.parameters())
            ids = torch.randint(0, vocab, (args.batch, args.seq), dtype=torch.long, device=device)

            def forward_loss(offload):
                if use_tiled_ce:
                    hidden = _backbone_hidden(model, ids)
                    return _tiled_fused_ce_loss(model, hidden, ids)
                return model(input_ids=ids, labels=ids).loss

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            def zero_grad():
                optimizer.zero_grad(set_to_none=True)

            def optimizer_step():
                optimizer.step()

        ds_async_engine = None
        if args.act_offload_mode in ("ds_async", "ds_ckpt_only") and not args.no_act_offload:
            import deepspeed.runtime.activation_checkpointing.checkpointing as ds_ckpt
            from transformers import GradientCheckpointingLayer

            if args.act_offload_mode == "ds_async":
                from deepspeed.runtime.activation_checkpointing.offload_activations import _ActivationOffloadEngine

                ds_async_engine = _ActivationOffloadEngine(keep_last_count=1,
                                                            min_offload_bytes=0,
                                                            use_pin_memory=bool(args.pin))
                ds_ckpt._cpu_offload_engine = ds_async_engine
                ds_ckpt.configure(mpu_=None, checkpoint_in_cpu=True)
            # ds_ckpt_only: swap only the checkpoint function itself. No engine,
            # no checkpoint_in_cpu -- CPU_CHECKPOINT stays unset, so
            # non_reentrant_checkpoint takes the pure-recompute path (no host
            # traffic at all). Isolates the swap's own memory behavior from the
            # offload engine's.
            backbone = engine.module if engine is not None else model
            patched_layers = 0
            for m in backbone.modules():
                if isinstance(m, GradientCheckpointingLayer):
                    m.gradient_checkpointing = True
                    m._gradient_checkpointing_func = ds_ckpt.non_reentrant_checkpoint
                    patched_layers += 1
            if patched_layers == 0:
                raise RuntimeError(f"{args.act_offload_mode}: no GradientCheckpointingLayer modules found to patch")
    else:
        ds_async_engine = None
        model = _build_synthetic(args.hidden, args.layers, device)
        n_params = sum(p.numel() for p in model.parameters())
        x = torch.randn(args.batch, args.seq, args.hidden, device=device, requires_grad=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        def forward_loss(offload):
            return model(x, offload)

        def zero_grad():
            optimizer.zero_grad(set_to_none=True)

        def optimizer_step():
            optimizer.step()

    use_hf_offload = args.act_offload_mode == "hf" and not args.no_act_offload
    offload = CheckpointHiddenStatesOffload(use_pin_memory=bool(args.pin),
                                             use_streams=True,
                                             min_offload_bytes=0,
                                             keep_last_count=1) if use_hf_offload else None

    def _mem_gb():
        mod = torch.get_device_module(str(device).split(":")[0])
        return {
            "alloc": round(mod.memory_allocated() / 1e9, 2),
            "reserved": round(mod.memory_reserved() / 1e9, 2),
            "peak_alloc": round(mod.max_memory_allocated() / 1e9, 2),
        }

    debug_mem_active = [False]
    if args.debug_mem and args.model:
        backbone = engine.module if engine is not None else model
        layers_list = getattr(getattr(backbone, "model", backbone), "layers", None)
        if layers_list:
            def _fwd_hook(idx):
                def hook(module_, inp, out):
                    if debug_mem_active[0]:
                        print(f"MEMDEBUG fwd-layer{idx} {_mem_gb()}", flush=True)

                return hook

            def _bwd_hook(idx):
                def hook(module_, grad_input, grad_output):
                    if debug_mem_active[0]:
                        print(f"MEMDEBUG bwd-layer{idx} {_mem_gb()}", flush=True)

                return hook

            for i, layer in enumerate(layers_list):
                layer.register_forward_hook(_fwd_hook(i))
                layer.register_full_backward_hook(_bwd_hook(i))
        else:
            print("MEMDEBUG: no decoder .model.layers found, skipping layer hooks", flush=True)

    def _reset_peak():
        try:
            torch.get_device_module(str(device).split(":")[0]).reset_peak_memory_stats()
        except Exception:
            try:
                accelerator.reset_peak_memory_stats()
            except Exception:
                pass

    def _peak():
        try:
            return round(torch.get_device_module(str(device).split(":")[0]).max_memory_allocated() / 1e9, 2)
        except Exception:
            try:
                return round(accelerator.max_memory_allocated() / 1e9, 2)
            except Exception:
                return None

    last_stats = None
    step_times = []
    step_peaks = []
    for step in range(args.warmup + args.steps):
        if step >= args.warmup:
            # Reset every measured step so the recorded peak reflects that
            # step's own steady-state memory, not a one-off transient (e.g.
            # ZeRO gradient/reduce-scatter buffer materialization) from an
            # earlier step in the measured window. gpu_peak_gb below uses
            # only the last step's peak.
            _reset_peak()
        debug_last_step = args.debug_mem and step == args.warmup + args.steps - 1
        debug_mem_active[0] = debug_last_step
        accelerator.synchronize()
        t0 = time.perf_counter()
        zero_grad()
        if debug_last_step:
            print(f"MEMDEBUG pre-forward {_mem_gb()}", flush=True)
        with (offload if offload is not None else contextlib.nullcontext()):
            loss = forward_loss(offload)
            if debug_last_step:
                accelerator.synchronize()
                print(f"MEMDEBUG post-forward {_mem_gb()}", flush=True)
            if engine is not None:
                engine.backward(loss)
            else:
                loss.backward()
        if debug_last_step:
            accelerator.synchronize()
            print(f"MEMDEBUG post-backward {_mem_gb()}", flush=True)
        optimizer_step()
        if debug_last_step:
            accelerator.synchronize()
            print(f"MEMDEBUG post-optimizer-step {_mem_gb()}", flush=True)
        debug_mem_active[0] = False
        accelerator.synchronize()
        if step >= args.warmup:
            step_times.append(time.perf_counter() - t0)
            step_peaks.append(_peak())
            active_stats = offload.stats if offload is not None else (ds_async_engine.stats
                                                                        if ds_async_engine is not None else None)
            if active_stats is not None:
                last_stats = {
                    "marked_tensors": getattr(active_stats, "marked_tensors", 0),
                    "offloaded_tensors": active_stats.offloaded_tensors,
                    "offloaded_bytes": active_stats.offloaded_bytes,
                    "restored_bytes": active_stats.restored_bytes,
                }
        if offload is not None:
            offload.reset()
        if ds_async_engine is not None:
            ds_async_engine.reset()

    if (offload is not None or ds_async_engine is not None) and (last_stats is None
                                                                   or last_stats["offloaded_tensors"] == 0):
        raise RuntimeError("activation offload moved 0 tensors; checkpoint marker did not fire")

    gb = 1024**3
    training_step = "causal_lm_forward_backward_adamw"
    if args.offload_optimizer:
        training_step = f"causal_lm_fwd_bwd_zero{args.zero_stage}_cpu_adamw"
    print_arm_result({
        "experiment": "activation_offload",
        "use_pin_memory": bool(args.pin),
        "use_streams": True,
        "no_act_offload": bool(args.no_act_offload),
        "act_offload_mode": args.act_offload_mode if not args.no_act_offload else None,
        "training_step": training_step,
        "offload_optimizer": bool(args.offload_optimizer),
        "zero_stage": args.zero_stage if args.offload_optimizer else None,
        "act_mem_opt": act_mem_opt,
        "device": accelerator.device_name(),
        "model": args.model or f"synthetic-h{args.hidden}-l{args.layers}",
        "hf_model_type": hf_model_type,
        "transformers": hf_transformers,
        "params_b": round(n_params / 1e9, 3),
        "hidden": args.hidden,
        "layers": args.layers,
        "batch": args.batch,
        "seq": args.seq,
        "tokens_per_step": args.batch * args.seq,
        "steps": len(step_times),
        "step_avg_s": sum(step_times) / len(step_times),
        "step_min_s": min(step_times),
        "gpu_peak_gb": step_peaks[-1] if step_peaks else None,
        "offload_gb": round(last_stats["offloaded_bytes"] / gb, 3) if last_stats else 0,
        "restore_gb": round(last_stats["restored_bytes"] / gb, 3) if last_stats else 0,
        "offloaded_tensors": last_stats["offloaded_tensors"] if last_stats else 0,
        "marked_tensors": last_stats["marked_tensors"] if last_stats else 0,
    })


def run_driver(args):
    script = os.path.abspath(__file__)
    base = [
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
    if args.model:
        base.extend(["--model", args.model])
    if args.offload_optimizer:
        base.append("--offload-optimizer")
    base.extend(["--zero-stage", str(args.zero_stage)])
    base.append("--act-mem-opt" if args.act_mem_opt else "--no-act-mem-opt")
    base.append("--tiled-mlp" if args.tiled_mlp else "--no-tiled-mlp")
    base.extend(["--act-offload-mode", args.act_offload_mode])
    results = {}
    for name, pin in (("pageable", 0), ("pinned", 1)):
        results[name] = run_arm_subprocess(script, base + ["--pin", str(pin)])

    pageable = results["pageable"]
    pinned = results["pinned"]
    print("\n================ Activation offload (async on) ================")
    print(f"device: {pinned['device']}  model: {pinned['model']}  params: {pinned['params_b']}B")
    print(f"batch: {pinned['batch']} x seq: {pinned['seq']}  ({pinned['tokens_per_step']} tokens/step)  "
          f"use_streams: True")
    if pinned.get("offload_optimizer"):
        if pinned.get("zero_stage") == 3:
            print("optimizer: ZeRO-3 CPU offload  (params+optimizer on CPU)")
        else:
            print("optimizer: ZeRO-2 CPU offload  (params on GPU)")
    if pinned.get("act_mem_opt"):
        print(f"act mem opt: {', '.join(pinned['act_mem_opt'])}")
    print(f"activation offload: {pinned.get('offload_gb', 0):.3f} GB D2H  "
          f"{pinned.get('restore_gb', 0):.3f} GB H2D  "
          f"tensors: {pinned.get('offloaded_tensors', 0)}")
    print(f"{'arm':<22}{'avg step (s)':>14}{'min step (s)':>14}{'GPU peak (GB)':>16}{'act offload (GB)':>18}")
    for name in ("pageable", "pinned"):
        row = results[name]
        print(f"{name:<22}{row['step_avg_s']:>14.3f}{row['step_min_s']:>14.3f}"
              f"{(row['gpu_peak_gb'] or 0):>16.2f}{(row.get('offload_gb') or 0):>18.3f}")
    speedup = pageable["step_avg_s"] / pinned["step_avg_s"]
    print()
    print(f"pin vs pageable step-time ratio: {speedup:.2f}x (async held on)")
    print_driver_result({"pageable": pageable, "pinned": pinned, "speedup": speedup})


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.pin is None:
        run_driver(parsed)
    else:
        run_arm(parsed)
