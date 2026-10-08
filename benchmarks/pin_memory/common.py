# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""Shared helpers for pin_memory experiment drivers (subprocess arms, JSON lines)."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys


def free_port():
    # A stale listener from an interrupted rank makes the next init hang in a
    # collective, so always rendezvous on a fresh ephemeral port.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def dist_env():
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(free_port()),
        RANK="0",
        WORLD_SIZE="1",
        LOCAL_RANK="0",
    )


def patch_cpp_extension_drop_cxx17():
    # torch-nightly requires C++20; SYCL toolchain flags may carry -std=c++17,
    # which (appearing last) downgrades the dialect and breaks torch headers.
    import torch.utils.cpp_extension as cpp_ext

    orig_load = cpp_ext.load

    def load_without_cxx17(*args, **kwargs):
        for key in ("extra_cflags", "extra_cxxflags"):
            if kwargs.get(key):
                kwargs[key] = [flag for flag in kwargs[key] if flag != "-std=c++17"]
        return orig_load(*args, **kwargs)

    cpp_ext.load = load_without_cxx17


def print_arm_result(result):
    print("ARMRESULT=" + json.dumps(result), flush=True)


def print_driver_result(summary):
    print("DRIVERRESULT=" + json.dumps(summary), flush=True)


def parse_arm_result(stdout):
    for line in stdout.splitlines():
        if line.startswith("ARMRESULT="):
            return json.loads(line[len("ARMRESULT="):])
    return None


def run_arm_subprocess(script_path, extra_args, env=None):
    command = [sys.executable, os.path.abspath(script_path), *extra_args]
    print(f"[driver] {' '.join(command)}", flush=True)
    proc = subprocess.run(command, env=env or os.environ.copy(), capture_output=True, text=True)
    arm = parse_arm_result(proc.stdout)
    if arm is None:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"arm produced no ARMRESULT (rc={proc.returncode})")
    return arm


def param_numel(param):
    return int(getattr(param, "ds_numel", param.numel()))


def build_hf_causal_lm(model_id, torch_dtype):
    """Config + random weights. Prefer text-only if the hub card is multimodal.

    Skip Hugging Face ``post_init`` weight fill: under ``deepspeed.zero.Init``
    parameters are already partitioned (local ``.shape`` can be 0), so Qwen3.5
    ``copy_`` init and ``Embedding.weight.shape[0]`` as vocab both break.
    """
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM
    try:
        # transformers 5.16 (Qwen3.5 pin): moved out of modeling_utils.
        from transformers.initialization import no_init_weights
    except ImportError:
        from transformers.modeling_utils import no_init_weights

    print(f"transformers={transformers.__version__} model_id={model_id}", flush=True)
    config = AutoConfig.from_pretrained(model_id)
    text_config = config
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        text_config = get_text_config() or config
    elif getattr(config, "text_config", None) is not None:
        text_config = config.text_config

    model_type = getattr(text_config, "model_type", None) or getattr(config, "model_type", "")
    if "moe" in str(model_type):
        raise ValueError(f"MoE model_type {model_type!r} is out of scope for this bench")

    def from_config(**extra):
        try:
            return AutoModelForCausalLM.from_config(text_config, dtype=torch_dtype, **extra)
        except TypeError:
            return AutoModelForCausalLM.from_config(text_config, torch_dtype=torch_dtype, **extra)

    # Pin/offload wall time does not need pretrained-style init values.
    with no_init_weights():
        try:
            model = from_config(attn_implementation="flash_attention_2")
        except Exception as exc:
            print(f"flash_attention_2 unavailable ({type(exc).__name__}: {exc}); default attn", flush=True)
            model = from_config()

    vocab = int(getattr(text_config, "vocab_size", 0) or getattr(config, "vocab_size", 0) or 0)
    if vocab <= 0:
        weight = model.get_input_embeddings().weight
        vocab = int(getattr(weight, "ds_shape", weight.shape)[0])
    if vocab <= 0:
        raise ValueError(f"could not determine vocab_size for {model_id}")
    return model, vocab, transformers.__version__, model_type
