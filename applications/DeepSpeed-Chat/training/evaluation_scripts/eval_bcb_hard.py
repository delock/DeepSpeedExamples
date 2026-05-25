#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

"""
BigCodeBench-Hard (BCB-Hard) evaluation script.

Usage:
    python eval_bcb_hard.py --model_path ./output/actor
    python eval_bcb_hard.py --model_path Qwen/Qwen2.5-0.5B  # baseline

Evaluates pass@1 on 148 BCB-Hard problems using greedy decoding.
"""

import argparse
import subprocess
import sys
import tempfile
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="BCB-Hard pass@1 evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"])
    return parser.parse_args()


def load_model(model_path, device, dtype):
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    torch_dtype = dtype_map[dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype
    ).to(device)
    model.eval()
    return model, tokenizer


def generate_solution(model, tokenizer, prompt, max_new_tokens, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=2048).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def execute_bcb_task(code: str, test: str, entry_point: str, timeout: float) -> bool:
    """
    Execute generated code + BCB-Hard test cases.
    BCB-Hard uses unittest, so we need to run the full test class.
    """
    full_source = f"""{code}

{test}

if __name__ == "__main__":
    unittest.main(exit=False, verbosity=0)
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(full_source)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        # unittest returns 0 on success
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def evaluate(model, tokenizer, dataset, max_new_tokens, timeout, device):
    passed = 0
    total = len(dataset)
    results = []

    for i, ex in enumerate(dataset):
        prompt = ex["complete_prompt"]
        entry_point = ex["entry_point"]
        test = ex["test"]

        # Generate the function body
        response = generate_solution(model, tokenizer, prompt, max_new_tokens, device)

        # Full code = prompt (imports + signature + docstring) + generated body
        full_code = prompt + response

        # Execute
        success = execute_bcb_task(full_code, test, entry_point, timeout)

        if success:
            passed += 1

        results.append({
            "task_id": ex["task_id"],
            "passed": success,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{total}] pass@1: {passed}/{i+1} = {passed/(i+1):.3f}")

    return passed, total


def main():
    args = parse_args()
    dataset = load_dataset("bigcode/bigcodebench-hard", split="v0.1.2")
    print(f"Loaded BCB-Hard: {len(dataset)} problems")
    print(f"Model: {args.model_path}")
    print(f"Max new tokens: {args.max_new_tokens}, Timeout: {args.timeout}s")
    print()

    model, tokenizer = load_model(args.model_path, args.device, args.dtype)
    passed, total = evaluate(
        model, tokenizer, dataset, args.max_new_tokens, args.timeout, args.device
    )

    pass_at_1 = passed / total
    print(f"\n{'='*60}")
    print(f"BCB-Hard pass@1: {passed}/{total} = {pass_at_1:.4f}")
    print(f"{'='*60}")

    return pass_at_1


if __name__ == "__main__":
    main()
