#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

"""
HumanEval pass@1 evaluation script.

Usage:
    python eval_humaneval.py --model_path ./grpo_output_humaneval_qwen0.5b_z2/actor

Evaluates a model on all 164 HumanEval problems using greedy decoding
and reports pass@1 (fraction of problems solved correctly).
"""

import argparse
import sys
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from dschat.utils.reward.humaneval_reward import _execute_code


def parse_args():
    parser = argparse.ArgumentParser(description="HumanEval pass@1 evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max tokens to generate per problem")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="Execution timeout per problem (seconds)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--baseline", action="store_true",
                        help="Also evaluate the base model for comparison")
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="Base model path (for --baseline comparison)")
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
    """Generate a completion for a HumanEval prompt using greedy decoding."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy for pass@1
            pad_token_id=tokenizer.pad_token_id,
        )
    # Extract only the generated part
    generated_ids = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def evaluate_model(model, tokenizer, dataset, max_new_tokens, timeout, device):
    """Run pass@1 evaluation on all HumanEval problems."""
    passed = 0
    total = len(dataset)
    results = []

    for i, ex in enumerate(dataset):
        prompt = ex["prompt"]
        entry_point = ex["entry_point"]
        test = ex["test"]

        # Generate solution
        response = generate_solution(model, tokenizer, prompt, max_new_tokens, device)

        # Reconstruct full code: prompt + generated body
        full_code = prompt + response

        # Execute
        success = _execute_code(
            code=full_code,
            test=test,
            entry_point=entry_point,
            timeout=timeout,
        )

        if success:
            passed += 1

        results.append({
            "task_id": ex["task_id"],
            "passed": success,
            "response_preview": response[:100],
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{total}] pass@1 so far: {passed}/{i+1} = {passed/(i+1):.3f}")

    pass_at_1 = passed / total
    return pass_at_1, passed, total, results


def main():
    args = parse_args()
    dataset = load_dataset("openai/openai_humaneval", split="test")
    print(f"Loaded HumanEval: {len(dataset)} problems")

    # Evaluate trained model
    print(f"\n{'='*60}")
    print(f"Evaluating trained model: {args.model_path}")
    print(f"{'='*60}")
    model, tokenizer = load_model(args.model_path, args.device, args.dtype)
    pass_at_1, passed, total, results = evaluate_model(
        model, tokenizer, dataset, args.max_new_tokens, args.timeout, args.device
    )
    print(f"\n[TRAINED] HumanEval pass@1: {passed}/{total} = {pass_at_1:.4f}")
    del model
    torch.cuda.empty_cache()

    # Optionally evaluate baseline
    if args.baseline:
        base_path = args.base_model_path or "Qwen/Qwen2.5-0.5B"
        print(f"\n{'='*60}")
        print(f"Evaluating baseline model: {base_path}")
        print(f"{'='*60}")
        model, tokenizer = load_model(base_path, args.device, args.dtype)
        base_pass_at_1, base_passed, base_total, _ = evaluate_model(
            model, tokenizer, dataset, args.max_new_tokens, args.timeout, args.device
        )
        print(f"\n[BASELINE] HumanEval pass@1: {base_passed}/{base_total} = {base_pass_at_1:.4f}")
        del model
        torch.cuda.empty_cache()

        print(f"\n{'='*60}")
        print(f"COMPARISON:")
        print(f"  Baseline:  {base_pass_at_1:.4f}")
        print(f"  Trained:   {pass_at_1:.4f}")
        print(f"  Delta:     {pass_at_1 - base_pass_at_1:+.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
