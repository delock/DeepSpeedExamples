# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""
MBPP reward function for GRPO training.

Usage:
    from dschat.utils.reward.mbpp_reward import MBPPRewardFn
    reward_fn = MBPPRewardFn(timeout=5.0)
    scores = reward_fn(prompts_text, responses_text)
"""

import subprocess
import sys
import tempfile
import os
from typing import List, Dict, Optional
from datasets import load_dataset


# --------------------------------------------------------------------------- #
# Code execution                                                                #
# --------------------------------------------------------------------------- #

def _execute_mbpp(code: str, test_list: List[str], test_setup_code: str,
                  timeout: float) -> bool:
    """
    Execute `code` + test assertions in a subprocess.
    Returns True if all assertions pass within `timeout` seconds.
    """
    parts = []
    if test_setup_code:
        parts.append(test_setup_code)
    parts.append(code)
    parts.extend(test_list)
    full_source = "\n".join(parts)

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
        )
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


# --------------------------------------------------------------------------- #
# Prompt → task mapping                                                         #
# --------------------------------------------------------------------------- #

MBPP_PROMPT_TEMPLATE = """# Task: {text}
# Write a Python function to solve the task.

"""


def format_mbpp_prompt(text: str) -> str:
    """Format an MBPP task description into a code-generation prompt."""
    return MBPP_PROMPT_TEMPLATE.format(text=text)


def _build_mbpp_index(dataset) -> Dict[str, Dict]:
    """
    Build a dict: normalised prompt text -> {test_list, test_setup_code, code, text}
    """
    index = {}
    for ex in dataset:
        prompt = format_mbpp_prompt(ex["text"])
        key = prompt.strip()
        index[key] = {
            "text": ex["text"],
            "code": ex["code"],
            "test_list": ex["test_list"],
            "test_setup_code": ex.get("test_setup_code", ""),
        }
    return index


def _find_mbpp_task(index: Dict[str, Dict], prompt_text: str) -> Optional[Dict]:
    """Look up a task by prompt text."""
    key = prompt_text.strip()
    if key in index:
        return index[key]

    # Fuzzy fallback: substring matching
    for task_prompt, task in index.items():
        if task_prompt in prompt_text or prompt_text in task_prompt:
            return task

    return None


# --------------------------------------------------------------------------- #
# Public API                                                                    #
# --------------------------------------------------------------------------- #

class MBPPRewardFn:
    """
    Callable reward function based on MBPP unit-test execution.

    Signature matches the GRPO trainer interface:
        reward_fn(prompts_text: List[str], responses_text: List[str]) -> List[float]

    Rewards:
        +1.0  — generated code passes all assertions
         0.0  — code fails

    Args:
        timeout: seconds allowed per subprocess execution.
        split: which MBPP split to use for the reward index.
               Use the same split as training data.
    """

    def __init__(
        self,
        timeout: float = 5.0,
        split: str = "train",
    ):
        self.timeout = timeout
        ds = load_dataset("google-research-datasets/mbpp", "full", split=split)
        self._index = _build_mbpp_index(ds)

    def __call__(
        self,
        prompts_text: List[str],
        responses_text: List[str],
    ) -> List[float]:
        assert len(prompts_text) == len(responses_text)
        scores = []
        for prompt, response in zip(prompts_text, responses_text):
            scores.append(self._score_one(prompt, response))
        return scores

    def _score_one(self, prompt: str, response: str) -> float:
        task = _find_mbpp_task(self._index, prompt)
        if task is None:
            return 0.0

        # The model generates code directly; use response as-is
        code = response

        passed = _execute_mbpp(
            code=code,
            test_list=task["test_list"],
            test_setup_code=task["test_setup_code"],
            timeout=self.timeout,
        )
        return 1.0 if passed else 0.0

    @property
    def task_count(self) -> int:
        return len(self._index)
