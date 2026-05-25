# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""
HumanEval reward function for GRPO training.

Usage:
    from dschat.utils.reward.humaneval_reward import HumanEvalRewardFn
    reward_fn = HumanEvalRewardFn(timeout=5.0)
    scores = reward_fn(prompts_text, responses_text)  # List[float], 0.0 or 1.0
"""

import subprocess
import sys
import textwrap
import tempfile
import os
from typing import List, Dict, Optional
from datasets import load_dataset


# --------------------------------------------------------------------------- #
# Code execution                                                                #
# --------------------------------------------------------------------------- #

_RUNNER_TEMPLATE = """\
{code}

{test}

check({entry_point})
"""


def _execute_code(code: str, test: str, entry_point: str, timeout: float) -> bool:
    """
    Execute `code` + `test` + `check(entry_point)` in a subprocess.
    Returns True if all assertions pass within `timeout` seconds.
    """
    full_source = _RUNNER_TEMPLATE.format(
        code=code,
        test=test,
        entry_point=entry_point,
    )
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

def _build_prompt_index(dataset) -> Dict[str, Dict]:
    """
    Build a dict: normalised_prompt_text -> {test, entry_point, prompt}
    Normalisation: strip whitespace so tokeniser round-trips don't break lookup.
    """
    index = {}
    for ex in dataset:
        key = ex["prompt"].strip()
        index[key] = {
            "prompt": ex["prompt"],
            "test": ex["test"],
            "entry_point": ex["entry_point"],
        }
    return index


def _find_task(index: Dict[str, Dict], prompt_text: str) -> Optional[Dict]:
    """
    Look up a task by prompt.  Falls back to substring matching on entry_point
    when the tokeniser round-trip introduces minor whitespace differences.
    """
    key = prompt_text.strip()
    if key in index:
        return index[key]

    # Fuzzy fallback: find the task whose prompt is a substring of the
    # decoded prompt (handles leading/trailing special tokens).
    for task_prompt, task in index.items():
        if task_prompt in prompt_text or prompt_text in task_prompt:
            return task

    return None


# --------------------------------------------------------------------------- #
# Public API                                                                    #
# --------------------------------------------------------------------------- #

class HumanEvalRewardFn:
    """
    Callable reward function based on HumanEval unit-test execution.

    Signature matches the GRPO trainer interface:
        reward_fn(prompts_text: List[str], responses_text: List[str]) -> List[float]

    Rewards:
        +1.0  — generated code passes all unit tests
         0.0  — code fails (syntax error, wrong answer, timeout, …)

    Args:
        timeout: seconds allowed per subprocess execution (default 5.0).
        dataset_split: HumanEval split to load (always 'test' for HumanEval).
        partial_credit: if True, award 0.5 for code that at least runs without
                        crashing (i.e. no SyntaxError / ImportError) even if
                        assertions fail.  Defaults to False.
    """

    def __init__(
        self,
        timeout: float = 5.0,
        dataset_split: str = "test",
        partial_credit: bool = False,
    ):
        self.timeout = timeout
        self.partial_credit = partial_credit

        ds = load_dataset("openai/openai_humaneval", split=dataset_split)
        self._index = _build_prompt_index(ds)

    # ---------------------------------------------------------------------- #

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

    # ---------------------------------------------------------------------- #

    def _score_one(self, prompt: str, response: str) -> float:
        task = _find_task(self._index, prompt)
        if task is None:
            # Prompt not found in HumanEval — give 0 to avoid silent bugs
            return 0.0

        # The model generates the function body; prepend the original prompt
        # to reconstruct a complete, runnable module.
        full_code = task["prompt"] + response

        passed = _execute_code(
            code=full_code,
            test=task["test"],
            entry_point=task["entry_point"],
            timeout=self.timeout,
        )
        if passed:
            return 1.0

        if self.partial_credit:
            # Check if the code at least runs without crashing
            runs_ok = _execute_code(
                code=full_code,
                test="def check(candidate): pass",
                entry_point=task["entry_point"],
                timeout=self.timeout,
            )
            return 0.5 if runs_ok else 0.0

        return 0.0

    # ---------------------------------------------------------------------- #

    @property
    def task_count(self) -> int:
        return len(self._index)
