# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""
Tests for HumanEval reward function and dataset.
"""

import pytest
from dschat.utils.reward.humaneval_reward import (
    _execute_code,
    _build_prompt_index,
    _find_task,
    HumanEvalRewardFn,
)


# --------------------------------------------------------------------------- #
# _execute_code                                                                 #
# --------------------------------------------------------------------------- #

class TestExecuteCode:

    def test_correct_solution_passes(self):
        code = "def add(a, b):\n    return a + b"
        test = "def check(candidate):\n    assert candidate(1, 2) == 3"
        assert _execute_code(code, test, "add", timeout=5.0) is True

    def test_wrong_solution_fails(self):
        code = "def add(a, b):\n    return a - b"
        test = "def check(candidate):\n    assert candidate(1, 2) == 3"
        assert _execute_code(code, test, "add", timeout=5.0) is False

    def test_syntax_error_fails(self):
        code = "def add(a b):\n    return a + b"  # missing comma
        test = "def check(candidate):\n    pass"
        assert _execute_code(code, test, "add", timeout=5.0) is False

    def test_infinite_loop_times_out(self):
        code = "def spin():\n    while True: pass"
        test = "def check(candidate):\n    candidate()"
        assert _execute_code(code, test, "spin", timeout=1.0) is False

    def test_runtime_exception_fails(self):
        code = "def bad(x):\n    return 1 / 0"
        test = "def check(candidate):\n    candidate(1)"
        assert _execute_code(code, test, "bad", timeout=5.0) is False


# --------------------------------------------------------------------------- #
# _build_prompt_index / _find_task                                              #
# --------------------------------------------------------------------------- #

class TestPromptIndex:

    @pytest.fixture(scope="class")
    def index(self):
        from datasets import load_dataset
        ds = load_dataset("openai/openai_humaneval", split="test")
        return _build_prompt_index(ds)

    def test_index_has_164_entries(self, index):
        assert len(index) == 164

    def test_exact_lookup(self, index):
        # Take any key and find it back
        first_key = next(iter(index))
        task = _find_task(index, first_key)
        assert task is not None
        assert "test" in task
        assert "entry_point" in task

    def test_fuzzy_lookup_with_extra_whitespace(self, index):
        first_key = next(iter(index))
        # Add leading/trailing whitespace to simulate tokeniser round-trip
        task = _find_task(index, "  " + first_key + "\n")
        assert task is not None

    def test_unknown_prompt_returns_none(self, index):
        task = _find_task(index, "this is definitely not a humaneval prompt")
        assert task is None


# --------------------------------------------------------------------------- #
# HumanEvalRewardFn                                                             #
# --------------------------------------------------------------------------- #

class TestHumanEvalRewardFn:

    @pytest.fixture(scope="class")
    def reward_fn(self):
        return HumanEvalRewardFn(timeout=10.0)

    @pytest.fixture(scope="class")
    def first_task(self, reward_fn):
        """Return the first HumanEval task for use in reward tests."""
        from datasets import load_dataset
        ds = load_dataset("openai/openai_humaneval", split="test")
        return ds[0]   # has_close_elements

    def test_correct_solution_gets_1(self, reward_fn, first_task):
        prompt = first_task["prompt"]
        correct_body = first_task["canonical_solution"]
        scores = reward_fn([prompt], [correct_body])
        assert scores == [1.0], f"Expected 1.0 for correct solution, got {scores}"

    def test_wrong_solution_gets_0(self, reward_fn, first_task):
        prompt = first_task["prompt"]
        wrong_body = "    return False  # always wrong\n"
        scores = reward_fn([prompt], [wrong_body])
        assert scores == [0.0], f"Expected 0.0 for wrong solution, got {scores}"

    def test_batch_length_preserved(self, reward_fn, first_task):
        prompts = [first_task["prompt"]] * 4
        responses = [first_task["canonical_solution"]] * 4
        scores = reward_fn(prompts, responses)
        assert len(scores) == 4

    def test_unknown_prompt_gets_0(self, reward_fn):
        scores = reward_fn(["not a humaneval prompt"], ["    pass"])
        assert scores == [0.0]

    def test_task_count(self, reward_fn):
        assert reward_fn.task_count == 164

    def test_partial_credit_for_runnable_code(self, first_task):
        rf = HumanEvalRewardFn(timeout=10.0, partial_credit=True)
        prompt = first_task["prompt"]
        # Code that runs but gives wrong answers
        wrong_but_runnable = "    return False\n"
        scores = rf([prompt], [wrong_but_runnable])
        # Should get 0.5 (runs ok) not 1.0 (passes tests) not 0.0 (crashes)
        assert scores == [0.5], f"Expected partial credit 0.5, got {scores}"


# --------------------------------------------------------------------------- #
# HumanEvalRawDataset (data pipeline integration)                               #
# --------------------------------------------------------------------------- #

class TestHumanEvalRawDataset:

    @pytest.fixture(scope="class")
    def ds(self):
        from dschat.utils.data.humaneval_dataset import HumanEvalRawDataset
        return HumanEvalRawDataset(
            output_path="/tmp", seed=42, local_rank=0,
            dataset_name="openai/openai_humaneval"
        )

    def test_train_data_length(self, ds):
        assert len(ds.get_train_data()) == 164

    def test_eval_data_length(self, ds):
        assert len(ds.get_eval_data()) == 164

    def test_get_prompt_returns_string(self, ds):
        sample = ds.get_train_data()[0]
        prompt = ds.get_prompt(sample)
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_get_prompt_contains_def(self, ds):
        sample = ds.get_train_data()[0]
        prompt = ds.get_prompt(sample)
        assert "def " in prompt, "Prompt should contain a function definition"

    def test_get_raw_dataset_dispatch(self):
        """data_utils.get_raw_dataset should route humaneval correctly."""
        import types, os, tempfile
        from dschat.utils.data.data_utils import get_raw_dataset
        from dschat.utils.data.humaneval_dataset import HumanEvalRawDataset
        with tempfile.TemporaryDirectory() as tmp:
            ds = get_raw_dataset("openai/openai_humaneval", tmp, 0, 0)
        assert isinstance(ds, HumanEvalRawDataset)
