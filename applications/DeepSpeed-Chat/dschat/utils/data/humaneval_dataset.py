# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""
HumanEval prompt dataset for GRPO training.

Integrates with the existing dschat data pipeline:
  get_raw_dataset("openai/openai_humaneval", ...) → HumanEvalRawDataset

The dataset returns the raw HumanEval prompt (function signature + docstring)
as the GRPO prompt; the model is expected to generate the function body.

Compatible with create_prompt_dataset (train_phase=3).
"""

from datasets import load_dataset
from torch.utils.data import Dataset


class HumanEvalRawDataset:
    """
    Wraps the openai/openai_humaneval HuggingFace dataset so it fits into the
    dschat raw-dataset interface used by create_prompt_dataset / create_dataset.
    """

    dataset_name = "openai/openai_humaneval"
    dataset_name_clean = "openai_humaneval"

    def __init__(self, output_path, seed, local_rank, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        # HumanEval has only a "test" split (164 problems).
        # We use it for both train and eval to keep the existing API happy;
        # callers typically use data_split to carve out sub-sets.
        raw = load_dataset("openai/openai_humaneval", split="test")
        self.raw_datasets = {"train": raw, "test": raw}

    # ---------------------------------------------------------------------- #
    # dschat raw-dataset interface                                             #
    # ---------------------------------------------------------------------- #

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample) -> str:
        """
        Return the HumanEval function signature + docstring as the GRPO prompt.

        No chat template is applied — the model should directly continue the
        Python source.  Add a chat-template wrapper here if you use an
        instruction-tuned model.
        """
        return sample["prompt"]

    # The methods below are required by the interface but not used for phase-3
    # (GRPO) training.
    def get_chosen(self, sample):
        return sample["canonical_solution"]

    def get_rejected(self, sample):
        return None

    def get_prompt_and_chosen(self, sample):
        return sample["prompt"] + sample["canonical_solution"]

    def get_prompt_and_rejected(self, sample):
        return None
