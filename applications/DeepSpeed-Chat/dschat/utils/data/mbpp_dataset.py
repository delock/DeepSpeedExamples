# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""
MBPP prompt dataset for GRPO training.

Integrates with the existing dschat data pipeline:
  get_raw_dataset("google-research-datasets/mbpp", ...) → MBPPRawDataset
"""

from datasets import load_dataset
from dschat.utils.reward.mbpp_reward import format_mbpp_prompt


class MBPPRawDataset:
    """
    Wraps the MBPP HuggingFace dataset for the dschat raw-dataset interface.
    Uses the 'train' split (374 problems) for training.
    """

    dataset_name = "google-research-datasets/mbpp"
    dataset_name_clean = "mbpp"

    def __init__(self, output_path, seed, local_rank, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        ds = load_dataset("google-research-datasets/mbpp", "full")
        self.raw_datasets = {
            "train": ds["train"],   # 374 problems
            "test": ds["test"],     # 500 problems (for eval)
        }

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample) -> str:
        """Return the MBPP task as a code-generation prompt."""
        return format_mbpp_prompt(sample["text"])

    def get_chosen(self, sample):
        return sample["code"]

    def get_rejected(self, sample):
        return None

    def get_prompt_and_chosen(self, sample):
        return format_mbpp_prompt(sample["text"]) + sample["code"]

    def get_prompt_and_rejected(self, sample):
        return None
