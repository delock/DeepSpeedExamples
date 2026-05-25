# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

"""
CodeAlpaca-20k dataset for SFT training (Step 1).
"""

from datasets import load_dataset
from dschat.utils.data.raw_datasets import PromptRawDataset


CODEALAPCA_PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

CODEALAPCA_PROMPT_TEMPLATE_NO_INPUT = """### Instruction:
{instruction}

### Response:
{output}"""


def format_codealpaca(sample):
    if sample["input"].strip():
        return CODEALAPCA_PROMPT_TEMPLATE.format(**sample)
    else:
        return CODEALAPCA_PROMPT_TEMPLATE_NO_INPUT.format(
            instruction=sample["instruction"], output=sample["output"]
        )


def format_codealpaca_prompt_only(sample):
    """For phase 3 (GRPO) — just the instruction as prompt."""
    if sample["input"].strip():
        return f"### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}\n\n### Response:\n"
    else:
        return f"### Instruction:\n{sample['instruction']}\n\n### Response:\n"


class CodeAlpacaDataset(PromptRawDataset):

    dataset_name = "sahil2801/CodeAlpaca-20k"
    dataset_name_clean = "CodeAlpaca_20k"

    def __init__(self, output_path, seed, local_rank, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        raw = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        # Split 90/10 for train/eval
        split = raw.train_test_split(test_size=0.1, seed=seed)
        self.raw_datasets = {
            "train": split["train"],
            "test": split["test"],
        }

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return format_codealpaca_prompt_only(sample)

    def get_chosen(self, sample):
        return " " + sample["output"]

    def get_rejected(self, sample):
        return None

    def get_prompt_and_chosen(self, sample):
        return format_codealpaca(sample)

    def get_prompt_and_rejected(self, sample):
        return None
