"""
Mixed reward function that routes to HumanEval or MBPP reward
based on prompt content detection.
"""

from dschat.utils.reward.humaneval_reward import HumanEvalRewardFn
from dschat.utils.reward.mbpp_reward import MBPPRewardFn


class MixedRewardFn:
    """Routes prompts to HumanEval or MBPP reward based on prompt format."""

    def __init__(self, timeout: float = 10.0):
        self.humaneval_fn = HumanEvalRewardFn(timeout=timeout)
        self.mbpp_fn = MBPPRewardFn(timeout=timeout, split="train")
        self.task_count = self.humaneval_fn.task_count + self.mbpp_fn.task_count

    def _is_humaneval_prompt(self, prompt: str) -> bool:
        """HumanEval prompts contain function signatures with docstrings starting with >>>"""
        # HumanEval prompts typically have the format "from typing import..." or def fn(...):\n    \"\"\"..."
        # MBPP prompts are natural language task descriptions
        # Simple heuristic: if prompt contains ">>>" (doctest) it's HumanEval
        if ">>>" in prompt:
            return True
        # HumanEval prompts start with def/from/import at top level
        stripped = prompt.strip()
        if stripped.startswith("def ") or stripped.startswith("from ") or stripped.startswith("import "):
            return True
        return False

    def __call__(self, prompts, responses, **kwargs):
        """
        Route each (prompt, response) pair to the correct reward function.
        Returns list of float rewards.
        """
        rewards = []
        for prompt, response in zip(prompts, responses):
            if self._is_humaneval_prompt(prompt):
                r = self.humaneval_fn([prompt], [response], **kwargs)
            else:
                r = self.mbpp_fn([prompt], [response], **kwargs)
            rewards.append(r[0] if isinstance(r, list) else r)
        return rewards
