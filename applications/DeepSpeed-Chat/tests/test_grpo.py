# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""
GRPO test suite covering:
  - Unit tests: advantage computation, reward function interface
  - Integration test: OPT-125M + mock reward, 1 full GRPO step (single-process, no distributed)
"""

import os
import sys
import types
import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**overrides):
    """Build a minimal namespace that satisfies DeepSpeedGRPOTrainer.__init__."""
    defaults = dict(
        max_answer_seq_len=32,
        end_of_conversation_token="</s>",
        actor_zero_stage=0,
        compute_fp32_loss=False,
        grpo_num_generations=4,
        kl_ctl=0.1,
        cliprange=0.2,
        print_answers=False,
        print_answers_interval=1,
        local_rank=0,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Unit tests: compute_group_advantages
# ---------------------------------------------------------------------------

class TestComputeGroupAdvantages:
    """Tests for the group-relative advantage normalisation."""

    def _trainer(self):
        """Return a trainer instance without touching distributed or GPU."""
        from dschat.rlhf.grpo_trainer import DeepSpeedGRPOTrainer

        # Minimal stub engine
        class _FakeTokenizer:
            def __call__(self, text):
                return {"input_ids": [0]}

        class _FakeEngine:
            actor = None
            ref = None
            tokenizer = _FakeTokenizer()

        args = _make_args()
        trainer = DeepSpeedGRPOTrainer.__new__(DeepSpeedGRPOTrainer)
        trainer.args = args
        trainer.tokenizer = _FakeEngine.tokenizer
        trainer.max_answer_seq_len = args.max_answer_seq_len
        trainer.end_of_conversation_token_id = 2
        trainer.z3_enabled = False
        trainer.compute_fp32_loss = False
        trainer.num_generations = args.grpo_num_generations
        trainer.kl_ctl = args.kl_ctl
        trainer.cliprange = args.cliprange
        trainer.reward_fn = DeepSpeedGRPOTrainer._default_reward_fn
        trainer.last_generated_experience = None
        trainer.generate_time = 0.0
        trainer.rlhf_engine = None
        trainer.actor_model = None
        trainer.ref_model = None
        return trainer

    def test_output_shape(self):
        trainer = self._trainer()
        B, G = 3, 4
        rewards = torch.arange(float(B * G))
        adv = trainer.compute_group_advantages(rewards, B, G)
        assert adv.shape == (B * G,)

    def test_zero_mean_unit_std_per_group(self):
        trainer = self._trainer()
        B, G = 2, 4
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
        adv = trainer.compute_group_advantages(rewards, B, G)
        adv_grouped = adv.view(B, G)
        for i in range(B):
            assert abs(adv_grouped[i].mean().item()) < 1e-5, \
                f"group {i} mean not ~0: {adv_grouped[i].mean().item()}"
            assert abs(adv_grouped[i].std().item() - 1.0) < 1e-4, \
                f"group {i} std not ~1: {adv_grouped[i].std().item()}"

    def test_identical_rewards_no_nan(self):
        """When all rewards in a group are equal std≈0; should not produce NaN."""
        trainer = self._trainer()
        B, G = 2, 3
        rewards = torch.ones(B * G)
        adv = trainer.compute_group_advantages(rewards, B, G)
        assert not torch.isnan(adv).any(), "NaN in advantages when rewards are identical"
        assert torch.all(adv == 0.0), "Advantages should be 0 when rewards are identical"


# ---------------------------------------------------------------------------
# Unit tests: grpo_loss_fn
# ---------------------------------------------------------------------------

class TestGrpoLossFn:
    """Tests for the GRPO loss function."""

    def _trainer(self):
        from dschat.rlhf.grpo_trainer import DeepSpeedGRPOTrainer
        t = DeepSpeedGRPOTrainer.__new__(DeepSpeedGRPOTrainer)
        t.kl_ctl = 0.1
        t.cliprange = 0.2
        return t

    def test_output_is_scalar(self):
        t = self._trainer()
        T = 10
        lp = torch.zeros(2, T)
        old_lp = torch.zeros(2, T)
        adv = torch.ones(2, T)
        kl = torch.zeros(2, T)
        mask = torch.ones(2, T)
        loss = t.grpo_loss_fn(lp, old_lp, adv, kl, mask)
        assert loss.shape == (), f"Expected scalar, got {loss.shape}"

    def test_loss_decreases_with_positive_advantage(self):
        """When advantage > 0 and we increase log-prob, loss should decrease."""
        t = self._trainer()
        T = 5
        mask = torch.ones(1, T)
        adv = torch.ones(1, T)  # positive
        kl = torch.zeros(1, T)

        # ratio=1 (no policy change)
        lp_same = torch.zeros(1, T)
        old_lp = torch.zeros(1, T)
        loss_same = t.grpo_loss_fn(lp_same, old_lp, adv, kl, mask)

        # ratio>1 (policy moved in right direction, within clip)
        lp_better = torch.full((1, T), 0.1)
        loss_better = t.grpo_loss_fn(lp_better, old_lp, adv, kl, mask)

        assert loss_better < loss_same, \
            "Loss should be lower when policy improves on positive-advantage tokens"

    def test_clip_limits_ratio(self):
        """Ratio far outside [1-clip, 1+clip] should be clipped."""
        t = self._trainer()
        T = 4
        mask = torch.ones(1, T)
        adv = torch.ones(1, T)
        kl = torch.zeros(1, T)
        old_lp = torch.zeros(1, T)

        # Huge positive log-ratio (ratio >> 1+clip)
        lp_extreme = torch.full((1, T), 5.0)
        # Ratio just at clip boundary
        lp_clip = torch.full((1, T), torch.log(torch.tensor(1.0 + t.cliprange)))

        loss_extreme = t.grpo_loss_fn(lp_extreme, old_lp, adv, kl, mask)
        loss_clip = t.grpo_loss_fn(lp_clip, old_lp, adv, kl, mask)

        assert abs(loss_extreme.item() - loss_clip.item()) < 1e-4, \
            "Extreme ratio should be clipped to the same loss as the boundary ratio"


# ---------------------------------------------------------------------------
# Unit tests: default_reward_fn
# ---------------------------------------------------------------------------

class TestDefaultRewardFn:
    def test_returns_list_of_ones(self):
        from dschat.rlhf.grpo_trainer import DeepSpeedGRPOTrainer
        prompts = ["a", "b", "c"]
        responses = ["x", "y", "z"]
        result = DeepSpeedGRPOTrainer._default_reward_fn(prompts, responses)
        assert result == [1.0, 1.0, 1.0]

    def test_length_matches_input(self):
        from dschat.rlhf.grpo_trainer import DeepSpeedGRPOTrainer
        n = 7
        result = DeepSpeedGRPOTrainer._default_reward_fn(["p"] * n, ["r"] * n)
        assert len(result) == n


# ---------------------------------------------------------------------------
# Unit tests: gather_log_probs
# ---------------------------------------------------------------------------

class TestGatherLogProbs:
    def test_shape(self):
        from dschat.rlhf.grpo_trainer import gather_log_probs
        B, T, V = 2, 5, 100
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        out = gather_log_probs(logits, labels)
        assert out.shape == (B, T)

    def test_values_are_log_probs(self):
        from dschat.rlhf.grpo_trainer import gather_log_probs
        B, T, V = 1, 3, 10
        logits = torch.zeros(B, T, V)
        labels = torch.zeros(B, T, dtype=torch.long)
        out = gather_log_probs(logits, labels)
        expected = torch.log(torch.tensor(1.0 / V))
        assert torch.allclose(out, torch.full_like(out, expected), atol=1e-5)


# ---------------------------------------------------------------------------
# Integration test: OPT-125M, 1 full GRPO step (CPU / single-process)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestGRPOIntegration:
    """
    Runs one full generate_experience → train_grpo cycle on OPT-125M.
    Requires: GPU or at least enough CPU RAM.  Skipped automatically when
    facebook/opt-125m is not available in the HF cache.
    """

    MODEL_ID = "Qwen/Qwen2.5-0.5B"

    @pytest.fixture(autouse=True)
    def skip_if_no_model(self):
        # Quick check: can we load the config without downloading?
        try:
            from transformers import AutoConfig
            AutoConfig.from_pretrained(self.MODEL_ID, local_files_only=True)
        except Exception:
            pytest.skip(f"{self.MODEL_ID} not in local HF cache; skipping integration test")

    @pytest.fixture(autouse=True)
    def init_distributed(self):
        """Bootstrap a single-process fake distributed group."""
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29600")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="gloo", rank=0, world_size=1
            )
        yield

    def _build_engine_and_trainer(self):
        import deepspeed
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from dschat.rlhf.grpo_trainer import DeepSpeedGRPOTrainer

        tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID, local_files_only=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        # Minimal DeepSpeed config (ZeRO-0, fp32, CPU)
        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "train_batch_size": 1,
            "fp16": {"enabled": False},
            "bf16": {"enabled": False},
            "zero_optimization": {"stage": 0},
            "gradient_clipping": 1.0,
        }

        def _make_engine(lr=1e-5):
            model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID, local_files_only=True
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            engine, *_ = deepspeed.initialize(
                model=model, optimizer=optimizer, config=ds_config
            )
            return engine

        actor = _make_engine()
        ref = _make_engine()

        args = _make_args(actor_zero_stage=0)

        class _FakeRLHFEngine:
            pass

        rlhf_engine = _FakeRLHFEngine()
        rlhf_engine.actor = actor
        rlhf_engine.ref = ref
        rlhf_engine.tokenizer = tokenizer

        mock_reward_fn = lambda prompts, responses: [0.5] * len(prompts)
        trainer = DeepSpeedGRPOTrainer(rlhf_engine, args, reward_fn=mock_reward_fn)
        return trainer, tokenizer

    def test_one_grpo_step_runs_and_loss_is_finite(self):
        trainer, tokenizer = self._build_engine_and_trainer()

        device = next(trainer.actor_model.parameters()).device

        # Build a tiny batch: 1 prompt, generate G=4 responses
        text = "Hello, my name is"
        enc = tokenizer(text, return_tensors="pt", padding=True)
        prompts = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        # generate_experience
        experience = trainer.generate_experience(prompts, mask, step=0)
        assert experience is not None
        assert "logprobs" in experience
        assert "advantages" in experience
        assert not torch.isnan(experience["logprobs"]).any(), "NaN in logprobs"
        assert not torch.isnan(experience["advantages"]).any(), "NaN in advantages"

        # train_grpo
        loss = trainer.train_grpo(experience)
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        print(f"\n[integration] GRPO loss = {loss.item():.6f}")
