# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch.nn.functional as F
import time
import deepspeed
from concurrent.futures import ThreadPoolExecutor
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from deepspeed.accelerator import get_accelerator

from dschat.utils.utils import print_rank_0


def print_all_ranks(tag, value, rank):
    world_size = torch.distributed.get_world_size()
    all_tensor = torch.zeros(world_size, dtype=torch.float32).to(
        get_accelerator().current_device_name())
    all_tensor[rank] = value
    torch.distributed.all_reduce(all_tensor, op=torch.distributed.ReduceOp.SUM)
    print_rank_0(f'{tag} {all_tensor}', rank)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                total += float(param.float().norm())

    return total


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class DeepSpeedGRPOTrainer():

    def __init__(self, rlhf_engine, args, reward_fn=None):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        self.ref_model = self.rlhf_engine.ref
        self.tokenizer = self.rlhf_engine.tokenizer
        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        eot_token = args.end_of_conversation_token
        if eot_token:
            token_ids = self.tokenizer(eot_token)['input_ids']
            self.end_of_conversation_token_id = token_ids[-1] if token_ids else self.tokenizer.eos_token_id
        else:
            self.end_of_conversation_token_id = self.tokenizer.eos_token_id
        self.z3_enabled = args.actor_zero_stage == 3
        self.compute_fp32_loss = self.args.compute_fp32_loss

        self.num_generations = getattr(args, 'grpo_num_generations', 4)
        self.kl_ctl = getattr(args, 'kl_ctl', 0.1)
        self.cliprange = getattr(args, 'cliprange', 0.2)

        if reward_fn is not None:
            self.reward_fn = reward_fn
        else:
            self.reward_fn = self._default_reward_fn

        self.last_generated_experience = None
        self.generate_time = 0.0

    @staticmethod
    def _default_reward_fn(prompts_text, responses_text):
        return [1.0] * len(prompts_text)

    def _generate_sequence(self, prompts, mask, step):
        max_min_length = self.max_answer_seq_len + prompts.shape[1]

        # Always sample with temperature > 0 so that G generations per prompt
        # are diverse; greedy decoding (do_sample=False) makes all G outputs
        # identical and collapses the group advantage to 0.
        kwargs = dict(do_sample=True, temperature=0.9, top_p=0.95)

        with torch.no_grad():
            seq = self.actor_model.module.generate(
                prompts,
                attention_mask=mask,
                max_new_tokens=self.max_answer_seq_len,
                pad_token_id=self.tokenizer.pad_token_id,
                synced_gpus=self.z3_enabled,
                **kwargs)

        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length
        ans = seq[:, prompt_length:]
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)

        # Track sequence length stats for profiling
        self._last_seq_lengths = valid_ans_len.cpu().tolist()
        self._last_prompt_length = prompt_length
        self._last_total_seq_length = seq.shape[1]

        if self.args.print_answers and (step % self.args.print_answers_interval
                                        == 0):
            print(
                f"--- prompt --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(prompts, skip_special_tokens=True)}"
            )
            print(
                f"--- ans    --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(ans, skip_special_tokens=True)}"
            )

        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[i] <= 1:
                print(
                    f'Dropping too short generated answer: {step=}: \n'
                    f'prompts: {self.tokenizer.batch_decode(prompts, skip_special_tokens=False)}\n'
                    f'answers: {self.tokenizer.batch_decode(ans, skip_special_tokens=False)}'
                )
                continue
            else:
                out_seq.append(seq[i:i + 1])

        if not out_seq:
            print(
                f'All generated results are too short for rank={self.args.local_rank} step={step}\n'
                f'-> prompts: {self.tokenizer.batch_decode(prompts, skip_special_tokens=False)}\n'
                f'-> answers: {self.tokenizer.batch_decode(ans, skip_special_tokens=False)}'
            )
            return None

        out_seq = torch.cat(out_seq, dim=0)
        return out_seq

    def _generate_with_repetition(self, prompts, mask, step):
        G = self.num_generations

        if getattr(self.args, 'early_exit_generate', False):
            seq = self._generate_early_exit(prompts, mask, G, step)
        elif getattr(self.args, 'shared_prefix_generate', False):
            seq = self._generate_shared_prefix(prompts, mask, G, step)
        else:
            repeated_prompts = prompts.repeat_interleave(G, dim=0)
            repeated_mask = mask.repeat_interleave(G, dim=0)
            seq = self._generate_sequence(repeated_prompts, repeated_mask, step)

        if seq is None:
            return None

        repeated_prompts_out = prompts.repeat_interleave(G, dim=0)
        return {
            'prompts': prompts,
            'repeated_prompts': repeated_prompts_out,
            'seq': seq,
            'batch_size': prompts.shape[0],
            'num_generations': G,
        }

    def _generate_shared_prefix(self, prompts, mask, G, step):
        """
        Prefill each unique prompt once, copy KV cache G times,
        then decode G responses sharing the same prefix cache.
        Saves (G-1)/G of prefill compute.
        """
        kwargs = dict(do_sample=True, temperature=0.9, top_p=0.95)

        with torch.no_grad():
            # Step 1: Prefill unique prompts (batch_size forward passes worth)
            prefill_out = self.actor_model.module(
                prompts, attention_mask=mask, use_cache=True)
            past = prefill_out.past_key_values

            # Step 2: Expand KV cache G times per prompt
            expanded_past = past.batch_repeat_interleave(G)

            # Step 3: Expand input_ids and attention_mask
            expanded_ids = prompts.repeat_interleave(G, dim=0)
            expanded_mask = mask.repeat_interleave(G, dim=0)

            # Step 4: Generate with pre-computed cache
            seq = self.actor_model.module.generate(
                expanded_ids,
                attention_mask=expanded_mask,
                max_new_tokens=self.max_answer_seq_len,
                pad_token_id=self.tokenizer.pad_token_id,
                synced_gpus=self.z3_enabled,
                past_key_values=expanded_past,
                **kwargs)

        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length
        ans = seq[:, prompt_length:]
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)

        # Track sequence length stats for profiling
        self._last_seq_lengths = valid_ans_len.cpu().tolist()
        self._last_prompt_length = prompt_length
        self._last_total_seq_length = seq.shape[1]

        if self.args.print_answers and (step % self.args.print_answers_interval == 0):
            print(
                f"--- prompt --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(prompts, skip_special_tokens=True)}"
            )
            print(
                f"--- ans    --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(ans, skip_special_tokens=True)}"
            )

        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[i] <= 1:
                print(
                    f'Dropping too short generated answer: {step=}: \n'
                    f'prompts: {self.tokenizer.batch_decode(prompts, skip_special_tokens=False)}\n'
                    f'answers: {self.tokenizer.batch_decode(ans, skip_special_tokens=False)}'
                )
                continue
            else:
                out_seq.append(seq[i:i + 1])

        if not out_seq:
            print(
                f'All generated results are too short for rank={self.args.local_rank} step={step}\n'
                f'-> prompts: {self.tokenizer.batch_decode(prompts, skip_special_tokens=False)}\n'
                f'-> answers: {self.tokenizer.batch_decode(ans, skip_special_tokens=False)}'
            )
            return None

        out_seq = torch.cat(out_seq, dim=0)
        return out_seq

    def _sample_top_p(self, logits, temperature=0.9, top_p=0.95):
        """Sample from logits with temperature and nucleus (top-p) filtering."""
        logits = logits / temperature
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1), dim=-1)
        mask = (cumulative_probs - torch.softmax(sorted_logits, dim=-1)) >= top_p
        sorted_logits[mask] = -float('inf')
        probs = torch.softmax(sorted_logits, dim=-1)
        sampled = torch.multinomial(probs, 1)
        tokens = sorted_indices.gather(1, sampled)
        return tokens

    def _generate_early_exit(self, prompts, mask, G, step):
        """
        Custom decode loop with batch compaction: prefill once, expand KV cache,
        then decode token-by-token removing finished sequences from the batch.
        Combines shared-prefix (saves prefill) + early-exit (saves decode padding).
        """
        device = prompts.device
        B = prompts.shape[0]
        total = B * G
        prompt_len = prompts.shape[1]
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        with torch.no_grad():
            # Prefill unique prompts once
            out = self.actor_model.module(
                prompts, attention_mask=mask, use_cache=True)
            past = out.past_key_values

            # Expand KV cache G times
            past.batch_repeat_interleave(G)

            # Expand attention mask for decode (will grow each step)
            # Shape: [total, prompt_len] -> extended each decode step
            attn_mask = mask.repeat_interleave(G, dim=0)  # [total, prompt_len]

            # Sample first token from prefill logits
            first_logits = out.logits[:, -1, :].repeat_interleave(G, dim=0)
            next_tokens = self._sample_top_p(first_logits)  # [total, 1]

            # Storage for all generated tokens
            all_tokens = torch.full(
                (total, self.max_answer_seq_len), pad_token_id,
                dtype=torch.long, device=device)
            all_tokens[:, 0] = next_tokens.squeeze(1)
            gen_lens = torch.ones(total, dtype=torch.long, device=device)

            # Extend attention mask for the first generated token
            attn_mask = torch.cat([attn_mask, torch.ones(total, 1, dtype=attn_mask.dtype, device=device)], dim=1)

            # Track active sequences
            active_idx = torch.arange(total, device=device)
            finished = (next_tokens.squeeze(1) == eos_token_id)
            if finished.any():
                keep = ~finished
                active_idx = active_idx[keep]
                next_tokens = next_tokens[keep]
                past.reorder_cache(torch.where(keep)[0])
                attn_mask = attn_mask[keep]

            # Decode loop with compaction
            for decode_step in range(1, self.max_answer_seq_len):
                if active_idx.shape[0] == 0:
                    break

                out = self.actor_model.module(
                    next_tokens, attention_mask=attn_mask,
                    past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_tokens = self._sample_top_p(out.logits[:, -1, :])

                # Store tokens back to original positions
                all_tokens[active_idx, decode_step] = next_tokens.squeeze(1)
                gen_lens[active_idx] = decode_step + 1

                # Extend attention mask
                attn_mask = torch.cat([attn_mask, torch.ones(attn_mask.shape[0], 1, dtype=attn_mask.dtype, device=device)], dim=1)

                # Remove finished sequences
                finished = (next_tokens.squeeze(1) == eos_token_id)
                if finished.any():
                    keep = ~finished
                    if not keep.any():
                        break
                    active_idx = active_idx[keep]
                    next_tokens = next_tokens[keep]
                    past.reorder_cache(torch.where(keep)[0])
                    attn_mask = attn_mask[keep]

        # Build output: [prompt | generated]
        max_gen = gen_lens.max().item()
        all_tokens = all_tokens[:, :max_gen]
        expanded_prompts = prompts.repeat_interleave(G, dim=0)
        seq = torch.cat([expanded_prompts, all_tokens], dim=1)

        # Post-processing (same as _generate_shared_prefix)
        batch_size = seq.shape[0]
        self.prompt_length = prompt_len
        ans = seq[:, prompt_len:]
        valid_ans_len = (ans != pad_token_id).sum(dim=-1)

        self._last_seq_lengths = valid_ans_len.cpu().tolist()
        self._last_prompt_length = prompt_len
        self._last_total_seq_length = seq.shape[1]

        if self.args.print_answers and (step % self.args.print_answers_interval == 0):
            print(
                f"--- prompt --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(prompts, skip_special_tokens=True)}"
            )
            print(
                f"--- ans    --> step={step}, rank={torch.distributed.get_rank()}, {self.tokenizer.batch_decode(ans, skip_special_tokens=True)}"
            )

        out_seq = []
        for i in range(batch_size):
            if valid_ans_len[i] <= 1:
                print(
                    f'Dropping too short generated answer: {step=}: \n'
                    f'prompts: {self.tokenizer.batch_decode(prompts, skip_special_tokens=False)}\n'
                    f'answers: {self.tokenizer.batch_decode(ans, skip_special_tokens=False)}'
                )
                continue
            else:
                out_seq.append(seq[i:i + 1])

        if not out_seq:
            print(
                f'All generated results are too short for rank={self.args.local_rank} step={step}\n'
                f'-> prompts: {self.tokenizer.batch_decode(prompts, skip_special_tokens=False)}\n'
                f'-> answers: {self.tokenizer.batch_decode(ans, skip_special_tokens=False)}'
            )
            return None

        out_seq = torch.cat(out_seq, dim=0)
        return out_seq

    def compute_group_advantages(self, reward_scores, batch_size, G):
        rewards_grouped = reward_scores.view(batch_size, G)
        mean = rewards_grouped.mean(dim=1, keepdim=True)
        std = rewards_grouped.std(dim=1, keepdim=True)
        std = torch.clamp(std, min=1e-8)
        advantages_grouped = (rewards_grouped - mean) / std
        return advantages_grouped.view(-1)

    def generate_experience(self, prompts, mask, step):
        self.eval()
        generate_start = time.time()
        gen_result = self._generate_with_repetition(prompts, mask, step)
        generate_end = time.time()

        if gen_result is None:
            assert self.last_generated_experience is not None, \
                f'Invalid generated experience at {step=}'
            self.train()
            return self.last_generated_experience

        seq = gen_result['seq']
        batch_size = gen_result['batch_size']
        G = gen_result['num_generations']
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length

        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()

        logprob_start = time.time()
        with torch.no_grad():
            output = self.actor_model(seq, attention_mask=attention_mask)
            output_ref = self.ref_model(seq, attention_mask=attention_mask)
        logprob_end = time.time()

        logits = output.logits
        logits_ref = output_ref.logits
        if self.compute_fp32_loss:
            logits = logits.to(torch.float)
            logits_ref = logits_ref.to(torch.float)

        log_probs = gather_log_probs(logits[:, :-1, :], seq[:, 1:])
        ref_log_probs = gather_log_probs(logits_ref[:, :-1, :], seq[:, 1:])

        ans = seq[:, prompt_length:]
        actual_batch = seq.shape[0]
        prompts_text = self.tokenizer.batch_decode(
            gen_result['repeated_prompts'][:actual_batch],
            skip_special_tokens=True)
        responses_text = self.tokenizer.batch_decode(ans,
                                                     skip_special_tokens=True)
        reward_start = time.time()
        reward_scores = self.reward_fn(prompts_text, responses_text)
        reward_end = time.time()
        if not isinstance(reward_scores, torch.Tensor):
            reward_scores = torch.tensor(reward_scores,
                                         dtype=torch.float32,
                                         device=seq.device)

        if actual_batch == batch_size * G:
            advantages = self.compute_group_advantages(
                reward_scores, batch_size, G)
        else:
            advantages = reward_scores

        self.generate_time = generate_end - generate_start
        self.logprob_time = logprob_end - logprob_start
        self.reward_time = reward_end - reward_start

        experience = {
            'prompts': gen_result['repeated_prompts'][:actual_batch],
            'logprobs': log_probs,
            'ref_logprobs': ref_log_probs,
            'advantages': advantages,
            'reward_scores': reward_scores,
            'input_ids': seq,
            'attention_mask': attention_mask,
            'prompt_length': prompt_length,
            'batch_size': batch_size,
            'num_generations': G,
        }
        self.last_generated_experience = experience
        self.train()

        return experience

    def generate_experience_async(self, prompts, mask, step):
        """
        Like generate_experience but returns immediately with reward still
        computing in background. Call finalize_experience() to get the
        completed experience (blocks until reward is done).
        """
        self.eval()
        generate_start = time.time()
        gen_result = self._generate_with_repetition(prompts, mask, step)
        generate_end = time.time()

        if gen_result is None:
            assert self.last_generated_experience is not None, \
                f'Invalid generated experience at {step=}'
            self.train()
            return self.last_generated_experience, None  # already finalized

        seq = gen_result['seq']
        batch_size = gen_result['batch_size']
        G = gen_result['num_generations']
        prompt_length = prompts.shape[1]
        self.prompt_length = prompt_length

        pad_token_id = self.tokenizer.pad_token_id
        attention_mask = seq.not_equal(pad_token_id).long()

        logprob_start = time.time()
        with torch.no_grad():
            output = self.actor_model(seq, attention_mask=attention_mask)
            output_ref = self.ref_model(seq, attention_mask=attention_mask)
        logprob_end = time.time()

        logits = output.logits
        logits_ref = output_ref.logits
        if self.compute_fp32_loss:
            logits = logits.to(torch.float)
            logits_ref = logits_ref.to(torch.float)

        log_probs = gather_log_probs(logits[:, :-1, :], seq[:, 1:])
        ref_log_probs = gather_log_probs(logits_ref[:, :-1, :], seq[:, 1:])

        ans = seq[:, prompt_length:]
        actual_batch = seq.shape[0]
        prompts_text = self.tokenizer.batch_decode(
            gen_result['repeated_prompts'][:actual_batch],
            skip_special_tokens=True)
        responses_text = self.tokenizer.batch_decode(ans,
                                                     skip_special_tokens=True)

        # Submit reward to background thread (reward itself uses ProcessPool)
        reward_start = time.time()
        if not hasattr(self, '_reward_executor'):
            self._reward_executor = ThreadPoolExecutor(max_workers=1)
        reward_future = self._reward_executor.submit(
            self.reward_fn, prompts_text, responses_text)

        self.generate_time = generate_end - generate_start
        self.logprob_time = logprob_end - logprob_start
        self._reward_start = reward_start

        pending_experience = {
            'prompts': gen_result['repeated_prompts'][:actual_batch],
            'logprobs': log_probs,
            'ref_logprobs': ref_log_probs,
            'input_ids': seq,
            'attention_mask': attention_mask,
            'prompt_length': prompt_length,
            'batch_size': batch_size,
            'num_generations': G,
            '_reward_future': reward_future,
        }
        self.train()
        return pending_experience, reward_future

    def finalize_experience(self, pending_experience):
        """
        Block until reward is computed, then fill in advantages/reward_scores.
        """
        reward_future = pending_experience.pop('_reward_future')
        reward_scores = reward_future.result()
        reward_end = time.time()
        self.reward_time = reward_end - self._reward_start

        seq = pending_experience['input_ids']
        batch_size = pending_experience['batch_size']
        G = pending_experience['num_generations']

        if not isinstance(reward_scores, torch.Tensor):
            reward_scores = torch.tensor(reward_scores,
                                         dtype=torch.float32,
                                         device=seq.device)

        actual_batch = seq.shape[0]
        if actual_batch == batch_size * G:
            advantages = self.compute_group_advantages(
                reward_scores, batch_size, G)
        else:
            advantages = reward_scores

        pending_experience['advantages'] = advantages
        pending_experience['reward_scores'] = reward_scores
        self.last_generated_experience = pending_experience
        return pending_experience

    def train_grpo(self, inputs):
        log_probs = inputs['logprobs']
        ref_log_probs = inputs['ref_logprobs']
        advantages = inputs['advantages']
        attention_mask = inputs['attention_mask']
        seq = inputs['input_ids']
        prompt_length = inputs['prompt_length']

        start = prompt_length - 1
        action_mask = attention_mask[:, 1:]

        token_advantages = advantages.unsqueeze(1).expand_as(log_probs)

        kl_per_token = log_probs - ref_log_probs

        batch = {'input_ids': seq, 'attention_mask': attention_mask}
        actor_prob = self.actor_model(**batch, use_cache=False).logits
        if self.compute_fp32_loss:
            actor_prob = actor_prob.to(torch.float)
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], seq[:, 1:])

        actor_loss = self.grpo_loss_fn(
            actor_log_prob[:, start:],
            log_probs[:, start:],
            token_advantages[:, start:],
            kl_per_token[:, start:],
            action_mask[:, start:],
        )
        self.actor_model.backward(actor_loss)
        self.actor_model.step()

        return actor_loss

    def grpo_loss_fn(self, logprobs, old_logprobs, advantages, kl_div, mask):
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()

        kl_loss = torch.sum(kl_div * mask) / mask.sum()

        return pg_loss + self.kl_ctl * kl_loss

    def _validate_training_mode(self):
        assert self.actor_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.ref_model.module.training

    def train(self):
        self.actor_model.train()

    def eval(self):
        self.actor_model.eval()
        self.ref_model.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        print_all_ranks(f'{tag} global_actor_model_norm', actor_model_norm,
                        self.args.local_rank)
        print_all_ranks(f'{tag} global_ref_model_norm', ref_model_norm,
                        self.args.local_rank)


class DeepSpeedGRPOTrainerUnsupervised(DeepSpeedGRPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
