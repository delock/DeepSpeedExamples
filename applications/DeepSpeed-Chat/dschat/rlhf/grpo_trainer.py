# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy
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


def compute_log_probs_microbatch(model, seq, attention_mask, compute_fp32_loss=False, micro_batch_size=8):
    """Compute log probs with micro-batching to avoid OOM."""
    all_log_probs = []
    for i in range(0, seq.shape[0], micro_batch_size):
        mb_seq = seq[i:i+micro_batch_size]
        mb_mask = attention_mask[i:i+micro_batch_size]
        mb_logits = model(mb_seq, attention_mask=mb_mask).logits
        if compute_fp32_loss:
            mb_logits = mb_logits.to(torch.float)
        all_log_probs.append(gather_log_probs(mb_logits[:, :-1, :], mb_seq[:, 1:]))
        del mb_logits
    return torch.cat(all_log_probs, dim=0)


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

        if getattr(self.args, 'continuous_batching_generate', False):
            seq = self._generate_continuous_batching(prompts, mask, G, step)
        elif getattr(self.args, 'early_exit_generate', False):
            cb_size = getattr(self.args, 'continuous_batching_size', 0)
            if cb_size and cb_size < G:
                # Split G into micro-batches of cb_size for early_exit
                seq_parts = []
                for g_start in range(0, G, cb_size):
                    g_end = min(g_start + cb_size, G)
                    sub_G = g_end - g_start
                    part = self._generate_early_exit(prompts, mask, sub_G, step)
                    if part is not None:
                        seq_parts.append(part)
                if not seq_parts:
                    seq = None
                else:
                    seq = self._pad_and_cat(seq_parts)
            else:
                seq = self._generate_early_exit(prompts, mask, G, step)
        elif getattr(self.args, 'shared_prefix_generate', False):
            cb_size = getattr(self.args, 'continuous_batching_size', 0)
            if cb_size and cb_size < G:
                seq_parts = []
                for g_start in range(0, G, cb_size):
                    sub_G = min(cb_size, G - g_start)
                    part = self._generate_shared_prefix(prompts, mask, sub_G, step)
                    if part is not None:
                        seq_parts.append(part)
                seq = self._pad_and_cat(seq_parts) if seq_parts else None
            else:
                seq = self._generate_shared_prefix(prompts, mask, G, step)
        else:
            cb_size = getattr(self.args, 'continuous_batching_size', 0)
            if cb_size and cb_size < G:
                seq_parts = []
                for g_start in range(0, G, cb_size):
                    sub_G = min(cb_size, G - g_start)
                    repeated_prompts = prompts.repeat_interleave(sub_G, dim=0)
                    repeated_mask = mask.repeat_interleave(sub_G, dim=0)
                    part = self._generate_sequence(repeated_prompts, repeated_mask, step)
                    if part is not None:
                        seq_parts.append(part)
                seq = self._pad_and_cat(seq_parts) if seq_parts else None
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

    def _cb_replace_slot_inplace(self, slot_idx, past, prompt_past, attn_mask,
                                  first_logits, next_tokens, all_tokens, gen_lens,
                                  slot_rollout, slot_decode_step, slot_position,
                                  slot_pad_start,
                                  new_rollout_idx, prompt_len, B, device):
        """Replace a finished slot with a new rollout by left-padding prompt KV."""
        current_kv_len = past.layers[0].keys.shape[2]
        pad_len = current_kv_len - prompt_len

        src_prompt_idx = 0 if B == 1 else new_rollout_idx // (len(all_tokens) // B)

        # Replace KV cache for this slot: [zeros | prompt_kv]
        for layer_idx in range(len(past)):
            key, value = past[layer_idx]
            src_key, src_value = prompt_past[layer_idx]
            heads = key.shape[1]
            head_dim = key.shape[3]

            pad_kv = torch.zeros(1, heads, pad_len, head_dim,
                                 dtype=key.dtype, device=device)
            new_key = torch.cat([pad_kv, src_key[src_prompt_idx:src_prompt_idx+1]], dim=2)
            new_value = torch.cat([pad_kv, src_value[src_prompt_idx:src_prompt_idx+1]], dim=2)
            key[slot_idx] = new_key[0]
            value[slot_idx] = new_value[0]

        # Replace attention mask: [0...0, 1...1(prompt_len)]
        mask_len = attn_mask.shape[1]
        new_mask = torch.zeros(mask_len, dtype=attn_mask.dtype, device=device)
        new_mask[pad_len:pad_len + prompt_len] = 1
        attn_mask[slot_idx] = new_mask

        # Sample first token for new rollout from prefill logits
        new_first_logits = first_logits[src_prompt_idx:src_prompt_idx+1]
        new_token = self._sample_top_p(new_first_logits)
        next_tokens[slot_idx] = new_token[0]

        # Store
        slot_rollout[slot_idx] = new_rollout_idx
        all_tokens[new_rollout_idx, 0] = new_token[0, 0]
        gen_lens[new_rollout_idx] = 1
        slot_decode_step[slot_idx] = 1
        slot_position[slot_idx] = prompt_len  # next forward will use prompt_len as position
        slot_pad_start[slot_idx] = pad_len

    def _pad_and_cat(self, seq_parts):
        """Pad sequence parts to same length and concatenate."""
        max_len = max(p.shape[1] for p in seq_parts)
        padded = []
        for p in seq_parts:
            if p.shape[1] < max_len:
                pad = torch.full(
                    (p.shape[0], max_len - p.shape[1]),
                    self.tokenizer.pad_token_id,
                    dtype=p.dtype, device=p.device)
                padded.append(torch.cat([p, pad], dim=1))
            else:
                padded.append(p)
        return torch.cat(padded, dim=0)

    def _generate_continuous_batching(self, prompts, mask, G, step):
        """
        True continuous batching: fixed batch_size slots, when a sequence
        finishes (EOS or max_len), immediately replace that slot with the next
        rollout by left-padding prompt KV to match current KV seq_len.

        No pop/reorder_cache — batch size stays constant until all rollouts done.
        """
        device = prompts.device
        B = prompts.shape[0]  # number of unique prompts (typically 1)
        total = B * G
        prompt_len = prompts.shape[1]
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        batch_size = min(total, getattr(self.args, 'continuous_batching_size', 8))
        max_steps = self.max_answer_seq_len

        with torch.no_grad():
            # Prefill unique prompts once
            out = self.actor_model.module(
                prompts, attention_mask=mask, use_cache=True)
            prompt_past = out.past_key_values
            first_logits = out.logits[:, -1, :]  # [B, vocab]

            # Collect all completed sequences
            all_tokens = torch.full(
                (total, max_steps), pad_token_id,
                dtype=torch.long, device=device)
            gen_lens = torch.zeros(total, dtype=torch.long, device=device)
            completed_count = 0
            next_rollout_idx = 0

            # --- Initialize batch_size slots ---
            init_count = min(batch_size, total)

            # Expand prompt KV for initial slots
            past = copy.deepcopy(prompt_past)
            if B == 1:
                past.batch_repeat_interleave(init_count)
                init_logits = first_logits.repeat(init_count, 1)
            else:
                past.batch_repeat_interleave(G)
                if init_count < B * G:
                    past.reorder_cache(torch.arange(init_count, device=device))
                prompt_indices = torch.arange(init_count, device=device) // G
                init_logits = first_logits[prompt_indices]

            # Per-slot state
            slot_rollout = list(range(init_count))  # which rollout idx each slot serves
            slot_decode_step = [0] * init_count
            slot_position = [prompt_len] * init_count  # next position_id
            slot_active = [True] * init_count  # False = slot is idle (no more rollouts)
            slot_pad_start = [0] * init_count  # leading padding length per slot
            next_rollout_idx = init_count

            # Attention mask: [batch_size, prompt_len]
            if B == 1:
                attn_mask = mask.repeat(init_count, 1)
            else:
                attn_mask = mask.repeat_interleave(G, dim=0)[:init_count]

            # Sample first token
            next_tokens = self._sample_top_p(init_logits)  # [init_count, 1]
            for i in range(init_count):
                all_tokens[slot_rollout[i], 0] = next_tokens[i, 0]
                gen_lens[slot_rollout[i]] = 1
                slot_decode_step[i] = 1

            # Extend attn_mask for first token
            attn_mask = torch.cat([attn_mask, torch.ones(init_count, 1, dtype=attn_mask.dtype, device=device)], dim=1)

            # Check immediate EOS
            eos_now = (next_tokens.squeeze(1) == eos_token_id).cpu().tolist()
            for i in range(init_count):
                if eos_now[i]:
                    completed_count += 1
                    if next_rollout_idx < total:
                        self._cb_replace_slot_inplace(
                            i, past, prompt_past, attn_mask,
                            first_logits, next_tokens, all_tokens, gen_lens,
                            slot_rollout, slot_decode_step, slot_position,
                            slot_pad_start,
                            next_rollout_idx, prompt_len, B, device)
                        next_rollout_idx += 1
                    else:
                        slot_active[i] = False

            # Main decode loop
            decode_steps = 0
            total_active_slots = 0
            while completed_count < total:
                # Check if any slot is still active
                if not any(slot_active):
                    break

                num_slots = len(slot_rollout)
                decode_steps += 1
                total_active_slots += num_slots
                # Build position_ids
                pos_ids = torch.tensor(
                    [[slot_position[i]] for i in range(num_slots)],
                    device=device)

                # Forward pass
                out = self.actor_model.module(
                    next_tokens, attention_mask=attn_mask,
                    position_ids=pos_ids,
                    past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_tokens = self._sample_top_p(out.logits[:, -1, :])

                # Extend attention mask
                attn_mask = torch.cat([attn_mask, torch.ones(num_slots, 1, dtype=attn_mask.dtype, device=device)], dim=1)

                # Update positions and store tokens
                for i in range(num_slots):
                    if not slot_active[i]:
                        continue
                    slot_position[i] += 1
                    ds = slot_decode_step[i]
                    if ds < max_steps:
                        all_tokens[slot_rollout[i], ds] = next_tokens[i, 0]
                        gen_lens[slot_rollout[i]] = ds + 1
                    slot_decode_step[i] += 1

                # Check for finished slots
                eos_mask = (next_tokens.squeeze(1) == eos_token_id).cpu().tolist()
                slots_finished = []
                for i in range(num_slots):
                    if not slot_active[i]:
                        continue
                    if eos_mask[i] or slot_decode_step[i] >= max_steps:
                        completed_count += 1
                        # Replace slot with next rollout (or remove from batch)
                        if next_rollout_idx < total:
                            self._cb_replace_slot_inplace(
                                i, past, prompt_past, attn_mask,
                                first_logits, next_tokens, all_tokens, gen_lens,
                                slot_rollout, slot_decode_step, slot_position,
                                slot_pad_start,
                                next_rollout_idx, prompt_len, B, device)
                            next_rollout_idx += 1
                        else:
                            slots_finished.append(i)

                # Remove finished slots with no replacement (early exit / batch compaction)
                if slots_finished:
                    keep = [i for i in range(num_slots) if i not in slots_finished]
                    if not keep:
                        break
                    keep_t = torch.tensor(keep, device=device)
                    next_tokens = next_tokens[keep_t]
                    past.reorder_cache(keep_t)
                    attn_mask = attn_mask[keep_t]
                    # Compact per-slot state
                    slot_rollout = [slot_rollout[i] for i in keep]
                    slot_decode_step = [slot_decode_step[i] for i in keep]
                    slot_position = [slot_position[i] for i in keep]
                    slot_active = [slot_active[i] for i in keep]
                    slot_pad_start = [slot_pad_start[i] for i in keep]

                # KV cache left-trim: remove common leading padding across all active slots
                active_pads = [slot_pad_start[i] for i in range(len(slot_pad_start)) if slot_active[i]]
                if active_pads:
                    min_pad = min(active_pads)
                    if min_pad >= 16:
                        # Trim KV cache
                        for layer_idx in range(len(past)):
                            past.layers[layer_idx].keys = past.layers[layer_idx].keys[:, :, min_pad:]
                            past.layers[layer_idx].values = past.layers[layer_idx].values[:, :, min_pad:]
                        # Trim attention mask
                        attn_mask = attn_mask[:, min_pad:]
                        # Adjust pad_start
                        for i in range(len(slot_pad_start)):
                            slot_pad_start[i] -= min_pad

        # Build output: [prompt | generated]
        max_gen = gen_lens.max().item()
        all_tokens = all_tokens[:, :max_gen]
        expanded_prompts = prompts.repeat_interleave(G, dim=0)
        seq = torch.cat([expanded_prompts, all_tokens], dim=1)

        # Post-processing
        batch_size = seq.shape[0]
        self.prompt_length = prompt_len
        ans = seq[:, prompt_len:]
        valid_ans_len = (ans != pad_token_id).sum(dim=-1)

        self._last_seq_lengths = valid_ans_len.cpu().tolist()
        self._last_prompt_length = prompt_len
        self._last_total_seq_length = seq.shape[1]
        self._last_avg_batch_size = total_active_slots / max(decode_steps, 1)

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
            log_probs = compute_log_probs_microbatch(
                self.actor_model, seq, attention_mask, self.compute_fp32_loss)
            ref_log_probs = compute_log_probs_microbatch(
                self.ref_model, seq, attention_mask, self.compute_fp32_loss)
        logprob_end = time.time()

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
            log_probs = compute_log_probs_microbatch(
                self.actor_model, seq, attention_mask, self.compute_fp32_loss)
            ref_log_probs = compute_log_probs_microbatch(
                self.ref_model, seq, attention_mask, self.compute_fp32_loss)
        logprob_end = time.time()

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
