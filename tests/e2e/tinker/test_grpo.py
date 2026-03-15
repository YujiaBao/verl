#!/usr/bin/env python3
# Copyright 2025 Thinking Machines Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
End-to-end test: GRPO training with Tinker backend.

Runs a minimal GRPO training loop on arithmetic problems:
  1. Generate completions via TinkerRollout
  2. Score with a simple correctness reward
  3. Compute GRPO advantages
  4. Update policy via TinkerEngine

Requires:
  - TINKER_API_KEY environment variable
  - tinker Python package

Usage:
  python tests/e2e/tinker/test_grpo.py --model meta-llama/Llama-3.1-8B --steps 3
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
import torch

logger = logging.getLogger(__name__)


def make_arithmetic_prompts(n: int = 16) -> list[dict]:
    """Generate simple arithmetic problems as prompts."""
    rng = np.random.RandomState(42)
    problems = []
    for _ in range(n):
        a, b = rng.randint(1, 100, size=2)
        op = rng.choice(["+", "-", "*"])
        answer = eval(f"{a} {op} {b}")
        problems.append({
            "prompt": f"What is {a} {op} {b}? Answer with just the number.",
            "answer": str(answer),
        })
    return problems


def score_completions(completions: list[str], answers: list[str]) -> list[float]:
    """Score completions by checking if they contain the correct answer."""
    scores = []
    for completion, answer in zip(completions, answers):
        # Extract numbers from completion
        numbers = re.findall(r"-?\d+", completion)
        if answer in numbers:
            scores.append(1.0)
        else:
            scores.append(0.0)
    return scores


def run_grpo(model_name: str, n_steps: int, batch_size: int, lora_rank: int):
    """Run a GRPO training loop with Tinker backend."""
    import tinker
    from tensordict import TensorDict

    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    from verl.workers.engine.tinker.data_utils import (
        logprobs_to_padded_tensor,
        tensordict_to_datums,
        tensordict_to_model_inputs,
    )
    from verl.workers.engine.tinker.engine import TinkerEngine

    # -- Setup --
    engine_config = SimpleNamespace(
        base_url=os.environ.get("TINKER_BASE_URL", ""),
        lora_rank=lora_rank,
        learning_rate=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-12,
        weight_decay=0.0,
        grad_clip_norm=0.0,
    )
    model_config = SimpleNamespace(model_name=model_name)

    engine = TinkerEngine(
        model_config=model_config,
        engine_config=engine_config,
    )
    engine.initialize()
    logger.info(f"Initialized TinkerEngine for {model_name}")

    # Sampling client for generation
    sampling_client = engine.latest_sampling_client

    # -- Generate problems --
    problems = make_arithmetic_prompts(n=batch_size)
    tokenizer = engine.training_client.get_tokenizer()
    max_response_length = 64

    for step in range(n_steps):
        logger.info(f"=== Step {step + 1}/{n_steps} ===")

        # 1. Tokenize prompts
        prompt_ids_list = [tokenizer.encode(p["prompt"]) for p in problems]
        answers = [p["answer"] for p in problems]
        max_prompt_length = max(len(ids) for ids in prompt_ids_list)

        # 2. Generate completions
        sampling_params = tinker.SamplingParams(
            max_tokens=max_response_length,
            temperature=0.7,
        )
        futures = [
            sampling_client.sample(
                tinker.ModelInput.from_ints(ids),
                num_samples=1,
                sampling_params=sampling_params,
            )
            for ids in prompt_ids_list
        ]
        results = [f.result() for f in futures]

        # Decode completions
        completions = []
        response_ids_list = []
        response_logprobs_list = []
        for result in results:
            seq = result.sequences[0]
            tokens = list(seq.tokens)[:max_response_length]
            response_ids_list.append(tokens)
            completions.append(tokenizer.decode(tokens))
            if seq.logprobs is not None:
                response_logprobs_list.append(list(seq.logprobs)[:max_response_length])
            else:
                response_logprobs_list.append([0.0] * len(tokens))

        # 3. Score completions
        rewards = score_completions(completions, answers)
        mean_reward = np.mean(rewards)
        logger.info(f"  Mean reward: {mean_reward:.3f} ({sum(r > 0 for r in rewards)}/{len(rewards)} correct)")

        # 4. Build tensors for advantage computation
        max_resp_len = max(len(ids) for ids in response_ids_list)
        response_mask = torch.zeros(batch_size, max_resp_len)
        token_level_rewards = torch.zeros(batch_size, max_resp_len)
        for i, (ids, reward) in enumerate(zip(response_ids_list, rewards)):
            resp_len = len(ids)
            response_mask[i, :resp_len] = 1.0
            # Place reward on last token (outcome-based)
            token_level_rewards[i, resp_len - 1] = reward

        # 5. Compute GRPO advantages
        uids = np.array([f"uid_{i}" for i in range(batch_size)], dtype=object)
        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=uids,
        )
        logger.info(f"  Advantage mean: {advantages[response_mask.bool()].mean():.4f}, "
                     f"std: {advantages[response_mask.bool()].std():.4f}")

        # 6. Compute old log probs (current policy before update)
        old_logprobs_list = []
        seq_futures = []
        for prompt_ids, resp_ids in zip(prompt_ids_list, response_ids_list):
            full_seq = prompt_ids + resp_ids
            seq_futures.append(
                sampling_client.compute_logprobs(tinker.ModelInput.from_ints(full_seq))
            )
        all_seq_logprobs = [f.result() for f in seq_futures]

        old_log_probs = torch.zeros(batch_size, max_resp_len)
        for i, (prompt_ids, seq_lp) in enumerate(zip(prompt_ids_list, all_seq_logprobs)):
            resp_len = len(response_ids_list[i])
            prompt_len = len(prompt_ids)
            # Extract response portion logprobs
            resp_lp = seq_lp[prompt_len:prompt_len + resp_len]
            old_log_probs[i, :len(resp_lp)] = torch.tensor(resp_lp, dtype=torch.float32)

        # 7. Build TensorDict for training
        # Pad prompts (left) and responses (right) into input_ids
        input_ids = torch.zeros(batch_size, max_prompt_length + max_resp_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_prompt_length + max_resp_len)
        for i, (prompt_ids, resp_ids) in enumerate(zip(prompt_ids_list, response_ids_list)):
            p_len = len(prompt_ids)
            r_len = len(resp_ids)
            p_pad = max_prompt_length - p_len
            input_ids[i, p_pad:p_pad + p_len] = torch.tensor(prompt_ids)
            input_ids[i, max_prompt_length:max_prompt_length + r_len] = torch.tensor(resp_ids)
            attention_mask[i, p_pad:p_pad + p_len] = 1.0
            attention_mask[i, max_prompt_length:max_prompt_length + r_len] = 1.0

        data = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "old_log_probs": old_log_probs,
            "advantages": advantages,
        }, batch_size=[batch_size])

        # 8. Forward-backward via Tinker
        from functools import partial
        dummy_loss_fn = partial(lambda config, mo, d: None, config=SimpleNamespace(
            policy_loss={"loss_mode": "vanilla"},
            cliprange=0.2,
        ))

        output = engine.forward_backward_batch(data, dummy_loss_fn, forward_only=False)
        loss = output["loss"][0]
        logger.info(f"  Loss: {loss:.4f}")

        # 9. Optimizer step
        engine.optimizer_step()

        # 10. Update sampling client for next step
        engine.get_per_tensor_param()
        sampling_client = engine.latest_sampling_client
        logger.info(f"  Updated weights (snapshot step_{engine._step})")

    logger.info("GRPO training complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description="E2E GRPO test with Tinker backend")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B", help="Model name")
    parser.add_argument("--steps", type=int, default=3, help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not os.environ.get("TINKER_API_KEY"):
        logger.error("TINKER_API_KEY environment variable not set")
        sys.exit(1)

    success = run_grpo(args.model, args.steps, args.batch_size, args.lora_rank)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
