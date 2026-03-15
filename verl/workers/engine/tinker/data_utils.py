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
Data translation utilities between verl's TensorDict/DataProto and Tinker's Datum/ModelInput types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict

if TYPE_CHECKING:
    import tinker

    from verl import DataProto


def tensordict_to_datums(
    data: TensorDict,
    *,
    for_training: bool = False,
) -> list[tinker.Datum]:
    """
    Convert verl's batched TensorDict to a list of Tinker Datum objects.

    For inference (for_training=False):
        Only model_input is populated (token sequence).

    For training (for_training=True):
        Also populates loss_fn_inputs with old_log_probs, advantages, response_mask
        as required by Tinker's built-in loss functions (ppo, importance_sampling, etc.).

    Args:
        data: TensorDict with keys like input_ids, attention_mask, response_mask,
              old_log_probs, advantages.
        for_training: Whether to include loss_fn_inputs for training.

    Returns:
        List of tinker.Datum objects, one per example in the batch.
    """
    import tinker

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    bs = input_ids.shape[0]

    datums = []
    for i in range(bs):
        # Extract non-padded tokens using attention mask
        mask_i = attention_mask[i].bool()
        tokens = input_ids[i][mask_i].tolist()
        model_input = tinker.ModelInput.from_ints(tokens)

        loss_fn_inputs: dict[str, tinker.TensorData] = {}

        if for_training:
            response_mask = data["response_mask"]
            resp_mask_i = response_mask[i]
            resp_len = int(resp_mask_i.sum().item())
            seq_len = len(tokens)  # non-padded sequence length

            if resp_len > 0:
                # Tinker expects all loss_fn_inputs to be full-sequence-length,
                # matching the number of tokens in model_input.

                # Target tokens: full sequence (Tinker applies shifting internally)
                loss_fn_inputs["target_tokens"] = tinker.TensorData.from_numpy(
                    np.array(tokens, dtype=np.int64)
                )

                # Prompt length in the non-padded sequence
                prompt_len = seq_len - resp_len

                # Old policy log probs — zero for prompt, real values for response
                if "old_log_probs" in data.keys():
                    full_lp = torch.zeros(seq_len, dtype=torch.float32)
                    old_lp = data["old_log_probs"][i][:resp_len]
                    full_lp[prompt_len : prompt_len + resp_len] = old_lp
                    loss_fn_inputs["logprobs"] = tinker.TensorData.from_torch(full_lp)

                # Per-token advantages — zero for prompt, real values for response
                if "advantages" in data.keys():
                    full_adv = torch.zeros(seq_len, dtype=torch.float32)
                    adv = data["advantages"][i][:resp_len]
                    full_adv[prompt_len : prompt_len + resp_len] = adv
                    loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(full_adv)

        datums.append(tinker.Datum(model_input=model_input, loss_fn_inputs=loss_fn_inputs))

    return datums


def tensordict_to_model_inputs(data: TensorDict) -> list[tinker.ModelInput]:
    """
    Extract ModelInput objects from a TensorDict (for inference / logprob computation).

    Args:
        data: TensorDict with input_ids and attention_mask.

    Returns:
        List of tinker.ModelInput, one per example.
    """
    import tinker

    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    bs = input_ids.shape[0]

    model_inputs = []
    for i in range(bs):
        mask_i = attention_mask[i].bool()
        tokens = input_ids[i][mask_i].tolist()
        model_inputs.append(tinker.ModelInput.from_ints(tokens))

    return model_inputs


def logprobs_to_padded_tensor(
    logprobs_list: list[list[float]],
    response_lengths: list[int],
    max_response_length: int,
) -> torch.Tensor:
    """
    Convert per-example logprob lists to a padded (bs, max_response_len) tensor.

    Tinker returns logprobs for the full sequence. We extract only the response
    portion and pad to a uniform length.

    Args:
        logprobs_list: Per-example logprobs from Tinker (full sequence length).
        response_lengths: Number of response tokens per example.
        max_response_length: Target padding length.

    Returns:
        Padded tensor of shape (bs, max_response_length).
    """
    padded = []
    for logprobs, resp_len in zip(logprobs_list, response_lengths):
        # Take the last resp_len logprobs (response portion)
        resp_logprobs = logprobs[-resp_len:] if resp_len > 0 else []
        t = torch.tensor(resp_logprobs, dtype=torch.float32)
        pad_size = max_response_length - len(t)
        if pad_size > 0:
            t = F.pad(t, (0, pad_size), value=0.0)
        padded.append(t)

    return torch.stack(padded, dim=0)


def sample_responses_to_dataproto(
    prompt_ids_list: list[list[int]],
    sample_results: list,
    max_prompt_length: int,
    max_response_length: int,
    pad_token_id: int,
) -> DataProto:
    """
    Convert Tinker SampleResponse objects to verl's DataProto format.

    Args:
        prompt_ids_list: Per-example prompt token IDs.
        sample_results: List of Tinker SampleResponse (one per prompt).
        max_prompt_length: Max prompt length for left-padding.
        max_response_length: Max response length for right-padding.
        pad_token_id: Token ID used for padding.

    Returns:
        DataProto with batch keys: prompts, responses, input_ids,
        attention_mask, rollout_log_probs.
    """
    from verl import DataProto

    all_prompts = []
    all_responses = []
    all_input_ids = []
    all_attention_mask = []
    all_rollout_logprobs = []

    for prompt_ids, result in zip(prompt_ids_list, sample_results):
        seq = result.sequences[0]
        response_ids = list(seq.tokens)

        # Truncate response to max length
        response_ids = response_ids[:max_response_length]
        resp_len = len(response_ids)

        # Left-pad prompt
        prompt_len = len(prompt_ids)
        prompt_pad = max_prompt_length - prompt_len
        padded_prompt = [pad_token_id] * prompt_pad + prompt_ids

        # Right-pad response
        resp_pad = max_response_length - resp_len
        padded_response = response_ids + [pad_token_id] * resp_pad

        # Concatenated input_ids
        padded_input = padded_prompt + padded_response

        # Attention mask: 0 for padding, 1 for real tokens
        prompt_mask = [0] * prompt_pad + [1] * prompt_len
        response_mask = [1] * resp_len + [0] * resp_pad
        attn_mask = prompt_mask + response_mask

        all_prompts.append(torch.tensor(padded_prompt, dtype=torch.long))
        all_responses.append(torch.tensor(padded_response, dtype=torch.long))
        all_input_ids.append(torch.tensor(padded_input, dtype=torch.long))
        all_attention_mask.append(torch.tensor(attn_mask, dtype=torch.float32))

        # Rollout logprobs (if available)
        if seq.logprobs is not None:
            lp = list(seq.logprobs)[:max_response_length]
            lp_padded = lp + [0.0] * (max_response_length - len(lp))
            all_rollout_logprobs.append(torch.tensor(lp_padded, dtype=torch.float32))
        else:
            all_rollout_logprobs.append(torch.zeros(max_response_length, dtype=torch.float32))

    batch = {
        "prompts": torch.stack(all_prompts),
        "responses": torch.stack(all_responses),
        "input_ids": torch.stack(all_input_ids),
        "attention_mask": torch.stack(all_attention_mask),
        "rollout_log_probs": torch.stack(all_rollout_logprobs),
    }

    return DataProto.from_single_dict(batch)


def extract_prompt_ids(prompts: DataProto) -> list[list[int]]:
    """
    Extract per-example prompt token IDs from a DataProto, stripping padding.

    Handles both formats:
      - E2E tests: batch keys "prompts" and "attention_mask"
      - Training driver: batch keys "input_ids" and "attention_mask"

    Args:
        prompts: DataProto with prompt token IDs and attention mask.

    Returns:
        List of token ID lists (no padding).
    """
    if "prompts" in prompts.batch.keys():
        prompt_ids = prompts.batch["prompts"]
    elif "input_ids" in prompts.batch.keys():
        prompt_ids = prompts.batch["input_ids"]
    else:
        raise KeyError(f"Expected 'prompts' or 'input_ids' in batch, got: {list(prompts.batch.keys())}")

    attention_mask = prompts.batch["attention_mask"]
    prompt_len = prompt_ids.shape[1]
    bs = prompt_ids.shape[0]

    result = []
    for i in range(bs):
        # Attention mask covers full sequence; take only prompt portion
        mask_i = attention_mask[i][:prompt_len].bool()
        tokens = prompt_ids[i][mask_i].tolist()
        result.append(tokens)

    return result
