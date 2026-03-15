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
Tinker-specific AgentLoopManager.

Replaces the default AgentLoopManager which spins up local LLM servers
(vLLM, SGLang, TRT-LLM). Tinker handles generation server-side, so
this manager delegates directly to the worker group's generate_sequences
and computes rewards inline.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from verl import DataProto
from verl.utils.ray_utils import auto_await

logger = logging.getLogger(__name__)


class TinkerAgentLoopManager:
    """Lightweight rollout manager for the Tinker backend.

    Instead of launching local inference servers, this manager calls
    generate_sequences on the ActorRolloutRefWorker (which uses
    TinkerRollout internally), then computes rewards using verl's
    reward function.
    """

    def __init__(self, config, worker_group, rollout_resource_pool=None, reward_loop_worker_handles=None):
        self.config = config
        self.worker_group = worker_group
        self.reward_loop_worker_handles = reward_loop_worker_handles
        # Empty list — Tinker has no local rollout replicas
        self.rollout_replicas = []

        # Initialize tokenizer and reward function
        self._init_reward()

    def _init_reward(self):
        """Load the reward function from config."""
        from verl.trainer.ppo.reward import get_custom_reward_fn
        from verl.utils import hf_tokenizer
        from verl.utils.fs import copy_to_local
        from verl.utils.reward_score import default_compute_score

        model_path = self.config.actor_rollout_ref.model.path
        local_path = copy_to_local(model_path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.compute_score_fn = get_custom_reward_fn(self.config) or default_compute_score

    def _compute_rewards(self, output: DataProto, prompts: DataProto) -> DataProto:
        """Compute reward scores for generated sequences.

        Decodes responses, calls the reward function, and adds rm_scores to the batch.
        """
        bs = output.batch.shape[0]
        response_ids = output.batch["responses"]
        attention_mask = output.batch["attention_mask"]
        prompt_length = output.batch["prompts"].shape[-1]
        response_length = response_ids.shape[-1]
        # response_mask: 1 for real response tokens, 0 for padding
        response_attention = attention_mask[:, prompt_length:]
        response_mask = response_attention[:, :response_length]

        scores = []
        reward_extra_infos = []

        for i in range(bs):
            # Decode response
            resp_len = int(response_mask[i].sum().item())
            valid_resp_ids = response_ids[i][:resp_len]
            response_str = self.tokenizer.decode(valid_resp_ids, skip_special_tokens=True)

            # Get ground truth and data source from non_tensor_batch
            data_source = prompts.non_tensor_batch.get("data_source", np.array(["unknown"] * bs))[i]
            extra_info = prompts.non_tensor_batch.get("extra_info", np.array([{}] * bs))[i]

            # Compute reward
            score = 0.0
            extra = {}
            if self.compute_score_fn is not None:
                try:
                    result = self.compute_score_fn(
                        data_source=data_source,
                        solution_str=response_str,
                        ground_truth=extra_info.get("answer", ""),
                        extra_info=extra_info,
                    )
                    if isinstance(result, dict):
                        score = result.get("score", 0.0)
                        extra = result
                    else:
                        score = float(result)
                except Exception as e:
                    logger.warning(f"Reward computation failed for sample {i}: {e}")
                    score = 0.0

            scores.append(score)
            reward_extra_infos.append(extra)

        # Place reward on last valid response token (outcome-based)
        rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
        for i in range(bs):
            resp_len = int(response_mask[i].sum().item())
            if resp_len > 0:
                rm_scores[i, resp_len - 1] = scores[i]

        output.batch["rm_scores"] = rm_scores
        return output

    @classmethod
    @auto_await
    async def create(cls, config, worker_group=None, rollout_resource_pool=None, reward_loop_worker_handles=None):
        return cls(config, worker_group, rollout_resource_pool, reward_loop_worker_handles)

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences and compute rewards.

        Calls the worker group's generate_sequences (which dispatches to
        TinkerRollout), then computes rewards using the configured reward function.
        """
        # Call generate_sequences through the worker group.
        # The method uses ALL_TO_ALL dispatch which returns a list (one per worker).
        result = self.worker_group.generate_sequences(prompts)
        output = result[0] if isinstance(result, list) else result

        if output is None:
            raise RuntimeError("generate_sequences returned None")
        if output.meta_info is None:
            output.meta_info = {}
        if "timing" not in output.meta_info:
            output.meta_info["timing"] = {}

        # Compute rewards
        output = self._compute_rewards(output, prompts)

        return output

    def start_profile(self):
        pass

    def stop_profile(self):
        pass
