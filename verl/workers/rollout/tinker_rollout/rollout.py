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
Tinker rollout implementation for verl.

Handles text generation via Tinker's remote SamplingClient.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Generator

import torch

from verl import DataProto

from ..base import BaseRollout
from ...engine.tinker.data_utils import extract_prompt_ids, sample_responses_to_dataproto

if TYPE_CHECKING:
    import tinker

logger = logging.getLogger(__name__)


class TinkerRollout(BaseRollout):
    """
    Rollout backend that generates text via Tinker's SamplingClient.

    Unlike vLLM/SGLang rollouts, this does not manage local GPU memory or
    weight loading. Weights are managed server-side by Tinker.
    """

    def __init__(self, config, model_config, device_mesh=None, *args, **kwargs):
        # Intentionally do NOT call super().__init__() — it requires a non-None
        # device_mesh which Tinker doesn't need.
        self.config = config
        self.model_config = model_config
        self.device_mesh = device_mesh

        self.sampling_client: tinker.SamplingClient | None = None
        self._engine_ref = None  # Set by ActorRolloutRefWorker to share SamplingClient

    async def resume(self, tags: list[str]):
        """No-op — Tinker has no local GPU memory to resume."""
        pass

    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        **kwargs,
    ):
        """
        Update the sampling client to use the latest trained weights.

        With Tinker, weight tensors are not transferred. Instead, the
        TinkerEngine saves a snapshot server-side and updates its
        latest_sampling_client. We pick up that reference here.
        """
        if self._engine_ref is not None:
            self.sampling_client = self._engine_ref.latest_sampling_client
            logger.debug("TinkerRollout updated sampling client from engine reference")
        else:
            logger.warning("TinkerRollout has no engine reference; cannot update weights")

    async def release(self):
        """No-op — Tinker has no local GPU memory to release."""
        pass

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        Generate completions for a batch of prompts using Tinker.

        Args:
            prompts: DataProto with batch keys "prompts" and "attention_mask".

        Returns:
            DataProto with batch keys: prompts, responses, input_ids,
            attention_mask, rollout_log_probs.
        """
        import tinker

        if self.sampling_client is None:
            raise RuntimeError(
                "TinkerRollout.sampling_client is None. "
                "Call update_weights() or set _engine_ref before generating."
            )

        prompt_ids_list = extract_prompt_ids(prompts)

        temperature = prompts.meta_info.get("temperature", 1.0)
        response_length = getattr(self.config, "response_length", 1024)

        sampling_params = tinker.SamplingParams(
            max_tokens=response_length,
            temperature=temperature,
        )

        # Submit all generation requests concurrently
        futures = []
        for prompt_ids in prompt_ids_list:
            model_input = tinker.ModelInput.from_ints(prompt_ids)
            futures.append(
                self.sampling_client.sample(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=sampling_params,
                    include_prompt_logprobs=False,
                )
            )

        # Collect results
        results = [f.result() for f in futures]

        # Determine padding dimensions
        max_prompt_length = max(len(ids) for ids in prompt_ids_list)
        pad_token_id = getattr(self.config, "pad_token_id", 0)

        return sample_responses_to_dataproto(
            prompt_ids_list=prompt_ids_list,
            sample_results=results,
            max_prompt_length=max_prompt_length,
            max_response_length=response_length,
            pad_token_id=pad_token_id,
        )
