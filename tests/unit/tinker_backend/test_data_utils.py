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
"""Unit tests for Tinker data translation utilities."""

import pytest
import torch
from tensordict import TensorDict

tinker = pytest.importorskip("tinker", reason="tinker package not installed")

from verl.workers.engine.tinker.data_utils import (
    extract_prompt_ids,
    logprobs_to_padded_tensor,
    tensordict_to_datums,
    tensordict_to_model_inputs,
)


def _make_tensordict(
    bs: int = 4,
    prompt_len: int = 10,
    response_len: int = 8,
    include_training_data: bool = False,
) -> TensorDict:
    """Create a synthetic TensorDict mimicking verl's batch format."""
    seq_len = prompt_len + response_len

    # Prompt tokens (left-padded: first 2 positions are padding)
    pad = 2
    input_ids = torch.randint(1, 1000, (bs, seq_len))
    input_ids[:, :pad] = 0  # padding tokens

    attention_mask = torch.ones(bs, seq_len)
    attention_mask[:, :pad] = 0

    response_mask = torch.zeros(bs, response_len)
    # First `response_len - 1` tokens are real, last 1 is padding
    real_resp = response_len - 1
    response_mask[:, :real_resp] = 1

    data = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
    }

    if include_training_data:
        data["old_log_probs"] = torch.randn(bs, response_len)
        data["advantages"] = torch.randn(bs, response_len)

    return TensorDict(data, batch_size=[bs])


class TestTensordictToDatums:
    def test_inference_mode(self):
        """In inference mode, datums should have empty loss_fn_inputs."""
        data = _make_tensordict(bs=3, prompt_len=10, response_len=8)
        datums = tensordict_to_datums(data, for_training=False)

        assert len(datums) == 3
        for datum in datums:
            assert isinstance(datum, tinker.Datum)
            assert datum.loss_fn_inputs == {}
            # Model input should have tokens (non-padded)
            tokens = datum.model_input.chunks[0].tokens
            assert len(tokens) == 10 + 8 - 2  # seq_len minus 2 padding tokens

    def test_training_mode(self):
        """In training mode, datums should have loss_fn_inputs populated."""
        data = _make_tensordict(bs=2, prompt_len=10, response_len=8, include_training_data=True)
        datums = tensordict_to_datums(data, for_training=True)

        assert len(datums) == 2
        for datum in datums:
            assert "target_tokens" in datum.loss_fn_inputs
            assert "logprobs" in datum.loss_fn_inputs
            assert "advantages" in datum.loss_fn_inputs
            assert "mask" in datum.loss_fn_inputs

            # Check lengths match response length (minus 1 padding token in response_mask)
            real_resp = 7  # response_len(8) - 1 padding
            assert len(datum.loss_fn_inputs["target_tokens"].data) == real_resp
            assert len(datum.loss_fn_inputs["logprobs"].data) == real_resp
            assert len(datum.loss_fn_inputs["advantages"].data) == real_resp

    def test_padding_stripped(self):
        """Verify that padding tokens are stripped from model_input."""
        bs, prompt_len, response_len = 1, 5, 3
        seq_len = prompt_len + response_len

        input_ids = torch.tensor([[0, 0, 10, 20, 30, 40, 50, 60]])  # 2 pad + 6 real
        attention_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        response_mask = torch.tensor([[1.0, 1.0, 0.0]])

        data = TensorDict(
            {"input_ids": input_ids, "attention_mask": attention_mask, "response_mask": response_mask},
            batch_size=[1],
        )

        datums = tensordict_to_datums(data, for_training=False)
        tokens = datums[0].model_input.chunks[0].tokens
        assert tokens == [10, 20, 30, 40, 50, 60]  # padding stripped


class TestTensordictToModelInputs:
    def test_basic(self):
        data = _make_tensordict(bs=2, prompt_len=5, response_len=3)
        model_inputs = tensordict_to_model_inputs(data)
        assert len(model_inputs) == 2
        for mi in model_inputs:
            assert isinstance(mi, tinker.ModelInput)


class TestLogprobsToPaddedTensor:
    def test_padding(self):
        logprobs_list = [
            [0.1, 0.2, 0.3, 0.4, 0.5],  # 5 tokens total, last 3 are response
            [0.6, 0.7, 0.8],  # 3 tokens total, last 2 are response
        ]
        resp_lens = [3, 2]
        max_resp_len = 4

        result = logprobs_to_padded_tensor(logprobs_list, resp_lens, max_resp_len)
        assert result.shape == (2, 4)
        # First example: last 3 logprobs + 1 padding
        assert torch.allclose(result[0], torch.tensor([0.3, 0.4, 0.5, 0.0]))
        # Second example: last 2 logprobs + 2 padding
        assert torch.allclose(result[1], torch.tensor([0.7, 0.8, 0.0, 0.0]))


class TestExtractPromptIds:
    def test_basic(self):
        # Simulate DataProto-like object
        prompts_tensor = torch.tensor([[0, 0, 10, 20, 30], [0, 40, 50, 60, 70]])
        attn_mask = torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # 2 pad + 3 prompt + 3 response
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # 1 pad + 4 prompt + 3 response
            ]
        )

        class FakeDataProto:
            batch = {"prompts": prompts_tensor, "attention_mask": attn_mask}

        ids = extract_prompt_ids(FakeDataProto())
        assert ids == [[10, 20, 30], [40, 50, 60, 70]]
