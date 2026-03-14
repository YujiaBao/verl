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
Tinker engine implementation for verl.

Routes all GPU compute (forward, backward, optimizer, generation) through
Tinker's remote training and sampling APIs. No local model or GPU required.
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from functools import partial
from typing import Any, Callable, ContextManager, Generator, Optional

import torch
from tensordict import TensorDict

from verl.workers.engine.base import BaseEngine, EngineRegistry

from .data_utils import logprobs_to_padded_tensor, tensordict_to_datums, tensordict_to_model_inputs

logger = logging.getLogger(__name__)


# -- Loss function mapping --

VERL_LOSS_MODE_TO_TINKER = {
    "vanilla": "ppo",
    "cispo": "cispo",
    "bypass_mode": "importance_sampling",
}


def _resolve_tinker_loss(loss_function: Callable) -> tuple[str, dict[str, float]]:
    """
    Map a verl loss function (typically partial(ppo_loss, config=...)) to a
    Tinker loss function name and config dict.

    Returns:
        (tinker_loss_name, tinker_loss_config)
    """
    # Extract the ActorConfig from the partial's keywords
    config = None
    if isinstance(loss_function, partial):
        config = loss_function.keywords.get("config", None)

    if config is None:
        # Default to importance_sampling (REINFORCE-style, no clipping)
        return "importance_sampling", {}

    loss_mode = "vanilla"
    if hasattr(config, "policy_loss") and isinstance(config.policy_loss, dict):
        loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    if loss_mode not in VERL_LOSS_MODE_TO_TINKER:
        raise NotImplementedError(
            f"Loss mode '{loss_mode}' is not yet supported with the Tinker backend. "
            f"Supported modes: {list(VERL_LOSS_MODE_TO_TINKER.keys())}"
        )

    tinker_name = VERL_LOSS_MODE_TO_TINKER[loss_mode]
    tinker_config: dict[str, float] = {}

    if tinker_name == "ppo" and hasattr(config, "cliprange"):
        tinker_config["clip_param"] = config.cliprange

    return tinker_name, tinker_config


@EngineRegistry.register(model_type="language_model", backend="tinker")
class TinkerEngine(BaseEngine):
    """
    Engine that delegates all compute to Tinker's remote training service.

    Implements the BaseEngine interface so it can be used as a drop-in
    replacement for FSDPEngine, MegatronEngine, etc. in verl's training loop.
    """

    def __init__(
        self,
        model_config,
        engine_config,
        optimizer_config=None,
        checkpoint_config=None,
    ):
        super().__init__()
        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

        self._lr: float = getattr(engine_config, "learning_rate", 1e-4)
        self._step: int = 0
        self.mode = None

        # Tinker clients — created in initialize()
        self.service_client = None
        self.training_client = None
        self.latest_sampling_client = None

    def initialize(self):
        """Create Tinker training and sampling clients."""
        import tinker

        base_url = getattr(self.engine_config, "base_url", "") or os.environ.get("TINKER_BASE_URL", "")
        self.service_client = tinker.ServiceClient(base_url=base_url) if base_url else tinker.ServiceClient()

        model_name = self.model_config.model_name
        lora_rank = getattr(self.engine_config, "lora_rank", 32)

        logger.info(f"Creating Tinker training client for {model_name} with lora_rank={lora_rank}")
        self.training_client = self.service_client.create_lora_training_client(model_name, rank=lora_rank)

        # Create initial sampling client for inference (compute_log_prob, generation)
        self.latest_sampling_client = self.training_client.save_weights_and_get_sampling_client("init")
        logger.info("Tinker engine initialized")

    # -- Properties --

    @property
    def is_param_offload_enabled(self) -> bool:
        return False

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return False

    def is_mp_src_rank_with_outputs(self):
        return True

    def get_data_parallel_size(self):
        return 1

    def get_data_parallel_rank(self):
        return 0

    def get_data_parallel_group(self):
        return None

    # -- Mode switching (no-ops) --

    def train_mode(self, **kwargs):
        return nullcontext()

    def eval_mode(self, **kwargs):
        return nullcontext()

    def optimizer_zero_grad(self):
        pass

    # -- Core compute --

    def forward_backward_batch(
        self,
        data: TensorDict,
        loss_function: Callable,
        forward_only: bool = False,
    ) -> dict[str, Any]:
        """
        Forward (and optionally backward) pass via Tinker API.

        For inference (forward_only=True):
            Computes log_probs using the latest sampling client.

        For training (forward_only=False):
            Calls training_client.forward_backward() which computes loss,
            gradients, and returns new-policy logprobs.

        Returns dict matching verl's expected format:
            {"model_output": {"log_probs": Tensor}, "loss": [float], "metrics": dict}
        """
        if forward_only:
            return self._infer(data, loss_function)
        else:
            return self._train(data, loss_function)

    def _infer(self, data: TensorDict, loss_function: Callable | None) -> dict[str, Any]:
        """Compute log_probs for the current policy (inference only)."""
        model_inputs = tensordict_to_model_inputs(data)

        # Submit all compute_logprobs requests concurrently
        futures = [self.latest_sampling_client.compute_logprobs(mi) for mi in model_inputs]
        all_logprobs = [f.result() for f in futures]

        # Determine response lengths from response_mask
        response_mask = data.get("response_mask", None)
        if response_mask is not None:
            resp_lens = response_mask.sum(dim=-1).int().tolist()
            max_resp_len = int(response_mask.shape[-1])
        else:
            # Fallback: use full sequence logprobs
            resp_lens = [len(lp) for lp in all_logprobs]
            max_resp_len = max(resp_lens) if resp_lens else 0

        log_probs = logprobs_to_padded_tensor(all_logprobs, resp_lens, max_resp_len)

        return {
            "model_output": {"log_probs": log_probs},
            "loss": [0.0],
            "metrics": {},
        }

    def _train(self, data: TensorDict, loss_function: Callable) -> dict[str, Any]:
        """Forward + backward pass via Tinker's training API."""
        import tinker

        # Translate data
        datums = tensordict_to_datums(data, for_training=True)

        # Resolve loss function
        tinker_loss_name, tinker_loss_config = _resolve_tinker_loss(loss_function)

        # Call Tinker forward_backward
        result: tinker.ForwardBackwardOutput = self.training_client.forward_backward(
            data=datums,
            loss_fn=tinker_loss_name,
            loss_fn_config=tinker_loss_config if tinker_loss_config else None,
        ).result()

        # Extract new-policy logprobs from result
        logprobs_list = [
            output["logprobs"].to_torch().tolist() for output in result.loss_fn_outputs
        ]

        # Build padded log_probs tensor
        response_mask = data.get("response_mask", None)
        if response_mask is not None:
            resp_lens = response_mask.sum(dim=-1).int().tolist()
            max_resp_len = int(response_mask.shape[-1])
        else:
            resp_lens = [len(lp) for lp in logprobs_list]
            max_resp_len = max(resp_lens) if resp_lens else 0

        log_probs = logprobs_to_padded_tensor(logprobs_list, resp_lens, max_resp_len)

        loss_value = result.metrics.get("loss:sum", 0.0)

        self._step += 1

        return {
            "model_output": {"log_probs": log_probs},
            "loss": [loss_value],
            "metrics": {"actor/pg_loss": loss_value},
        }

    def optimizer_step(self) -> float:
        """Perform optimizer step via Tinker API."""
        import tinker

        adam_params = tinker.AdamParams(
            learning_rate=self._lr,
            beta1=getattr(self.engine_config, "adam_beta1", 0.9),
            beta2=getattr(self.engine_config, "adam_beta2", 0.95),
            eps=getattr(self.engine_config, "adam_eps", 1e-12),
            weight_decay=getattr(self.engine_config, "weight_decay", 0.0),
            grad_clip_norm=getattr(self.engine_config, "grad_clip_norm", 0.0),
        )
        self.training_client.optim_step(adam_params).result()

        # grad_norm is not available from Tinker
        return 0.0

    def lr_scheduler_step(self):
        """
        LR scheduling is handled externally by verl's TrainingWorker.

        The current LR value should be set via set_lr() before optimizer_step().
        """
        pass

    def set_lr(self, lr: float):
        """Set the learning rate for the next optimizer_step call."""
        self._lr = lr

    # -- Weight sync --

    def get_per_tensor_param(
        self, **kwargs
    ) -> tuple[Generator[tuple[str, torch.Tensor], None, None], Optional[dict]]:
        """
        Save a weight snapshot on Tinker and update the latest sampling client.

        Returns an empty generator since Tinker handles weights server-side —
        no weight tensors need to be transferred to the rollout worker.
        """
        name = f"step_{self._step}"
        self.latest_sampling_client = self.training_client.save_weights_and_get_sampling_client(name)
        logger.info(f"Saved Tinker snapshot: {name}")
        return iter([]), None

    # -- Checkpoint --

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Save training state to Tinker."""
        name = f"ckpt_step_{global_step}"
        self.training_client.save_state(name=name)
        logger.info(f"Saved Tinker checkpoint: {name}")

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: bool = True, **kwargs
    ) -> None:
        """Load checkpoint from Tinker (handled during initialize via resume)."""
        logger.warning("TinkerEngine.load_checkpoint is a no-op; use ServiceClient resume APIs directly.")

    # -- Device management (all no-ops) --

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        pass

    def disable_adapter(self) -> ContextManager:
        return nullcontext()
