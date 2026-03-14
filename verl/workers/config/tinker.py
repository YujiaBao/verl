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

from dataclasses import dataclass


@dataclass
class TinkerEngineConfig:
    """Configuration for the Tinker remote training backend."""

    model_name: str = ""
    base_url: str = ""  # Tinker service URL; falls back to TINKER_BASE_URL env var
    lora_rank: int = 32
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-12
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0
