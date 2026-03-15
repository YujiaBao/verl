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
this manager delegates directly to the worker group's generate_sequences.
"""

from __future__ import annotations

from verl import DataProto
from verl.utils.ray_utils import auto_await


class TinkerAgentLoopManager:
    """Lightweight rollout manager for the Tinker backend.

    Instead of launching local inference servers, this manager calls
    generate_sequences on the ActorRolloutRefWorker (which uses
    TinkerRollout internally).
    """

    def __init__(self, config, worker_group, rollout_resource_pool=None, reward_loop_worker_handles=None):
        self.config = config
        self.worker_group = worker_group
        # Empty list — Tinker has no local rollout replicas
        self.rollout_replicas = []

    @classmethod
    @auto_await
    async def create(cls, config, worker_group=None, rollout_resource_pool=None, reward_loop_worker_handles=None):
        return cls(config, worker_group, rollout_resource_pool, reward_loop_worker_handles)

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate sequences by calling the worker group's rollout directly.

        We call generate_sequences on the worker group which dispatches to the
        ActorRolloutRefWorker, which in turn calls TinkerRollout.generate_sequences.
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
        return output

    def start_profile(self):
        pass

    def stop_profile(self):
        pass
