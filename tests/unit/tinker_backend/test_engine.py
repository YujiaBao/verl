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
"""Unit tests for TinkerEngine (no Tinker API calls)."""

from functools import partial
from types import SimpleNamespace

import pytest

from verl.workers.engine.tinker.engine import _resolve_tinker_loss


class TestResolveTinkerLoss:
    def test_vanilla_ppo(self):
        """vanilla loss mode maps to Tinker's ppo loss."""
        config = SimpleNamespace(
            policy_loss={"loss_mode": "vanilla"},
            cliprange=0.2,
        )
        loss_fn = partial(lambda config, model_output, data: None, config=config)

        name, extra = _resolve_tinker_loss(loss_fn)
        assert name == "ppo"
        assert extra["clip_low_threshold"] == pytest.approx(0.8)
        assert extra["clip_high_threshold"] == pytest.approx(1.2)

    def test_cispo(self):
        """cispo loss mode maps directly."""
        config = SimpleNamespace(
            policy_loss={"loss_mode": "cispo"},
        )
        loss_fn = partial(lambda config, model_output, data: None, config=config)

        name, cfg = _resolve_tinker_loss(loss_fn)
        assert name == "cispo"

    def test_bypass_mode(self):
        """bypass_mode maps to importance_sampling."""
        config = SimpleNamespace(
            policy_loss={"loss_mode": "bypass_mode"},
        )
        loss_fn = partial(lambda config, model_output, data: None, config=config)

        name, cfg = _resolve_tinker_loss(loss_fn)
        assert name == "importance_sampling"

    def test_no_config_defaults_to_importance_sampling(self):
        """When no config is provided, default to importance_sampling."""
        loss_fn = lambda model_output, data: None

        name, cfg = _resolve_tinker_loss(loss_fn)
        assert name == "importance_sampling"
        assert cfg == {}

    def test_unsupported_raises(self):
        """Unsupported loss modes should raise NotImplementedError."""
        config = SimpleNamespace(
            policy_loss={"loss_mode": "dppo_tv"},
        )
        loss_fn = partial(lambda config, model_output, data: None, config=config)

        with pytest.raises(NotImplementedError, match="dppo_tv"):
            _resolve_tinker_loss(loss_fn)

    def test_default_vanilla_when_loss_mode_missing(self):
        """When policy_loss dict exists but loss_mode is missing, default to vanilla."""
        config = SimpleNamespace(
            policy_loss={},
            cliprange=0.1,
        )
        loss_fn = partial(lambda config, model_output, data: None, config=config)

        name, cfg = _resolve_tinker_loss(loss_fn)
        assert name == "ppo"
