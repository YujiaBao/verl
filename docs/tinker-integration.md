# Tinker Backend Integration for verl

## Overview

This document summarizes the integration of [Tinker](https://thinkingmachines.ai/tinker/) as a remote compute backend for verl. Tinker is an LLM post-training service that handles all GPU work (forward pass, backward pass, optimizer step, generation) server-side. This integration allows verl users to run RL training algorithms (GRPO, RLOO, REINFORCE++) without local GPUs.

## Background & Design Decisions

### Why verl?

We evaluated three integration targets:

1. **TRL (Hugging Face)** — `transformers.Trainer` inheritance chain is deeply coupled to local PyTorch. Every trainer mixes orchestration logic with local tensor operations. Replacing the compute backend would require reimplementing essentially the entire trainer — no code reuse of algorithm-level features (loss variants, advantage estimation, reward composition). Rejected.

2. **Standalone `trl-tinker` package** — Would require constant rebasing against TRL's fast-moving main branch. Maintenance burden outweighs distribution benefit. Rejected.

3. **verl** — Already has a `BaseEngine` abstraction and `EngineRegistry` for pluggable compute backends (FSDP, Megatron, TorchTitan). The training driver (`RayPPOTrainer.fit()`) orchestrates workers via RPC without touching GPU compute directly. Advantage estimators, reward computation, and logging all run on the CPU driver and are backend-agnostic. **Chosen.**

### Architecture

```
RayPPOTrainer.fit()                   ← UNCHANGED
         │
         ├── ActorRolloutRefWorker    ← Tinker init branch added
         │     ├── actor: _TinkerTrainingWorkerShim
         │     │     └── TinkerEngine(BaseEngine)
         │     │           └── tinker.TrainingClient
         │     ├── rollout: TinkerRollout(BaseRollout)
         │     │     └── tinker.SamplingClient
         │     └── ref: _TinkerTrainingWorkerShim (optional)
         │           └── TinkerEngine (separate TrainingClient)
         │
         └── No critic worker (Phase 1)
```

**Key design decisions:**

- **`_TinkerTrainingWorkerShim`** instead of full `TrainingWorker`: The standard `TrainingWorker.__init__` calls `initialize_global_process_group_ray()` and creates a device mesh, which requires GPUs. The shim provides the same interface (`infer_batch`, `train_batch`, `train_mini_batch`) but delegates directly to `TinkerEngine` without distributed setup.

- **Weight sync via shared reference**: Tinker weights never leave the server. `TinkerEngine.get_per_tensor_param()` creates a server-side snapshot and returns an empty weight generator. `TinkerRollout` picks up the new `SamplingClient` from `engine.latest_sampling_client` via a shared reference.

- **Loss function mapping**: verl passes `partial(ppo_loss, config=actor_config)` as a callable. `TinkerEngine` inspects the partial's config to determine the Tinker loss name (`"ppo"`, `"cispo"`, `"importance_sampling"`). Unsupported loss modes raise `NotImplementedError` with a clear message.

- **Concurrent generation**: `TinkerRollout.generate_sequences()` submits all `sample()` calls as futures, then collects results. Same pattern used by the tinker-cookbook for RL rollouts.

## What's Been Implemented

### New Files

| File | Purpose |
|---|---|
| `verl/workers/engine/tinker/__init__.py` | Package init |
| `verl/workers/engine/tinker/engine.py` | `TinkerEngine(BaseEngine)` — routes forward/backward/optim to Tinker API |
| `verl/workers/engine/tinker/data_utils.py` | TensorDict ↔ Tinker Datum/ModelInput translation |
| `verl/workers/rollout/tinker_rollout/__init__.py` | Package init |
| `verl/workers/rollout/tinker_rollout/rollout.py` | `TinkerRollout(BaseRollout)` — generation via SamplingClient |
| `verl/workers/config/tinker.py` | `TinkerEngineConfig` dataclass |
| `examples/tinker/grpo_gsm8k.yaml` | Example GRPO config with Tinker backend |
| `tests/unit/tinker_backend/test_data_utils.py` | Data translation unit tests (12/12 passing) |
| `tests/unit/tinker_backend/test_engine.py` | Loss mapping unit tests (12/12 passing) |
| `tests/e2e/tinker/test_grpo.py` | E2E GRPO training loop |
| `tests/e2e/tinker/test_rloo.py` | E2E RLOO training loop |
| `tests/e2e/tinker/test_reinforce_pp.py` | E2E REINFORCE++ training loop |
| `tests/e2e/tinker/README.md` | E2E test documentation |

### Modified Files

| File | Change |
|---|---|
| `verl/workers/rollout/base.py` | Added `("tinker", "async")` to `_ROLLOUT_REGISTRY` (+1 line) |
| `verl/workers/engine_workers.py` | Added `_TinkerTrainingWorkerShim` class and `_init_tinker_model()` in `ActorRolloutRefWorker` |

### Test Results

- **12/12 unit tests passing** (data translation + loss mapping)
- **3 E2E test scripts** ready for validation with a live Tinker service

## What Remains (TODO)

### Must-Have for Initial PR

- [ ] **Run E2E tests against live Tinker service** — The E2E tests (`test_grpo.py`, `test_rloo.py`, `test_reinforce_pp.py`) need validation with a real `TINKER_API_KEY`. May surface data format issues in the translation layer.

- [ ] **Validate `forward_backward` return format** — Verify that `result.loss_fn_outputs[i]["logprobs"]` logprob lengths match verl's expected `(bs, response_len)` shape after the `logprobs_to_padded_tensor` conversion. The logprobs from Tinker cover the full sequence; we extract the response portion — need to confirm alignment.

- [ ] **Test with verl's full training driver (`main_ppo.py`)** — The E2E tests bypass Ray and call TinkerEngine directly. Need to also verify the integration works through `RayPPOTrainer.fit()` → `ActorRolloutRefWorker._init_tinker_model()` → full pipeline. This requires a single-GPU node to satisfy Ray's setup.

- [ ] **LR scheduling bridge** — `TinkerEngine` stores `self._lr` and passes it to `optim_step()`. Need to verify that verl's `TrainingWorker` calls `lr_scheduler_step()` at the right time and that we correctly read the scheduled LR. May need to hook into verl's LR scheduler output.

### Nice-to-Have (Phase 2)

- [ ] **`forward_backward_custom` support** — For exotic loss modes (`dppo_tv`, `gspo`, `sapo`, `vespo`), use Tinker's `forward_backward_custom` which gives us logprobs to compute arbitrary losses client-side. Currently these raise `NotImplementedError`.

- [ ] **Nested tensor support** — Phase 1 uses padded format (`use_remove_padding=False`). Adding nested tensor (jagged layout) support would improve memory efficiency for variable-length sequences.

- [ ] **CPU-only mode** — Currently requires a single GPU node to satisfy Ray/verl's distributed setup. A Tinker-specific init path that fully bypasses `init_device_mesh()` and NCCL would enable pure CPU operation.

- [ ] **Critic support (PPO with GAE)** — Phase 1 covers critic-free algorithms. PPO with GAE needs a value model, which would be a second `TrainingClient`. The architecture supports this (separate `TinkerEngine` for critic), but the data flow (value predictions, GAE computation) needs implementation.

- [ ] **`TrainingWorker._postprocess_output` compatibility** — The shim bypasses `_postprocess_output` which handles `all_reduce` on metrics. For single-process Tinker runs this is fine, but if Tinker is used alongside local workers (e.g., local critic + Tinker actor), metric aggregation may need attention.

- [ ] **Checkpoint resume** — `TinkerEngine.load_checkpoint()` is currently a no-op. Need to implement resume via `ServiceClient.create_training_client_from_state_async()`.

- [ ] **Upstream PR to verl-project/verl** — The integration uses soft imports (`import tinker` inside methods) and all new code is in `tinker/` subdirectories. No existing tests or functionality are affected. The PR pitch: "Tinker lets users run verl's RL algorithms without local GPUs — same algorithms, zero infrastructure."

## How to Run

### Unit tests (no Tinker API needed)

```bash
cd ~/Repos/verl
.venv/bin/python -m pytest tests/unit/tinker_backend/ -v
```

### E2E tests (requires TINKER_API_KEY)

```bash
export TINKER_API_KEY=your_key_here

# GRPO
.venv/bin/python tests/e2e/tinker/test_grpo.py --model meta-llama/Llama-3.1-8B --steps 3

# RLOO (4 samples per prompt)
.venv/bin/python tests/e2e/tinker/test_rloo.py --model meta-llama/Llama-3.1-8B --steps 3

# REINFORCE++
.venv/bin/python tests/e2e/tinker/test_reinforce_pp.py --model meta-llama/Llama-3.1-8B --steps 3
```

## Repository

- **Fork:** https://github.com/YujiaBao/verl
- **Branch:** `feature/tinker-backend`
- **Upstream:** https://github.com/verl-project/verl
