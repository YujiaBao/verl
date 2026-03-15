# Tinker Backend for verl

## What is Tinker?

[Tinker](https://thinkingmachines.ai/tinker/) is an LLM post-training API that handles all GPU compute — forward pass, backward pass, optimizer step, and text generation — server-side. This integration lets verl users run RL training algorithms (GRPO, RLOO, REINFORCE++) through Tinker's API, without managing local model weights or GPU infrastructure.

## How it works

verl's training driver (`RayPPOTrainer`) orchestrates the RL loop: generate responses, compute rewards, estimate advantages, update the policy. Normally, each step runs on local GPUs via FSDP or Megatron. With the Tinker backend, these steps are routed to Tinker's remote API instead:

```
RayPPOTrainer.fit()
         │
         ├── generate_sequences()
         │     └── TinkerRollout → tinker.SamplingClient.sample()
         │
         ├── compute_log_prob()
         │     └── TinkerEngine → tinker.SamplingClient.compute_logprobs()
         │
         ├── compute_advantages()        ← runs locally (CPU, no change)
         │
         ├── update_actor()
         │     └── TinkerEngine → tinker.TrainingClient.forward_backward()
         │                      → tinker.TrainingClient.optim_step()
         │
         └── update_weights()
               └── TinkerEngine → tinker.TrainingClient.save_weights_and_get_sampling_client()
```

Advantage estimation, reward computation, and logging all run on the CPU driver and are backend-agnostic — they work exactly the same as with FSDP or Megatron.

## Quick start

### Prerequisites

```bash
pip install tinker
export TINKER_API_KEY=your_key_here
```

### Run GRPO training

```bash
python -m verl.trainer.main_ppo \
  --config-path=examples/tinker \
  --config-name=grpo_gsm8k \
  actor_rollout_ref.model.path=Qwen/Qwen3-8B \
  actor_rollout_ref.actor.engine.model_name=Qwen/Qwen3-8B \
  data.train_files=path/to/train.parquet \
  data.val_files=path/to/test.parquet
```

### Run E2E tests (standalone, no Ray)

```bash
# GRPO
python tests/e2e/tinker/test_grpo.py --model Qwen/Qwen3-8B --steps 3

# RLOO
python tests/e2e/tinker/test_rloo.py --model Qwen/Qwen3-8B --steps 3

# REINFORCE++
python tests/e2e/tinker/test_reinforce_pp.py --model Qwen/Qwen3-8B --steps 3
```

### Run unit tests (no API key needed)

```bash
python -m pytest tests/unit/tinker_backend/ -v
```

## Architecture

### New components

| Component | File | Role |
|---|---|---|
| `TinkerEngine` | `verl/workers/engine/tinker/engine.py` | `BaseEngine` subclass — routes forward/backward/optimizer to Tinker API |
| `TinkerRollout` | `verl/workers/rollout/tinker_rollout/rollout.py` | `BaseRollout` subclass — generates text via Tinker's SamplingClient |
| `TinkerAgentLoopManager` | `verl/workers/rollout/tinker_rollout/manager.py` | Replaces the default AgentLoopManager (which manages local vLLM/SGLang servers) with a lightweight manager that calls TinkerRollout and computes rewards |
| `_TinkerTrainingWorkerShim` | `verl/workers/engine_workers.py` | Lightweight wrapper mimicking `TrainingWorker` interface without distributed setup |
| Data translation | `verl/workers/engine/tinker/data_utils.py` | Converts between verl's TensorDict and Tinker's Datum/ModelInput formats |

### Changes to existing files

| File | Change |
|---|---|
| `verl/workers/rollout/base.py` | +1 line: register `("tinker", "async")` in rollout registry |
| `verl/workers/engine_workers.py` | Add shim class, `_init_tinker_model()`, and `generate_sequences()` |
| `verl/trainer/main_ppo.py` | +9 lines: register `tinker` strategy, skip separate ref policy worker |
| `verl/protocol.py` | +4 lines: add `DataProto.cpu()` convenience method |

All Tinker imports are soft (inside methods, not at module level). verl works normally without tinker installed.

## Design decisions

### Data format alignment

Tinker expects a right-shifted sequence format (matching the [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) conventions):

```
Original tokens:    [A, B, C, D, E, F]    (prompt=[A,B,C], response=[D,E,F])
model_input:        [A, B, C, D, E]        (tokens[:-1], what the model sees)
target_tokens:      [B, C, D, E, F]        (tokens[1:], what the model predicts)
logprobs:           [0, 0, lp_D, lp_E, lp_F]   (0 for observation, real for action)
advantages:         [0, 0, adv_D, adv_E, adv_F] (0 for observation, real for action)
```

The observation (prompt) positions have zero advantages, so the loss gradient is zero for prompt tokens regardless of the target values.

### Weight synchronization

Tinker weights never leave the server. Instead of transferring model parameters:

1. `TinkerEngine.get_per_tensor_param()` calls `save_weights_and_get_sampling_client()` on the Tinker server, creating a server-side snapshot
2. The new `SamplingClient` is shared with `TinkerRollout` via a reference
3. `TinkerRollout.update_weights()` picks up the latest `SamplingClient`

### Loss function mapping

verl's `ppo_loss` config is mapped to Tinker loss functions:

| verl loss_mode | Tinker loss_fn | Notes |
|---|---|---|
| `vanilla` | `ppo` | Clip thresholds via `loss_fn_config` |
| `cispo` | `cispo` | |
| `bypass_mode` | `importance_sampling` | REINFORCE-style, no clipping |

## Supported algorithms

- **GRPO** — Group Relative Policy Optimization (tested E2E)
- **RLOO** — Leave-One-Out baseline (tested E2E)
- **REINFORCE++** — Token-level discounted returns (tested E2E)

## Current limitations

- **Critic-free only** — PPO with GAE (value model) is not yet supported. A second `TrainingClient` for the critic model is architecturally possible but not implemented.
- **No checkpoint resume** — `TinkerEngine.load_checkpoint()` is a no-op. Resume via `ServiceClient.create_training_client_from_state()` is planned.
- **No mask field** — The `mask` key in `loss_fn_inputs` (0 for observation, 1 for action) is used in the tinker-cookbook but not yet accepted by the server (v0.15). Correctness is maintained because observation positions have zero advantages.
