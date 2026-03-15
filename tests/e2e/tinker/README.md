# Tinker Backend E2E Tests

End-to-end tests for verl's Tinker remote compute backend.

Each test runs a complete RL training loop using Tinker for all GPU compute
(generation, log prob computation, forward/backward, optimizer step).
The tests use simple arithmetic problems as the dataset and verify that
the full pipeline works.

## Prerequisites

```bash
export TINKER_API_KEY=your_key_here
pip install tinker
```

## Running

```bash
# GRPO (Group Relative Policy Optimization)
python tests/e2e/tinker/test_grpo.py --model meta-llama/Llama-3.1-8B --steps 3

# RLOO (Leave-One-Out baseline, 4 samples per prompt)
python tests/e2e/tinker/test_rloo.py --model meta-llama/Llama-3.1-8B --steps 3

# REINFORCE++ (token-level discounted returns)
python tests/e2e/tinker/test_reinforce_pp.py --model meta-llama/Llama-3.1-8B --steps 3
```

## What each test does

### GRPO (`test_grpo.py`)
1. Generate arithmetic prompts ("What is 42 + 17?")
2. Sample completions from the model via `SamplingClient`
3. Score with exact-match reward (1.0 if correct, 0.0 otherwise)
4. Compute GRPO advantages (group-normalized by prompt)
5. Compute old-policy log probs via `compute_logprobs`
6. Run `forward_backward` + `optim_step` via `TrainingClient`
7. Save weight snapshot, repeat

### RLOO (`test_rloo.py`)
Same as GRPO but generates **N samples per prompt** (default 4) and uses
the leave-one-out baseline for variance reduction.

### REINFORCE++ (`test_reinforce_pp.py`)
Same as GRPO but uses **token-level discounted cumulative returns**
with batch-level whitening instead of group normalization.

## Algorithms × Tinker API mapping

| Step | verl component | Tinker API |
|------|---------------|------------|
| Generation | `SamplingClient.sample()` | Concurrent futures |
| Old log probs | `SamplingClient.compute_logprobs()` | Concurrent futures |
| Advantages | `core_algos.compute_*_advantage()` | Local CPU (no Tinker call) |
| Training | `TrainingClient.forward_backward()` | Server-side loss + gradients |
| Optimizer | `TrainingClient.optim_step()` | Server-side Adam update |
| Weight sync | `save_weights_and_get_sampling_client()` | Server-side snapshot |
