## Why
Implement the reinforcement learning training pipeline (Phase 3) including the MaskablePPO agent wrapper, a robust training script with configurations, a SARSA(λ) baseline agent with Tile Coding, and detailed reward tracking logging. This enables the core neural network training, comparison, and policy optimization.

## What Changes
- Create `src/agents/ppo_agent.py` wrapping `MaskablePPO` from `sb3_contrib`.
- Create `src/agents/sarsa_agent.py` implementing SARSA(λ) with Tile Coding to handle high-dimensional observations.
- Create `scripts/train.py` driving the training process with callbacks, checkpoints, and TensorBoard logging.
- Create `configs/train_ppo.yaml` with PPO hyperparameters.
- Set up custom tensorboard callbacks to log reward components (wait time penalty, energy penalty, emergency penalty, bonuses) independently.

## Impact
- Affected specs: `ppo-agent`, `training-pipeline`, `sarsa-agent`, `reward-tracking` (all new capabilities)
- Affected code: New agent wrappers, training entry scripts, configurations, and callbacks.
