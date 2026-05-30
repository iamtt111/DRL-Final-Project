## 1. PPO Agent Wrapper & Configurations
- [x] 1.1 Create `configs/train_ppo.yaml` containing PPO neural network and hyperparameters
- [x] 1.2 Implement `src/agents/ppo_agent.py` wrapping `MaskablePPO` and implementing training/evaluation hooks

## 2. SARSA(λ) Baseline Agent
- [x] 2.1 Implement `src/agents/sarsa_agent.py` implementing Tile Coding function approximation and eligibility traces for high-dimensional action selection

## 3. Training & Reward Tracking Pipeline
- [x] 3.1 Implement `scripts/train.py` managing training setup, checkpoints, evaluation callbacks, and logging
- [x] 3.2 Implement a custom TensorBoard logger callback in `src/utils/logger.py` to record decomposed reward signals

## 4. Verification & Testing
- [x] 4.1 Create `tests/test_ppo.py` verifying PPO agent interaction with `action_masks`
- [x] 4.2 Create `tests/test_sarsa.py` verifying Tile Coding feature extraction and SARSA(λ) updating
- [x] 4.3 Verify with pytest
