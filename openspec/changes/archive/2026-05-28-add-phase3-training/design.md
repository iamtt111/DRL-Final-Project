## Context
We need to train a Deep Reinforcement Learning (DRL) agent to control the elevator group control system, evaluate it against a traditional RL baseline (SARSA(λ)), and track multi-objective reward metrics to ensure convergence and prevent reward hacking.

## Goals / Non-Goals
- Goals:
  - Wrap `MaskablePPO` from `sb3-contrib` to filter invalid dispatch decisions during training.
  - Implement a training script that schedules checkpoints and logs metrics to TensorBoard.
  - Adapt SARSA(λ) to the 183-dimensional continuous observation space using hash-based Tile Coding (linear function approximation).
  - Implement decomposed reward logging to track wait time, energy, and emergency penalties separately.
- Non-Goals:
  - Phase 4 final comparisons and statistical tests.
  - Pygame rendering during high-speed training (rendering will be disabled during training to optimize performance).

## Decisions
- **Tile Coding Design**:
  - We use $N_T = 8$ overlapping tilings.
  - A hash function with table size $D = 8192$ will map continuous multidimensional coordinates to a sparse index vector $\phi(\mathbf{s}) \in \{0, 1\}^D$.
  - We use standard replacing eligibility traces $\mathbf{e}$ with decay rate $\lambda = 0.5$.
- **TensorBoard Logging Callback**:
  - We will implement a custom SB3 Callback `RewardTrackingCallback` that retrieves step-level reward component info from the environment and records it under the tensorboard logs.

## Risks / Trade-offs
- **Divergence of SARSA(λ) in high dimensions**: Tabular methods struggle with high-dimensional spaces even with Tile Coding.
  - *Mitigation*: We use a relatively high tile width and a small number of active features per update to keep the updates sparse and stable.
- **Reward Hacking**: The agent could optimize the energy penalty at the expense of wait times (or vice versa).
  - *Mitigation*: We log each penalty component separately, allowing us to inspect curves and balance the reward weights $w_1, w_2, w_3$.

## Open Questions
- None.
