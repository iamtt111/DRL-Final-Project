## Why
The current single-agent MaskablePPO central dispatcher has a massive state space and suffers from poor average waiting time (AWT) and load imbalance. Introducing a cooperative Multi-Agent PPO (MAPPO) with parameter-sharing decentralized actors and a centralized critic will break this bottleneck and improve passenger service quality while maintaining energy efficiency.

## What Changes
- Add `elevator_ma_env.py` containing the new Multi-Agent Gymnasium-like Environment interface.
- Add `mappo_agent.py` implementing the centralized training with decentralized execution PyTorch policy.
- Add `train_mappo.py` for standard MAPPO training algorithm implementation and training CLI.
- Modify `compare_baselines.py` and `demo.py` to support evaluating and visualizing MAPPO agents.
- **NO BREAKING CHANGES** to existing baseline codes or tests.

## Impact
- Affected specs: `mappo-agent` (NEW), `elevator-ma-env` (NEW)
- Affected code: `src/envs/elevator_ma_env.py`, `src/agents/mappo_agent.py`, `scripts/train_mappo.py`, `scripts/compare_baselines.py`, `scripts/demo.py`
