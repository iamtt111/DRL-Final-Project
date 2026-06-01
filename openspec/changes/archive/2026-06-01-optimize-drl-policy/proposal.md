## Why
To resolve performance collapse and speed up convergence of DRL agents (MaskablePPO and MAPPO), we need to optimize the observation space with one-hot floor encoding (eliminating continuous floor height mapping difficulties) and smooth out discontinuous reward spikes.

## What Changes
- **ADDED**: Encode each elevator's current floor as a 16-dimensional one-hot vector in the global state, increasing the state vector dimension from 183 to 243.
- **MODIFIED**: Smooth out the emergency waiting time penalty in `src/rewards/reward_functions.py` to prevent gradient explosion and policy collapse.
- **MODIFIED**: Set learning rate to `1e-4` and entropy coefficient to `0.05` in PPO and MAPPO configs.

## Impact
- Affected specs: `elevator-gym-env` (ADDED One-Hot Floor Observation Space requirement)
- Affected code: `src/envs/building.py`, `src/envs/elevator_env.py`, `src/rewards/reward_functions.py`, `src/agents/mappo_agent.py`, `scripts/train_mappo.py`, `configs/train_ppo.yaml`, `configs/train_mappo.yaml`, `OpenSpec.md`
