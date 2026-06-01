## 1. Implementation
- [ ] 1.1 Update `get_state_vector` in `src/envs/building.py` to use a 16-dimensional one-hot representation for each elevator's floor index
- [ ] 1.2 Update the observation space size calculation in `src/envs/elevator_env.py` to match the new 243 dimensions
- [ ] 1.3 Smooth the emergency waiting penalty in `src/rewards/reward_functions.py`
- [ ] 1.4 Update the default `state_dim` parameter of `MAPPOAgent` in `src/agents/mappo_agent.py` to 243
- [ ] 1.5 Update hyperparameter yaml files (`train_ppo.yaml` and `train_mappo.yaml`)
- [ ] 1.6 Update `OpenSpec.md` to reflect the 243-dimensional state vector and updated hyperparameters
- [ ] 1.7 Run unit tests and short training sweeps to verify correctness
