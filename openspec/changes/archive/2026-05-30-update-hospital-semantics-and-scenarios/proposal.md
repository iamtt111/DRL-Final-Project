## Why
Based on evaluation feedback, the physical simulator needs to capture hospital boarding/alighting delays realistically, since emergency stretcher beds and wheelchairs take significantly longer to load/unload than general passengers. Furthermore, a disaster crisis stress test scenario is needed to validate dispatcher stability under sudden bursts of high-rate emergency patient arrivals.

## What Changes
- Implement priority-dependent boarding and deboarding delays in the physics engine.
- Update the traffic generator to load priority event rates from scenario-specific configurations.
- Create a new `disaster_crisis` scenario config representing an influx of emergency casualties.
- Increase the default training timesteps to 1M steps in PPO and MAPPO configs.

## Impact
- Affected specs: `priority-system`, `traffic-generator`
- Affected code: `src/envs/building.py`, `src/envs/traffic_generator.py`, `configs/scenarios/disaster_crisis.yaml`, `configs/train_ppo.yaml`, `configs/train_mappo.yaml`
