## Why
Initialize the project infrastructure (Phase 1) including the core simulation components: single elevator physical model, multi-elevator building coordinator, passenger lifecycle tracker, and Poisson traffic generator. This forms the foundation for the Gymnasium environment and training algorithms.

## What Changes
- Create `src/envs/elevator.py` containing elevator physics, kinematics, and state machine.
- Create `src/envs/passenger.py` containing passenger states and priority levels.
- Create `src/envs/building.py` managing floors, passenger queues, and elevator coordination.
- Create `src/envs/traffic_generator.py` implementing Poisson-based and hospital-specific arrivals.
- Set up project configs in `configs/env_default.yaml`.
- Set up project metadata in `pyproject.toml` and dependencies in `requirements.txt`.

## Impact
- Affected specs: `elevator-physics`, `building-model`, `passenger-model`, `traffic-generator` (all new capabilities)
- Affected code: New codebase setup under `src/envs/`
