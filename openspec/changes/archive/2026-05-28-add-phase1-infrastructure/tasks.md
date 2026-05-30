## 1. Project Configuration & Boilerplate
- [x] 1.1 Create `requirements.txt` with dependencies (gymnasium, numpy, stable-baselines3, pygame, pyyaml)
- [x] 1.2 Create `pyproject.toml` to support editable installs (`pip install -e .`)
- [x] 1.3 Create `configs/env_default.yaml` with default building and elevator physical parameters

## 2. Simulation Core Modules
- [x] 2.1 Implement `src/envs/passenger.py` (Passenger data structure, PassengerState enum)
- [x] 2.2 Implement `src/envs/elevator.py` (Elevator physics, kinematics updates, state machine, SCAN logic)
- [x] 2.3 Implement `src/envs/building.py` (Floor queues, multi-elevator management, state vector generation)
- [x] 2.4 Implement `src/envs/traffic_generator.py` (Poisson arrival process, hospital priority events)

## 3. Unit Tests & Verification
- [x] 3.1 Create `tests/test_elevator.py` for physics, motion kinematics, and state transitions
- [x] 3.2 Create `tests/test_passenger.py` for state transitions and wait time logging
- [x] 3.3 Create `tests/test_traffic.py` for Poisson distribution check and priority event injection
- [x] 3.4 Verify that unit tests run and pass using `pytest`
