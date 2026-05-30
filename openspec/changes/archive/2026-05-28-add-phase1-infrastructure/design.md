## Context
We are building a Deep Reinforcement Learning (DRL) elevator group control system for a hospital. The system needs to respect three priority levels (Wheelchair, Medical Staff, Emergency Bed) and coordinate multiple elevators. The first phase requires a solid simulation engine that is mathematically precise, fast, and obeys realistic kinematics.

## Goals / Non-Goals
- Goals:
  - Implement a continuous physical simulation of elevator movement.
  - Implement passenger queues at each floor.
  - Implement Poisson-based traffic generators with hospital-specific priority event rates.
  - Create a clean test suite to verify physics and logic correctness.
- Non-Goals:
  - Gymnasium env wrapper or RL agent training (Phase 2 and 3).
  - Pygame visualization (Phase 2).

## Decisions
- **Kinematics Model**: We use continuous heights (meters) rather than discrete floor indices to calculate velocity, acceleration, and travel times. This aligns with modern simulator practices and ensures realistic energy consumption calculations.
- **SCAN Logic**: Elevators will follow a standard SCAN (selective collective) algorithm to process their `pending_stops` unless overridden by priority events.

## Risks / Trade-offs
- **Kinematics integration error**: Using large time steps (`dt = 1.0s`) could lead to overshooting target floors.
  - *Mitigation*: The `update(dt)` loop will analytically compute whether the elevator reaches the target floor during the time step and snap it to the exact floor height to prevent overshoot.

## Open Questions
- None at this stage.
