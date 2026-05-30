## Context
We need to connect the simulation components developed in Phase 1 to reinforcement learning agents via a standard Gymnasium interface. Additionally, we need traditional baseline control (Nearest Car) and real-time visualization to evaluate the agent's performance.

## Goals / Non-Goals
- Goals:
  - Connect the `Building` physics engine to a `gymnasium.Env` wrapper.
  - Implement a multi-objective reward function.
  - Support Action Masking to filter out full or disabled elevators.
  - Implement emergency preemption logic.
  - Render simulator frames using Pygame.
- Non-Goals:
  - Training the PPO agent (Phase 3).
  - SARSA(λ) baseline implementation (Phase 3).

## Decisions
- **Event-Driven vs Time-Driven RL Action**:
  - The RL agent will act only when a dispatch event occurs (i.e. a new Hall Call or Priority Event is registered).
  - Inside the environment `step()`, the simulation will continuously step forward by `dt = 1.0` seconds until the next dispatch event is triggered, at which point control is returned to the agent with a new observation. This is standard event-driven MDP modeling.
- **Action Masking**: We will define `action_masks(self) -> np.ndarray` returning a boolean array where `True` represents available elevators (capacity not exceeded and not out of service).

## Risks / Trade-offs
- **Infinite Loop in Event-Driven step**: If no events occur, the step function might run forever.
  - *Mitigation*: The step function will terminate and return `terminated=True` when `current_time >= max_time` (600s), ensuring the loop always terminates.

## Open Questions
- None at this stage.
