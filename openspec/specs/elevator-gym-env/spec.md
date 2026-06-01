# elevator-gym-env Specification

## Purpose
TBD - created by archiving change add-phase2-integration. Update Purpose after archive.
## Requirements
### Requirement: Gym Wrapper and Stepping Loop
The elevator gymnasium environment SHALL wrap the building simulation, mapping actions to elevator assignments and stepping the simulation until the next dispatch event or episode end.

#### Scenario: Step advances until next event
- **WHEN** step is called with an elevator assignment
- **THEN** it assigns the call, advances the simulation by dt, and loops until a new hall call or priority event is registered or max time is exceeded

### Requirement: Action Masking
The environment SHALL provide a boolean mask indicating which elevators are eligible for dispatch.

#### Scenario: Full elevator is masked out
- **WHEN** an elevator's current load equals or exceeds its max capacity
- **THEN** the action mask for that elevator ID is False

### Requirement: One-Hot Floor Observation Space
The environment SHALL encode each elevator's current floor as a 16-dimensional one-hot vector in the state observation, resulting in a 243-dimensional global state vector.

#### Scenario: Generate one-hot floor observation
- **WHEN** observation vector is generated
- **THEN** it contains a 16-dimensional one-hot vector for each elevator representing its current floor

