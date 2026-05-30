## ADDED Requirements

### Requirement: Decomposed Reward Logging
The logging subsystem SHALL log waiting time, energy, and emergency reward components separately to TensorBoard.

#### Scenario: Step-wise logging of components
- **WHEN** a simulation step completes
- **THEN** it records wait time, energy, emergency penalties, and bonuses to the TensorBoard logger
