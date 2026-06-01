## ADDED Requirements
### Requirement: Action Transition Penalty
The Multi-Agent elevator environment SHALL apply an action transition penalty of 0.5 points per agent when an agent transitions from a moving state (MOVE_UP or MOVE_DOWN) to a stopping/reversing state, reducing frequent starts/stops (NSS).

#### Scenario: Apply action transition penalty
- **WHEN** an elevator agent transitions from moving to stopping or reversing
- **THEN** a penalty of 0.5 is subtracted from the step reward
