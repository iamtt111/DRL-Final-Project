# elevator-ma-env Specification

## Purpose
TBD - created by archiving change add-mappo-agent. Update Purpose after archive.
## Requirements
### Requirement: Multi-Agent Bidding Resolution
The Multi-Agent elevator environment SHALL receive continuous bids from all active elevator agents, map the call to the highest bidding eligible elevator, and advance the simulator.

#### Scenario: Step with bidding actions
- **WHEN** step is called with bidding actions for all elevators
- **THEN** it resolves the assignment to the highest bidder, steps the building simulator, and returns observation and reward dictionaries

### Requirement: Action Transition Penalty
The Multi-Agent elevator environment SHALL apply an action transition penalty of 0.5 points per agent when an agent transitions from a moving state (MOVE_UP or MOVE_DOWN) to a stopping/reversing state, reducing frequent starts/stops (NSS).

#### Scenario: Apply action transition penalty
- **WHEN** an elevator agent transitions from moving to stopping or reversing
- **THEN** a penalty of 0.5 is subtracted from the step reward

