# passenger-model Specification

## Purpose
TBD - created by archiving change add-phase1-infrastructure. Update Purpose after archive.
## Requirements
### Requirement: Passenger Lifecycle and State
The passenger model SHALL track waiting, boarding, transit, and arrival states, logging key times for performance metrics.

#### Scenario: Passenger boarding elevator
- **WHEN** passenger boards an elevator
- **THEN** passenger state transitions to IN_TRANSIT and board_time is recorded

#### Scenario: Passenger arriving at destination
- **WHEN** passenger exits the elevator at destination floor
- **THEN** passenger state transitions to ARRIVED and arrive_time is recorded

