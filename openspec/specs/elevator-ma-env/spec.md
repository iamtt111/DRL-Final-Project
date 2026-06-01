# elevator-ma-env Specification

## Purpose
TBD - created by archiving change add-mappo-agent. Update Purpose after archive.
## Requirements
### Requirement: Multi-Agent Bidding Resolution
The Multi-Agent elevator environment SHALL receive continuous bids from all active elevator agents, map the call to the highest bidding eligible elevator, and advance the simulator.

#### Scenario: Step with bidding actions
- **WHEN** step is called with bidding actions for all elevators
- **THEN** it resolves the assignment to the highest bidder, steps the building simulator, and returns observation and reward dictionaries

