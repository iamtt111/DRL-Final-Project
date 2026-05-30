# priority-system Specification

## Purpose
TBD - created by archiving change add-phase2-integration. Update Purpose after archive.
## Requirements
### Requirement: Level 3 Emergency Preemption
The priority system SHALL allow Level 3 emergency calls to preempt existing non-emergency assignments and redistribute preempted stops.

#### Scenario: Emergency preemption triggers
- **WHEN** a Level 3 emergency is registered
- **THEN** the system finds the best candidate elevator, clears its normal stops, assigns the emergency target, and redistributes its old stops to other elevators

