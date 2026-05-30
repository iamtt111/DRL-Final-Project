# priority-system Specification

## Purpose
TBD - created by archiving change add-phase2-integration. Update Purpose after archive.
## Requirements
### Requirement: Level 3 Emergency Preemption
The priority system SHALL allow Level 3 emergency calls to preempt existing non-emergency assignments and redistribute preempted stops.

#### Scenario: Emergency preemption triggers
- **WHEN** a Level 3 emergency is registered
- **THEN** the system finds the best candidate elevator, clears its normal stops, assigns the emergency target, and redistributes its old stops to other elevators

### Requirement: Priority Transfer Delays
The system SHALL implement priority-dependent boarding and alighting delays in the elevator physics engine to model stretcher beds, wheelchair users, and medical staff.

#### Scenario: Stretcher bed entry delay
- **WHEN** a Level 3 emergency passenger boarding or alighting triggers
- **THEN** the elevator door open duration is extended by 5.0 seconds

