## ADDED Requirements
### Requirement: Priority Transfer Delays
The system SHALL implement priority-dependent boarding and alighting delays in the elevator physics engine to model stretcher beds, wheelchair users, and medical staff.

#### Scenario: Stretcher bed entry delay
- **WHEN** a Level 3 emergency passenger boarding or alighting triggers
- **THEN** the elevator door open duration is extended by 5.0 seconds
