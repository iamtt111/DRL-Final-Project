## ADDED Requirements

### Requirement: Multi-Scenario Evaluation Loop
The evaluation script SHALL load a trained model, run simulation runs on configured scenarios, and log metrics.

#### Scenario: Running evaluation on mixed traffic
- **WHEN** evaluating with mixed traffic scenario configuration
- **THEN** it executes the episode, records wait times, and outputs average metrics
