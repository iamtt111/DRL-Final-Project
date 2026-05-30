# evaluation-loop Specification

## Purpose
TBD - created by archiving change add-phase4-evaluation-demo. Update Purpose after archive.
## Requirements
### Requirement: Multi-Scenario Evaluation Loop
The evaluation script SHALL load a trained model, run simulation runs on configured scenarios, and log metrics.

#### Scenario: Running evaluation on mixed traffic
- **WHEN** evaluating with mixed traffic scenario configuration
- **THEN** it executes the episode, records wait times, and outputs average metrics

