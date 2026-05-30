# baseline-comparison Specification

## Purpose
TBD - created by archiving change add-phase4-evaluation-demo. Update Purpose after archive.
## Requirements
### Requirement: Benchmarking Multiple Algorithms
The comparison engine SHALL benchmark MaskablePPO, SARSA(λ), and Nearest Car over a configured number of episodes (default 100) per scenario, tracking AWT, PWT, ERT, ECR, ENI, and LBI.

#### Scenario: Collect comparative KPIs
- **WHEN** benchmark is run
- **THEN** it generates raw metrics files for all three algorithms under each traffic scenario

