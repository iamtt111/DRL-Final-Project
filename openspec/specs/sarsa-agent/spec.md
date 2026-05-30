# sarsa-agent Specification

## Purpose
TBD - created by archiving change add-phase3-training. Update Purpose after archive.
## Requirements
### Requirement: Linear Tile Coding for High Dimensions
The SARSA(λ) baseline agent SHALL implement hash-based Tile Coding to approximate action-values linearly in the 183-dimensional state space.

#### Scenario: Linear approximation updating
- **WHEN** receiving step transition (s, a, r, s', a')
- **THEN** it updates the weight vector using eligibility traces and temporal difference error

