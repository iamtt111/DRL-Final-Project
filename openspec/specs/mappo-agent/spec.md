# mappo-agent Specification

## Purpose
TBD - created by archiving change add-mappo-agent. Update Purpose after archive.
## Requirements
### Requirement: Parameter-Sharing Actor Inference
The MAPPO agent policy SHALL share weights across all elevator agents to map local observations to bidding actions during evaluation and training.

#### Scenario: Bidding action selection
- **WHEN** predicting bids for all elevators with their respective local observations
- **THEN** it outputs a dictionary of bidding values, allowing the environment to choose the elevator with the highest bid to assign the call

