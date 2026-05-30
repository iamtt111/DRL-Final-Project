# mappo-agent Specification

## Purpose
The MAPPO agent capability wraps the multi-agent cooperative policy network using Centralized Training and Decentralized Execution.

## ADDED Requirements
### Requirement: Parameter-Sharing Actor Inference
The MAPPO agent policy SHALL share weights across all elevator agents to map local observations to bidding actions during evaluation and training.

#### Scenario: Bidding action selection
- **WHEN** predicting bids for all elevators with their respective local observations
- **THEN** it outputs a dictionary of bidding values, allowing the environment to choose the elevator with the highest bid to assign the call
