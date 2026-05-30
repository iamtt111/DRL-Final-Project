## ADDED Requirements

### Requirement: Nearest Car Dispatching
The rule-based agent SHALL dispatch the closest available elevator, applying compatibility penalties for incompatible directions.

#### Scenario: Closest elevator with compatible direction chosen
- **WHEN** dispatching a call at floor f in direction d
- **THEN** it chooses the elevator minimizing distance + direction compatibility penalty
