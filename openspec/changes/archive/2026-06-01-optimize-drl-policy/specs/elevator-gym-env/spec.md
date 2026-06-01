## ADDED Requirements
### Requirement: One-Hot Floor Observation Space
The environment SHALL encode each elevator's current floor as a 16-dimensional one-hot vector in the state observation, resulting in a 243-dimensional global state vector.

#### Scenario: Generate one-hot floor observation
- **WHEN** observation vector is generated
- **THEN** it contains a 16-dimensional one-hot vector for each elevator representing its current floor
