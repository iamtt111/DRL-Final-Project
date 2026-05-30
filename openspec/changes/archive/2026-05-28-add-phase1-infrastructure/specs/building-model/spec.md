## ADDED Requirements

### Requirement: Floor Queues and Elevator Coordination
The building model SHALL manage floors, floor-level waiting queues for passengers, and route hall call requests.

#### Scenario: Passenger arrival registering call
- **WHEN** passenger arrives at a floor with a destination floor
- **THEN** the passenger is added to the floor queue and a hall call is registered in that direction

#### Scenario: Dispatch allocation to elevator
- **WHEN** a hall call is assigned to an elevator
- **THEN** the elevator adds the floor to its pending_stops
