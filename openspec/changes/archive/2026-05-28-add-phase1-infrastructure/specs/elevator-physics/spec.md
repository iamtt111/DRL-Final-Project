## ADDED Requirements

### Requirement: State Machine Transitions
The elevator physics model SHALL implement a state machine with states: IDLE, ACCELERATE, CRUISE, DECELERATE, and DOOR_OPEN, transitioning based on velocity and distance to target floors.

#### Scenario: Accelerating from IDLE
- **WHEN** elevator is in IDLE state and `pending_stops` is not empty
- **THEN** it transitions to ACCELERATE state and begins increasing velocity

#### Scenario: Reaching cruise speed
- **WHEN** elevator velocity reaches the rated maximum speed in ACCELERATE state
- **THEN** it transitions to CRUISE state and maintains constant velocity

#### Scenario: Initiating deceleration
- **WHEN** distance to target floor is less than or equal to the minimum deceleration distance
- **THEN** it transitions to DECELERATE state and begins decreasing velocity

#### Scenario: Arriving at floor and opening doors
- **WHEN** velocity reaches zero or the elevator reaches target floor in DECELERATE state
- **THEN** it removes the target floor from stops, snaps position to floor, and transitions to DOOR_OPEN

### Requirement: Kinematics Calculations
The elevator physics model SHALL compute position and velocity updates continuously using constant acceleration and deceleration.

#### Scenario: Distance travel time calculation
- **WHEN** calculating travel time for a distance d
- **THEN** it returns 2 * sqrt(d/a) if distance is too short to reach max speed, else 2 * (v/a) + (d - v**2/a)/v
