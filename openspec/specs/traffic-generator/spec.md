# traffic-generator Specification

## Purpose
TBD - created by archiving change add-phase1-infrastructure. Update Purpose after archive.
## Requirements
### Requirement: Poisson Traffic Generation
The traffic generator SHALL generate passenger arrivals following a Poisson process with configuration-driven rates and direction patterns.

#### Scenario: Morning peak traffic distribution
- **WHEN** generating morning peak traffic
- **THEN** 74% of arrivals are incoming (lobby to higher floors), 11% are outgoing (higher floors to lobby), and 15% are interfloor

### Requirement: Hospital Priority Event Injection
The traffic generator SHALL inject high-priority events (Emergency Bed, Medical Staff, Wheelchair Users) according to independent Poisson rates.

#### Scenario: Emergency event injection
- **WHEN** an emergency event triggers
- **THEN** a Level 3 priority passenger is created and added to the building

### Requirement: Scenario Driven Priority Rates
The traffic generator SHALL override default priority event rates with scenario-specific priority configuration rates when loading a scenario.

#### Scenario: Extreme disaster priority rate override
- **WHEN** loading the disaster crisis scenario
- **THEN** the emergency arrival rate is overridden and set to 0.03 events per second

