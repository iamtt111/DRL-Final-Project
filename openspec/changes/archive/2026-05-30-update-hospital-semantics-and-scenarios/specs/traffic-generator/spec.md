## ADDED Requirements
### Requirement: Scenario Driven Priority Rates
The traffic generator SHALL override default priority event rates with scenario-specific priority configuration rates when loading a scenario.

#### Scenario: Extreme disaster priority rate override
- **WHEN** loading the disaster crisis scenario
- **THEN** the emergency arrival rate is overridden and set to 0.03 events per second
