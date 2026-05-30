## ADDED Requirements

### Requirement: Hypothesis Testing and Confidence Intervals
The metrics utility SHALL compute independent t-test p-values, 95% confidence intervals, and Cohen's d effect sizes for algorithm pairs.

#### Scenario: Verify p-value calculation
- **WHEN** comparing two lists of waits using independent t-test
- **THEN** it returns a p-value, lower and upper bounds of 95% CI, and Cohen's d value
