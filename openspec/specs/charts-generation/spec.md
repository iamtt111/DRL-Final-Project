# charts-generation Specification

## Purpose
TBD - created by archiving change add-phase4-evaluation-demo. Update Purpose after archive.
## Requirements
### Requirement: Comparative Bar Charts
The chart generator SHALL plot wait times and response times as bar charts and save them to docs/images/.

#### Scenario: Export figures to docs
- **WHEN** plotting results
- **THEN** it exports PNG files comparing AWT, PWT, and ERT into the docs/images directory

### Requirement: Cumulative Distribution Function Plot
The chart generator SHALL plot the cumulative probability distribution of passenger waiting times for all algorithms in each scenario.

#### Scenario: Export CDF plots
- **WHEN** plotting results
- **THEN** it exports a CDF plot comparison_cdf.png comparing the waiting times of all algorithms in the target scenario

### Requirement: Priority Waiting Box Plot
The chart generator SHALL plot passenger waiting times grouped by priority level as box plots.

#### Scenario: Export priority box plots
- **WHEN** plotting results
- **THEN** it exports a grouped box plot comparison_priority_boxplot.png illustrating wait times across normal, priority, and emergency passengers

### Requirement: Scenario Distribution Box Plot
The chart generator SHALL plot the distribution of episode-level AWTs over the evaluation runs as box plots.

#### Scenario: Export scenario box plots
- **WHEN** plotting results
- **THEN** it exports a box plot comparison_scenario_boxplot.png showing the median, quartiles, and outliers of episode-level AWTs

