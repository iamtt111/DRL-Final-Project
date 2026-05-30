## Why
The existing bar charts show only mean values (AWT, PWT, ERT), which do not capture the distribution of passenger waiting times or the occurrence of long wait times. Since hospital elevators must prioritize emergency calls and maintain stable overall service, we need more expressive visualizations (CDF, Box Plots) to evaluate passenger wait distributions, priority system effectiveness, and algorithm variance.

## What Changes
- Add a Cumulative Distribution Function (CDF) chart for waiting times to visualize the probability of long wait times.
- Add a grouped box plot of passenger waiting times by priority level (Normal vs. Priority vs. Emergency) to evaluate the hospital dispatch prioritization performance.
- Add scenario-wise box plots showing the distribution of episode-level AWTs over the 100 benchmark episodes to evaluate algorithm variance and stability.
- Enhance the AWT vs. NSS energy trade-off scatter plot to display clear Pareto frontiers.
- Modify the evaluation pipeline to collect and save individual passenger wait times in the JSON results.

## Impact
- Affected specs: `charts-generation`
- Affected code: `src/visualization/charts.py`, `scripts/evaluate.py`, `scripts/compare_baselines.py`
