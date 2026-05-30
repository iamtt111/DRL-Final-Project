## Context
We need to compare the performance of our trained MaskablePPO agent, the SARSA(λ) baseline, and the Nearest Car baseline using a rigorous evaluation framework. We must compile KPIs (AWT, PWT, ERT, ECR, ENI, LBI), perform statistical significance testing, generate clean graphs, and build an interactive visualization demo to display emergency preemption.

## Goals / Non-Goals
- Goals:
  - Implement a standardized comparison script executing 100 episodes per scenario for each algorithm.
  - Implement independent t-test, 95% confidence intervals, and Cohen's d effect size calculations using `scipy.stats`.
  - Output comparative bar charts and reward convergence curves directly to `docs/images/`.
  - Connect any of the agents to the Pygame renderer.
- Non-Goals:
  - Training logic modifications (this was covered in Phase 3).

## Decisions
- **Statistical Testing**:
  - We use `scipy.stats.ttest_ind` to perform independent two-sample t-tests between the MaskablePPO results and the baselines.
  - Cohen's d is calculated as: $d = \frac{\bar{x}_1 - \bar{x}_2}{s_p}$, where $s_p = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}}$ is the pooled standard deviation.
- **Chart Directory**: Figures will be output directly to `docs/images/` as PNG files. If the directory does not exist, the visualization module will create it.

## Risks / Trade-offs
- **Execution Time for Comparison**: Running 100 episodes for three algorithms and multiple scenarios can take several minutes.
  - *Mitigation*: The comparison script will run the episodes in fast non-rendering mode (disabling the Pygame renderer) to ensure benchmarking completes rapidly.

## Open Questions
- None.
