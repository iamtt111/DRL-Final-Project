## Why
Implement the final evaluation and visualization pipeline (Phase 4) including the multi-scenario evaluation script, comparison engine for benchmarking the three algorithms, statistical significance testing utility, automatic chart exporter, and visual interactive demo runner. This completes the project deliverables and scientifically validates the reinforcement learning agent's performance.

## What Changes
- Create `scripts/evaluate.py` to evaluate single model policies on different traffic configurations.
- Create `scripts/compare_baselines.py` to run at least 100 episodes per scenario for MaskablePPO, SARSA(λ), and Nearest Car.
- Create `src/utils/metrics.py` containing statistical tests: t-test p-value, 95% confidence interval, and Cohen's d effect size calculations using SciPy.
- Create `src/visualization/charts.py` to automatically output comparison graphs and save them as PNGs under `docs/images/`.
- Create `scripts/demo.py` linking any agent to our Pygame renderer to show live runs and emergency preemption.

## Impact
- Affected specs: `evaluation-loop`, `baseline-comparison`, `statistical-tests`, `charts-generation`, `interactive-demo` (all new capabilities)
- Affected code: New evaluation, comparison, plotting, and demo runner scripts.
