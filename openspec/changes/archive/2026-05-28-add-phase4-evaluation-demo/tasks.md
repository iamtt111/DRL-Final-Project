## 1. Metrics & Statistical Verification
- [x] 1.1 Implement statistical metrics in `src/utils/metrics.py` (t-test, 95% CI, and Cohen's d)

## 2. Evaluation & Benchmarking Pipelines
- [x] 2.1 Implement `scripts/evaluate.py` to run evaluation loops on different scenarios
- [x] 2.2 Implement `scripts/compare_baselines.py` to benchmark PPO, SARSA(λ), and Nearest Car over 100 episodes and output raw metrics

## 3. Visualization & Demo
- [x] 3.1 Implement `src/visualization/charts.py` generating comparison plots and saving them to `docs/images/`
- [x] 3.2 Implement `scripts/demo.py` driving real-time Pygame rendering with chosen agents and demonstrating preemption

## 4. Verification & Testing
- [x] 4.1 Create `tests/test_metrics.py` verifying t-test and Cohen's d calculations
- [x] 4.2 Create `tests/test_compare.py` verifying baseline comparison loading and runs
- [x] 4.3 Verify with pytest
