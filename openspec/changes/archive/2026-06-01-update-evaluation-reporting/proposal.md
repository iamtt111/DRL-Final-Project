## Why
To ensure scientific reproducibility and prevent IDE image caching issues during baseline evaluation, the user requires strict reporting protocols including console Markdown tables, timestamped visual charts, and a markdown dashboard in the docs folder.

## What Changes
- **ADDED**: Export benchmark results to a markdown report dashboard `docs/evaluation_report.md`.
- **ADDED**: Print side-by-side Markdown comparison tables (MAPPO vs SARSA vs Nearest Car) for all scenarios in console output.
- **MODIFIED**: Append timestamp (`_YYYYMMDD_HHMMSS`) to all exported charts to avoid caching issues.

## Impact
- Affected specs: `baseline-comparison`, `charts-generation`
- Affected code: `scripts/compare_baselines.py`, `src/visualization/charts.py`
