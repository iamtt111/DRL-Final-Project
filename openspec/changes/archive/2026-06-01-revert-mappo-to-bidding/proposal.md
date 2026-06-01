## Why
The decentralized second-by-second motor control architecture for MAPPO resulted in severe performance collapse (AWT > 65s, high NSS jitter ~500) and abnormal behavior in visual demos. Reverting to the original, highly stable centralized bidding/dispatch architecture solves these issues, returning AWT to ~24s and aligning with the specifications.

## What Changes
- **REVERTED**: Restored `src/agents/mappo_agent.py`, `src/envs/elevator_ma_env.py`, `scripts/train_mappo.py`, `scripts/compare_baselines.py`, `scripts/demo.py`, `scripts/evaluate.py`, and `configs/train_mappo.yaml` to their bidding-based implementations from commit `0e3581d`.
- **REMOVED**: Removed the `Action Transition Penalty` from the multi-agent environment as movement is handled by the SCAN heuristic.
- **RETAINED**: Kept the new reporting protocols (console Markdown tables, timestamped charts, and automated dashboard generation).
- **RESTORED**: Restored the original bidding-based trained models (`best_model.pt`, `final_model.pt`).

## Impact
- Affected specs: `elevator-ma-env` (REMOVED Action Transition Penalty requirement)
- Affected code: MAPPO agent, environment, training, and evaluation scripts.
