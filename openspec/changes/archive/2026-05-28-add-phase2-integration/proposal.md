## Why
Integrate the Phase 1 simulator modules into a standard Gymnasium environment interface (`elevator_env.py`) supporting action masking (for `MaskablePPO`), implement the hospital priority and preemption logic (`priority_system.py`), build a baseline rule-based dispatcher agent (`rule_based.py`), and construct the Pygame visualization skeleton (`pygame_renderer.py`). This bridges our simulator infrastructure to reinforcement learning and visual evaluation.

## What Changes
- Create `src/envs/elevator_env.py` wrapping the simulation into `gymnasium.Env` and supporting action masking.
- Create `src/envs/priority_system.py` defining the priority levels and Level 3 preemption task redistribution.
- Create `src/agents/rule_based.py` implementing the Nearest Car dispatching logic.
- Create `src/visualization/pygame_renderer.py` defining the rendering viewport, panels, and layouts.
- Implement reward calculations in `src/rewards/reward_functions.py` based on waiting times, energy, and emergency penalties.

## Impact
- Affected specs: `elevator-gym-env`, `priority-system`, `rule-based-agent`, `pygame-visualization` (all new capabilities)
- Affected code: New env wrappers and visualization structures.
