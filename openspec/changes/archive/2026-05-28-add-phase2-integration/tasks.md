## 1. Environment & Reward Integration
- [x] 1.1 Implement `src/envs/elevator_env.py` wrapping building physics and implementing `reset()`, `step()`, and `action_masks()`
- [x] 1.2 Implement `src/rewards/reward_functions.py` computing standard step-wise shaping rewards and terminal metrics

## 2. Priority Preemption & Baseline Agent
- [x] 2.1 Implement `src/envs/priority_system.py` containing priority handlers and Level 3 preemption task redistribution
- [x] 2.2 Implement `src/agents/rule_based.py` implementing the Nearest Car dispatching baseline agent

## 3. Presentation Layer
- [x] 3.1 Implement `src/visualization/pygame_renderer.py` scaffolding the viewport, building, shafts, queues, and statistics panel

## 4. Verification & Testing
- [x] 4.1 Create `tests/test_env.py` verifying Gymnasium API compliance (using `check_env`) and action masking
- [x] 4.2 Create `tests/test_priority.py` verifying Level 3 emergency preemption and task redistribution
- [x] 4.3 Create `tests/test_rule_based.py` verifying the Nearest Car agent's decision logic
- [x] 4.4 Verify with pytest
