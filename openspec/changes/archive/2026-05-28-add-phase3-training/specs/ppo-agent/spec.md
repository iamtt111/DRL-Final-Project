## ADDED Requirements

### Requirement: PPO Action Masking Integration
The PPO agent wrapper SHALL load `MaskablePPO` and integrate with the environment's action masks during training and inference.

#### Scenario: Inference with action mask
- **WHEN** predicting action from observation with an action mask
- **THEN** it chooses an action that is marked True in the mask
