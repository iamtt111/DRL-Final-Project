## ADDED Requirements

### Requirement: Hyperparameter Configurations and Checkpoints
The training pipeline SHALL load configurations from YAML, execute policy updates, periodically run evaluation episodes, and save the best model checkpoint.

#### Scenario: Save best model on evaluation improvement
- **WHEN** periodic evaluation achieves a higher mean reward than the previous best
- **THEN** it saves the model weights to models/ppo/best_model.zip
