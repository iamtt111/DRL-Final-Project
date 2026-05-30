import pytest
import numpy as np
from src.envs.elevator_env import HospitalElevatorEnv
from src.agents.ppo_agent import PPOAgent

def test_ppo_agent_predict():
    env = HospitalElevatorEnv()
    env.reset(seed=42)
    
    agent = PPOAgent(env=env)
    obs = env.building.get_state_vector()
    
    # 由於沒有載入實際模型，此時會觸發 fallback 隨機可用動作選取
    action, state = agent.predict(obs, deterministic=False)
    assert action in [0, 1, 2, 3]
    assert state is None
