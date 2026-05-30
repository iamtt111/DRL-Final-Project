import pytest
import numpy as np
from src.envs.elevator_env import HospitalElevatorEnv

def test_env_init():
    env = HospitalElevatorEnv()
    assert env.num_elevators == 4
    assert env.num_floors == 16
    assert env.observation_space.shape == (183,)
    assert env.action_space.n == 4

def test_env_reset_and_step():
    env = HospitalElevatorEnv()
    obs, info = env.reset(seed=42)
    assert obs.shape == (183,)
    assert isinstance(info, dict)
    assert "current_time" in info
    
    # 執行一次指派動作
    action = 0
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (183,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

def test_env_action_masking():
    env = HospitalElevatorEnv()
    obs, info = env.reset(seed=42)
    
    mask = env.action_masks()
    assert mask.shape == (4,)
    assert mask.dtype == bool
    # 初始狀態下，所有電梯應皆為可用 (True)
    assert np.all(mask)
