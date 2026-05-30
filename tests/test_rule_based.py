import pytest
import numpy as np
from src.envs.elevator_env import HospitalElevatorEnv
from src.envs.building import HallCall
from src.agents.rule_based import NearestCarAgent

def test_nearest_car_agent():
    env = HospitalElevatorEnv()
    env.reset(seed=42)
    
    # 將電梯 0 放至 2 樓
    env.building.elevators[0].current_position = env.building.floor_heights[2]
    # 將電梯 1 放至 5 樓
    env.building.elevators[1].current_position = env.building.floor_heights[5]
    
    agent = NearestCarAgent(env)
    
    # 測試在 3 樓的上行呼叫。
    # 電梯 0 距離較近 (距離為 1)，電梯 1 距離為 2。兩者均靜止，電梯 0 應被選擇。
    call = HallCall(3, 1)
    action, _ = agent.predict(None, current_call=call)
    assert action == 0
    
    # 測試在 4 樓的上行呼叫。
    # 假設電梯 0 正在下行 (direction = -1) 於 2 樓。
    # 電梯 1 於 5 樓靜止。
    # 電梯 0 的分數: 距離(2,4) = 2 + 方向相左懲罰 10 = 12。
    # 電梯 1 的分數: 距離(5,4) = 1 + 方向相容懲罰 0 = 1。
    # 電梯 1 應被選擇。
    env.building.elevators[0].current_direction = -1
    call2 = HallCall(4, 1)
    action2, _ = agent.predict(None, current_call=call2)
    assert action2 == 1
