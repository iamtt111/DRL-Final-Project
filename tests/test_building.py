import pytest
import numpy as np
from src.envs.building import Building, HallCall
from src.envs.passenger import Passenger, PassengerState

def test_building_init_and_reset():
    config = {
        "num_floors": 16,
        "floor_height_lobby": 4.0,
        "floor_height_normal": 3.0,
        "num_elevators": 4,
        "max_capacity": 12,
        "rated_speed": 2.5,
        "acceleration": 1.0,
        "door_open_time": 1.0,
        "door_close_time": 1.0,
        "boarding_time_per_person": 1.5,
        "door_extension_wheelchair": 3.0
    }
    
    b = Building(config)
    assert b.num_floors == 16
    # lobby (4.0m) + 14 * normal (3.0m) = 46.0m
    assert b.max_height == 46.0
    
    rng = np.random.default_rng(42)
    b.reset(rng)
    assert len(b.elevators) == 4
    assert len(b.floors) == 16

def test_building_boarding_deboarding():
    config = {
        "num_floors": 4,
        "floor_height_lobby": 4.0,
        "floor_height_normal": 3.0,
        "num_elevators": 1,
        "max_capacity": 5,
        "rated_speed": 2.0,
        "acceleration": 1.0,
        "door_open_time": 1.0,
        "door_close_time": 1.0,
        "boarding_time_per_person": 1.0,
        "door_extension_wheelchair": 3.0
    }
    
    b = Building(config)
    b.reset(None)
    
    # 強制將電梯放置在 0 樓
    elev = b.elevators[0]
    elev.current_position = 0.0
    
    # 新增一位 0 樓要到 2 樓的乘客
    p = Passenger(id=1, arrival_time=0.0, origin_floor=0, destination_floor=2, priority_level=0)
    b.add_passenger(p)
    
    # 確認大廳呼叫有被註冊
    calls = b.get_pending_hall_calls()
    assert HallCall(0, 1) in calls
    
    # 指派任務給該電梯
    elev.assign_hall_call(0, 1)
    
    # 更新建築物
    events = b.update(1.0)
    # 電梯已在 0 樓並收到 0 樓呼叫，應該開門
    assert elev.is_door_open
    # 乘客應該已經上車
    assert p.state == PassengerState.IN_TRANSIT
    assert p in elev.passengers
    
    # 目的樓層 2 應該被加入電梯停靠站
    assert 2 in elev.pending_stops

def test_state_vector():
    config = {
        "num_floors": 16,
        "floor_height_lobby": 4.0,
        "floor_height_normal": 3.0,
        "num_elevators": 4,
        "max_capacity": 12,
        "rated_speed": 2.5,
        "acceleration": 1.0,
        "door_open_time": 1.0,
        "door_close_time": 1.0,
        "boarding_time_per_person": 1.5,
        "door_extension_wheelchair": 3.0
    }
    b = Building(config)
    b.reset(None)
    
    vec = b.get_state_vector()
    assert vec.shape == (183,)
