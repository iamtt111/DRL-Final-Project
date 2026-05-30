import pytest
import numpy as np
from src.envs.building import Building
from src.envs.priority_system import PrioritySystem
from src.envs.elevator import ElevatorState


def test_priority_preemption():
    config = {
        "building": {
            "num_floors": 10,
            "floor_height_lobby": 4.0,
            "floor_height_normal": 3.0,
            "num_elevators": 2,
            "max_capacity": 5,
            "rated_speed": 2.0,
            "acceleration": 1.0,
            "door_open_time": 1.0,
            "door_close_time": 1.0,
            "boarding_time_per_person": 1.0,
            "door_extension_wheelchair": 3.0
        }
    }
    
    b = Building(config["building"])
    b.reset(None)
    
    # 電梯 0 在 0 樓，有 1 樓和 2 樓的呼叫停靠站
    elev0 = b.elevators[0]
    elev0.current_position = 0.0
    elev0.assign_hall_call(1, 1)
    elev0.assign_hall_call(2, 1)
    
    # 電梯 1 在 8 樓
    elev1 = b.elevators[1]
    elev1.current_position = b.floor_heights[8]
    
    # 建立搶佔系統
    ps = PrioritySystem(config)
    
    # 3 樓發生急診，電梯 0 距離 3 樓較近，應該被搶佔
    success = ps.check_and_apply_preemption(b, 3)
    assert success
    
    # 電梯 0 的緊急目標應設定為 3
    assert elev0.emergency_target == 3
    
    # 電梯 0 原本的外呼停靠站 (1 和 2) 應被重新分配給電梯 1
    assert 1 in elev1.pending_stops
    assert 2 in elev1.pending_stops


def test_moving_preemption():
    config = {
        "building": {
            "num_floors": 16,
            "floor_height_lobby": 4.0,
            "floor_height_normal": 3.0,
            "num_elevators": 1,
            "max_capacity": 5,
            "rated_speed": 2.5,
            "acceleration": 1.0,
            "door_open_time": 1.0,
            "door_close_time": 1.0,
            "boarding_time_per_person": 1.0,
            "door_extension_wheelchair": 3.0
        }
    }
    
    b = Building(config["building"])
    b.reset(None)
    
    elev = b.elevators[0]
    # 在 10 樓 (高度 31.0) 向下移動中
    elev.current_position = b.floor_heights[10]
    elev.current_direction = -1
    elev._state = ElevatorState.ACCELERATE
    elev.current_velocity = 2.0
    elev.assign_hall_call(2, -1)
    
    ps = PrioritySystem(config)
    # F12 發生急診
    success = ps.check_and_apply_preemption(b, 12)
    assert success
    assert elev.emergency_target == 12
    
    # 執行更新，確認電梯會減速並轉向
    # 第 1 步：減速
    b.update(1.0)
    assert elev.state == ElevatorState.DECELERATE
    assert elev.current_direction == -1
    
    # 第 2 步：減速到 0 原地變為 IDLE
    b.update(1.0)
    assert elev.state == ElevatorState.IDLE
    assert elev.current_direction == 0
    
    # 第 3 步：變為上行 ACCELERATE
    b.update(1.0)
    assert elev.state == ElevatorState.ACCELERATE
    assert elev.current_direction == 1


def test_emergency_preemption_door_close():
    config = {
        "building": {
            "num_floors": 10,
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
    }
    
    b = Building(config["building"])
    b.reset(None)
    
    elev = b.elevators[0]
    # 電梯門正開著，且門計時器很大 (例如 15.0 秒)
    elev._state = ElevatorState.DOOR_OPEN
    elev.door_timer = 15.0
    
    # 進行搶佔分配緊急呼叫到 2 樓
    ps = PrioritySystem(config)
    success = ps.check_and_apply_preemption(b, 2)
    assert success
    assert elev.emergency_target == 2
    
    # 驗證門計時器被強制縮短為 door_close_time (1.0s)
    assert elev.door_timer == 1.0


def test_emergency_passenger_direct_routing_and_boarding():
    config = {
        "building": {
            "num_floors": 10,
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
    }
    
    b = Building(config["building"])
    b.reset(None)
    
    elev = b.elevators[0]
    # 電梯在 0 樓且門開著，裡面有 1 個去 5 樓的乘客 (方向 UP=1)
    from src.envs.passenger import Passenger, PassengerState
    elev.passengers.append(Passenger(id=10, arrival_time=0.0, origin_floor=0, destination_floor=5, priority_level=0))
    elev._state = ElevatorState.DOOR_OPEN
    elev.door_timer = 5.0
    elev.current_direction = 1
    elev.assign_hall_call(5, 0)
    
    # 在 2 樓 (高度 7.0m) 發生急診乘客要去 0 樓 (方向 DOWN=-1)
    p_emergency = Passenger(id=99, arrival_time=0.0, origin_floor=2, destination_floor=0, priority_level=3)
    b.add_passenger(p_emergency)
    
    ps = PrioritySystem(config)
    success = ps.check_and_apply_preemption(b, 2)
    assert success
    assert elev.emergency_target == 2
    
    # 驗證一般乘客在電梯前往急診期間不能上車
    # 在 0 樓有一個普通等待乘客
    p_normal = Passenger(id=11, arrival_time=0.0, origin_floor=0, destination_floor=3, priority_level=0)
    b.add_passenger(p_normal)
    
    # 更新，0 樓乘客不能上車，門強制縮短並關閉
    b.update(1.0)
    assert p_normal in b.floors[0].waiting_queue # 沒有上車
    assert elev.state == ElevatorState.ACCELERATE
    assert elev.current_direction == 1 # 往 2 樓前進
    
    # 前往 2 樓 (高度 7.0m)。0 樓到 2 樓高度為 7.0m。
    # 執行更新直到抵達 2 樓
    for _ in range(10):
        b.update(1.0)
        if elev.state == ElevatorState.DOOR_OPEN:
            break
            
    assert elev.current_floor == 2
    assert elev.state == ElevatorState.DOOR_OPEN
    
    # 在 2 樓，急診乘客應該能上車 (即便電梯原本方向是 1 且急診乘客去 0 樓是 -1)
    # 更新 boarding
    b.update(1.0) # 觸發進出
    assert p_emergency.state == PassengerState.IN_TRANSIT
    
    # 急診乘客上車後，電梯緊急目標應更新為急診患者目的地 0 樓
    assert elev.emergency_target == 0
    
    # 更新直到門關閉並開始移動
    for _ in range(10):
        b.update(1.0)
        if elev.state != ElevatorState.DOOR_OPEN:
            break
            
    # 更新直到抵達 0 樓 (目的地，門再次打開)
    for _ in range(10):
        b.update(1.0)
        if elev.state == ElevatorState.DOOR_OPEN:
            break
            
    assert elev.current_floor == 0
    assert p_emergency.state == PassengerState.ARRIVED
    
    # 抵達目的地關門後，緊急目標應被清除
    for _ in range(10):
        b.update(1.0)
        if elev.state != ElevatorState.DOOR_OPEN:
            break
    assert elev.emergency_target is None


