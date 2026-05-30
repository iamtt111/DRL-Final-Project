import pytest
from src.envs.elevator import Elevator, ElevatorState
from src.envs.event import EventType

def test_elevator_init():
    heights = [0.0, 4.0, 7.0, 10.0]
    elev = Elevator(elevator_id=0, floor_heights=heights, max_capacity=5)
    assert elev.elevator_id == 0
    assert elev.floor_heights == heights
    assert elev.max_capacity == 5
    assert elev.current_position == 0.0
    assert elev.current_velocity == 0.0
    assert elev.current_direction == 0
    assert elev.state == ElevatorState.IDLE

def test_elevator_state_transitions_and_movement():
    heights = [0.0, 4.0, 7.0, 10.0]
    elev = Elevator(
        elevator_id=0,
        floor_heights=heights,
        max_capacity=5,
        rated_speed=2.0,
        acceleration=1.0,
        door_open_time=1.0,
        door_close_time=1.0
    )
    
    # 指派目標樓層 2 (高度 7.0m)
    elev.assign_hall_call(2, 1)
    assert 2 in elev.pending_stops
    
    # 更新 1.0 秒：此時應該開始加速
    events = elev.update(1.0)
    assert elev.state == ElevatorState.ACCELERATE
    assert elev.current_velocity == 1.0
    # 平均速度為 0.5 m/s，位置應該在 0.5m
    assert abs(elev.current_position - 0.5) < 1e-3
    
    # 再更新 1.0 秒：速度達到 2.0 (額定速度)，狀態轉為 CRUISE
    events = elev.update(1.0)
    assert elev.state == ElevatorState.CRUISE
    assert elev.current_velocity == 2.0
    
    # 再更新 1.0 秒：
    # 目前位置在 0.5 + 0.5*(1+2)*1 = 2.0m。
    # 距離目標 7.0m 還有 5.0m。
    # 2.0 m/s 煞車距離為 2^2 / (2*1) = 2.0m。
    # 5.0m > 2.0m，所以繼續以 2.0m/s 巡航。
    events = elev.update(1.0)
    assert elev.state == ElevatorState.CRUISE
    assert elev.current_velocity == 2.0
    assert abs(elev.current_position - 4.0) < 1e-3
    
    # 再更新 1.0 秒：
    # 距離目標剩下 3.0m。仍然大於煞車距離 2.0m，所以會繼續巡航。
    events = elev.update(1.0)
    assert abs(elev.current_position - 6.0) < 1e-3
    
    # 距離目標剩下 1.0m。小於等於煞車距離 2.0m，進入減速狀態，並且在此秒內會到達目標。
    events = elev.update(1.0)
    assert elev.state == ElevatorState.DOOR_OPEN
    assert elev.current_position == 7.0
    assert elev.current_velocity == 0.0
    assert len(events) == 1
    assert events[0].type == EventType.ELEVATOR_ARRIVED
    assert events[0].data["floor"] == 2

def test_elevator_emergency_direct_run():
    heights = [0.0, 4.0, 7.0, 10.0]
    elev = Elevator(elevator_id=0, floor_heights=heights)
    elev.assign_hall_call(1, 1)
    elev.assign_emergency(3)  # 緊急指派到 3 樓
    
    # 下一個目標應該直接是 3 樓，略過 1 樓
    assert elev._get_next_target_floor() == 3


def test_elevator_door_open_loop_prevention():
    heights = [0.0, 4.0, 7.0, 10.0]
    elev = Elevator(
        elevator_id=0,
        floor_heights=heights,
        max_capacity=5,
        door_open_time=1.0,
        door_close_time=1.0
    )
    
    # 已經在 1 樓 (高度 4.0m)，有 2 樓 (上行) 和 1 樓 (被指派的下行) 停靠站
    elev.current_position = 4.0
    elev.current_direction = 1  # 上行方向已建立
    elev._state = ElevatorState.DOOR_OPEN
    elev.door_timer = 0.5  # 門正在關閉
    
    # 設定待停靠站
    elev.assign_hall_call(2, 1)
    elev.assign_hall_call(1, -1) # 在當前樓層的呼叫
    
    # 更新 0.5 秒，門計時器到期 (doors should close)
    elev.update(0.5)
    
    # 驗證電梯沒有重開門，而是關門後往 2 樓加速
    assert elev.state == ElevatorState.ACCELERATE
    assert elev.current_direction == 1
    # 當前樓層 1 已經被清除
    assert 1 not in elev.pending_stops
    assert 2 in elev.pending_stops


def test_elevator_moving_loop_prevention():
    heights = [0.0, 4.0, 7.0, 10.0]
    elev = Elevator(
        elevator_id=0,
        floor_heights=heights,
        max_capacity=5,
        door_open_time=1.0,
        door_close_time=1.0
    )
    
    # 電梯在 1 樓 (高度 4.0m)，狀態為 ACCELERATE，方向為 1 (UP)
    # 這是剛關門出發的狀態
    elev.current_position = 4.0
    elev.current_direction = 1
    elev._state = ElevatorState.ACCELERATE
    
    # 此時代理人重新指派了 1 樓的呼叫 (例如 1 樓下行)，並且有 2 樓停靠站
    elev.assign_hall_call(2, 1)
    elev.assign_hall_call(1, -1)
    
    # 更新，它應該過濾掉當前樓層 1，不原地重開門，而是繼續前往 2 樓
    elev.update(0.5)
    assert elev.state == ElevatorState.ACCELERATE
    assert elev.current_direction == 1
    assert 1 not in elev.pending_stops
    assert 2 in elev.pending_stops


