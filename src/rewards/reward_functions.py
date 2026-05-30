import numpy as np
from typing import List
from src.envs.building import Building
from src.envs.elevator import Elevator, ElevatorState
from src.envs.passenger import Passenger, PassengerState

def calculate_reward(
    building: Building,
    events: list,
    weights: dict = None,
    thresholds: dict = None,
    gamma_weights: dict = None
) -> float:
    """
    計算多目標步長獎勵 (Step Reward)
    Rt = -(w1 * T_wait + w2 * E_energy + w3 * P_emergency) + R_bonus
    """
    if weights is None:
        weights = {"wait": 0.3, "energy": 0.1, "emergency": 0.6}
    if thresholds is None:
        thresholds = {3: 30.0, 2: 60.0, 1: 45.0}  # Level 3: 30s, Level 2: 60s, Level 1: 45s
    if gamma_weights is None:
        gamma_weights = {3: 5.0, 2: 2.0, 1: 1.0}

    current_time = building.current_time

    # ==========================================
    # 1. 等待時間懲罰 (T_wait)
    # ==========================================
    active_passengers = []
    for floor in building.floors:
        active_passengers.extend([p for p in floor.waiting_queue if p.state == PassengerState.WAITING])
    
    t_wait = 0.0
    if len(active_passengers) > 0:
        total_wait_ratio = sum(p.get_wait_duration(current_time) / 120.0 for p in active_passengers)
        t_wait = total_wait_ratio / len(active_passengers)

    # ==========================================
    # 2. 能耗懲罰 (E_energy)
    # ==========================================
    e_energy = 0.0
    n_e = len(building.elevators)
    if n_e > 0:
        total_energy_penalty = 0.0
        for elev in building.elevators:
            elev_penalty = 0.0
            # (a) 空載移動懲罰
            is_moving = elev.state in (ElevatorState.ACCELERATE, ElevatorState.CRUISE, ElevatorState.DECELERATE)
            if is_moving and elev.current_load == 0:
                elev_penalty += 0.1
            
            # (b) 無效停靠懲罰 (開門時沒有任何人上下車)
            # 我們透過檢查這次 update 發出的事件來判定
            # 如果發生了 ELEVATOR_ARRIVED 但沒有對應的 PASSENGER_BOARDED 或 PASSENGER_DELIVERED
            # 這裡簡化為：如果處於 DOOR_OPEN 狀態，且最近開門時沒有需要此樓層服務的乘客
            # 我們在事件處理中計算：當開門且無上下客時給予懲罰。
            # 或者是更簡單的：如果有 ELEVATOR_ARRIVED 事件且該電梯當前樓層無人要下車，且該樓層無同向乘客要上車。
            total_energy_penalty += elev_penalty
        
        # 檢查是否有無效停靠事件
        for event in events:
            if event.type.value == "elevator_arrived":
                elev_id = event.data["elevator_id"]
                floor_idx = event.data["floor"]
                elev = building.elevators[elev_id]
                
                # 檢查是否有乘客要下車
                has_deboard = any(p.destination_floor == floor_idx for p in elev.passengers)
                # 檢查該樓層是否有等待的乘客
                floor_queue = building.floors[floor_idx].waiting_queue
                has_board = any(p.state == PassengerState.WAITING for p in floor_queue)
                
                if not has_deboard and not has_board:
                    total_energy_penalty += 0.3  # 無效停靠懲罰

        e_energy = total_energy_penalty / n_e

    # ==========================================
    # 3. 緊急等待懲罰 (P_emergency - 非線性遞增)
    # ==========================================
    p_emergency = 0.0
    for p in active_passengers:
        if p.priority_level > 0:
            p_level = p.priority_level
            t_thresh = thresholds.get(p_level, 60.0)
            gamma = gamma_weights.get(p_level, 1.0)
            wait_time = p.get_wait_duration(current_time)
            
            # (t_wait / t_threshold)^2
            p_emergency += gamma * ((wait_time / t_thresh) ** 2)

    # ==========================================
    # 4. 獎金項 (R_bonus)
    # ==========================================
    r_bonus = 0.0
    # (a) 優先乘客在閾值內開始被服務 (上車)
    for event in events:
        if event.type.value == "passenger_boarded":
            p_level = event.data["priority"]
            if p_level > 0:
                wait_time = event.data["wait_time"]
                t_thresh = thresholds.get(p_level, 60.0)
                if wait_time <= t_thresh:
                    r_bonus += 2.0  # 閾值內服務獎勵

    # (b) 負載均衡獎勵
    if n_e > 1:
        loads = [elev.current_load for elev in building.elevators]
        if sum(loads) > 0:
            load_std = np.std(loads)
            if load_std < 1.5:
                r_bonus += 0.5

    # 綜合計算總獎勵
    penalty = (weights["wait"] * t_wait +
               weights["energy"] * e_energy +
               weights["emergency"] * p_emergency)
    
    components = {
        "wait_time_penalty": float(weights["wait"] * t_wait),
        "energy_penalty": float(weights["energy"] * e_energy),
        "emergency_penalty": float(weights["emergency"] * p_emergency),
        "bonus": float(r_bonus)
    }
    
    return -penalty + r_bonus, components
