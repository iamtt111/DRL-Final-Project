from typing import List, Optional
import numpy as np
from src.envs.passenger import Passenger, PassengerState
from src.envs.elevator import Elevator, ElevatorState
from src.envs.event import Event, EventType

class Floor:
    """單個樓層乘客佇列"""
    def __init__(self, floor_index: int):
        self.floor_index = floor_index
        self.waiting_queue: List[Passenger] = []

class HallCall:
    """大廳外呼信號"""
    def __init__(self, floor: int, direction: int):
        self.floor = floor
        self.direction = direction  # 1 = 上行, -1 = 下行

    def __eq__(self, other):
        if not isinstance(other, HallCall):
            return False
        return self.floor == other.floor and self.direction == other.direction

class Building:
    """建築物整體模型，管理所有電梯與各樓層狀態"""

    def __init__(self, config: dict):
        self.config = config
        
        # 支援傳入完整的 config 字典或僅傳入 building 子字典
        self.building_config = config.get("building", config) if isinstance(config, dict) else config
        self.elevator_config = config.get("elevator", config) if isinstance(config, dict) else config
        self.priority_config = config.get("priority", config) if isinstance(config, dict) else config
        
        self.num_floors = self.building_config.get("num_floors", 16)
        self.floor_height_lobby = self.building_config.get("floor_height_lobby", 4.0)
        self.floor_height_normal = self.building_config.get("floor_height_normal", 3.0)
        self.floor_heights = self._calculate_floor_heights()
        self.max_height = self.floor_heights[-1]

        # 初始化樓層佇列
        self.floors = [Floor(i) for i in range(self.num_floors)]
        
        # 初始化電梯 (稍後在 reset 中被載入或直接初始化)
        self.elevators: List[Elevator] = []
        
        # 全域模擬時間
        self.current_time: float = 0.0
        
        # 追蹤最近的乘客抵達（用於計算流量強度）
        self.arrival_history: List[float] = []

    def _calculate_floor_heights(self) -> List[float]:
        """計算各樓層高度"""
        heights = [0.0]
        if self.num_floors > 1:
            heights.append(self.floor_height_lobby)
        for f in range(2, self.num_floors):
            heights.append(self.floor_height_lobby + (f - 1) * self.floor_height_normal)
        return heights

    def reset(self, rng: np.random.Generator) -> None:
        """重置建築物狀態"""
        self.current_time = 0.0
        self.arrival_history.clear()
        self.floors = [Floor(i) for i in range(self.num_floors)]
        
        # 根據 config 初始化電梯數量
        num_elevators = self.elevator_config.get("num_elevators", 4)
        max_capacity = self.elevator_config.get("max_capacity", 12)
        rated_speed = self.elevator_config.get("rated_speed", 2.5)
        acceleration = self.elevator_config.get("acceleration", 1.0)
        door_open_time = self.elevator_config.get("door_open_time", 1.0)
        door_close_time = self.elevator_config.get("door_close_time", 1.0)
        boarding_time_per_person = self.elevator_config.get("boarding_time_per_person", 1.5)
        door_extension_wheelchair = self.priority_config.get("door_extension_wheelchair", 3.0)

        self.elevators = []
        for i in range(num_elevators):
            elev = Elevator(
                elevator_id=i,
                floor_heights=self.floor_heights,
                max_capacity=max_capacity,
                rated_speed=rated_speed,
                acceleration=acceleration,
                door_open_time=door_open_time,
                door_close_time=door_close_time,
                boarding_time_per_person=boarding_time_per_person,
                door_extension_wheelchair=door_extension_wheelchair
            )
            # 隨機初始化電梯位置 (樓層高度)
            if rng is not None:
                start_floor = rng.integers(0, self.num_floors)
            else:
                start_floor = 0
            elev.current_position = self.floor_heights[start_floor]
            self.elevators.append(elev)

    def add_passenger(self, passenger: Passenger) -> None:
        """新增乘客至出發樓層的佇列"""
        floor_idx = passenger.origin_floor
        passenger.state = PassengerState.WAITING
        passenger.wait_start_time = self.current_time
        self.floors[floor_idx].waiting_queue.append(passenger)
        # 依優先權等級降序排序，使高優先權乘客（如急診）排在隊列最前列以優先登機
        self.floors[floor_idx].waiting_queue.sort(key=lambda p: p.priority_level, reverse=True)
        self.arrival_history.append(self.current_time)

    def get_pending_hall_calls(self) -> List[HallCall]:
        """獲取當前所有未完成的大廳呼叫信號"""
        calls = []
        for floor in self.floors:
            has_up = any(p.direction == 1 for p in floor.waiting_queue if p.state == PassengerState.WAITING)
            has_down = any(p.direction == -1 for p in floor.waiting_queue if p.state == PassengerState.WAITING)
            if has_up:
                calls.append(HallCall(floor.floor_index, 1))
            if has_down:
                calls.append(HallCall(floor.floor_index, -1))
        return calls

    def update(self, dt: float) -> List[Event]:
        """推進建築物與所有電梯狀態 dt"""
        # 0. 如果有空閒的電梯停在某樓層，且該樓層有等待中的乘客，直接指派給該電梯原地開門服務
        for elev in self.elevators:
            if elev.state == ElevatorState.IDLE and elev.current_velocity == 0.0:
                current_floor_idx = elev.current_floor
                floor = self.floors[current_floor_idx]
                if any(p.state == PassengerState.WAITING for p in floor.waiting_queue):
                    elev.assign_hall_call(current_floor_idx, 0)

        self.current_time += dt
        
        # 清理過舊的到達歷史紀錄 (僅保留最近 60 秒的紀錄)
        self.arrival_history = [t for t in self.arrival_history if self.current_time - t <= 60.0]

        all_events = []

        # 1. 推進所有電梯物理狀態
        for elev in self.elevators:
            elev_events = elev.update(dt)
            all_events.extend(elev_events)

        # 2. 處理電梯開門時的乘客進出與等待計時
        for elev in self.elevators:
            if elev.state == ElevatorState.DOOR_OPEN:
                current_floor_idx = elev.current_floor
                floor = self.floors[current_floor_idx]

                # (a) 乘客下電梯 (Deboarding)
                deboarding_passengers = [p for p in elev.passengers if p.destination_floor == current_floor_idx]
                for p in deboarding_passengers:
                    p.state = PassengerState.ARRIVED
                    p.arrive_time = self.current_time
                    elev.passengers.remove(p)
                    
                    # 乘客下車時間延遲 (Hospital Semantics)
                    if p.priority_level == 3:       # 急診病床
                        delay = 5.0
                    elif p.priority_level == 2:     # 醫護 / VIP
                        delay = 2.5
                    elif p.priority_level == 1:     # 輪椅族
                        delay = elev.boarding_time_per_person + elev.door_extension_wheelchair
                    else:                           # 一般乘客
                        delay = elev.boarding_time_per_person
                    elev.door_timer = min(elev.door_timer + delay, 15.0)

                    all_events.append(Event(EventType.PASSENGER_DELIVERED, self.current_time, {
                        "passenger_id": p.id,
                        "elevator_id": elev.elevator_id,
                        "floor": current_floor_idx,
                        "wait_time": p.get_wait_duration(self.current_time),
                        "transit_time": p.arrive_time - p.board_time,
                        "priority": p.priority_level
                    }))

                # 如果電梯為空，清除方向使其可服務任何方向的大廳外呼
                if len(elev.passengers) == 0:
                    elev.current_direction = 0

                # (b) 決定電梯目前要服務的方向 (如果電梯原本是靜止的)
                if elev.current_direction == 0 and floor.waiting_queue:
                    # 採用第一個等待乘客的方向
                    first_waiting = floor.waiting_queue[0]
                    elev.current_direction = first_waiting.direction

                # (c) 乘客上電梯 (Boarding)
                boarding_passengers = []
                for p in list(floor.waiting_queue):
                    if elev.current_load + p.space_occupied > elev.max_capacity:
                        break
                    
                    # 如果電梯有優先搶佔目標，非該目標樓層或非 Level 3 緊急乘客禁止上車
                    if elev.emergency_target is not None:
                        if elev.emergency_target != current_floor_idx or p.priority_level != 3:
                            continue

                    # 方向一致才能上車，或者電梯為空且方向為 0，或者這是我們專門搶佔來接送的急診乘客
                    is_emergency_boarding = (p.priority_level == 3 and elev.emergency_target == current_floor_idx)
                    if elev.current_direction == 0 or p.direction == elev.current_direction or is_emergency_boarding:
                        p.state = PassengerState.IN_TRANSIT
                        p.board_time = self.current_time
                        elev.passengers.append(p)
                        floor.waiting_queue.remove(p)
                        boarding_passengers.append(p)

                        # 如果是我們搶佔來接送的急診乘客，上車後將直達目的地設為新的緊急搶佔目標
                        if is_emergency_boarding:
                            elev.emergency_target = p.destination_floor

                        # 乘客上車時間延遲 (Hospital Semantics)
                        if p.priority_level == 3:       # 急診病床
                            delay = 5.0
                        elif p.priority_level == 2:     # 醫護 / VIP
                            delay = 2.5
                        elif p.priority_level == 1:     # 輪椅族
                            delay = elev.boarding_time_per_person + elev.door_extension_wheelchair
                        else:                           # 一般乘客
                            delay = elev.boarding_time_per_person
                        elev.door_timer = min(elev.door_timer + delay, 15.0)

                        # 將目的地加入電梯的停靠站中
                        if p.destination_floor not in elev.pending_stops:
                            elev.pending_stops.append(p.destination_floor)

                        all_events.append(Event(EventType.PASSENGER_BOARDED, self.current_time, {
                            "passenger_id": p.id,
                            "elevator_id": elev.elevator_id,
                            "floor": current_floor_idx,
                            "wait_time": p.get_wait_duration(self.current_time),
                            "priority": p.priority_level
                        }))

                # (d) 檢查是否完全清空該方向呼叫，若是則發出信號已完成
                if boarding_passengers:
                    served_dir = elev.current_direction
                    # 檢查此樓層同方向是否仍有乘客在等
                    still_waiting = any(p.direction == served_dir for p in floor.waiting_queue)
                    if not still_waiting:
                        all_events.append(Event(EventType.HALL_CALL_SERVED, self.current_time, {
                            "floor": current_floor_idx,
                            "direction": served_dir,
                            "elevator_id": elev.elevator_id
                        }))

        return all_events

    def get_state_vector(self, max_time: float = 600.0) -> np.ndarray:
        """
        生成扁平化狀態向量 (dim=243 for Ne=4, Nf=16, using one-hot floor representation)
        """
        state_parts = []

        # 1. 電梯狀態子向量 (per elevator)
        for elev in self.elevators:
            # position: 16-dimensional one-hot representation of current floor
            floor_one_hot = [0.0] * self.num_floors
            floor_one_hot[elev.current_floor] = 1.0
            state_parts.extend(floor_one_hot)
            
            # direction: {-1, 0, 1}
            state_parts.append(float(elev.current_direction))
            
            # load_ratio: [0, 1]
            state_parts.append(elev.load_ratio)
            
            # door_state: closed = 0, open = 1
            door_state = 1.0 if elev.state == ElevatorState.DOOR_OPEN else 0.0
            state_parts.append(door_state)
            
            # internal_calls: binary multi-hot of size num_floors
            int_calls = [0.0] * self.num_floors
            for p in elev.passengers:
                int_calls[p.destination_floor] = 1.0
            state_parts.extend(int_calls)
            
            # time_since_idle: normalized (divided by 120s max)
            idle_norm = min(elev.time_since_idle / 120.0, 1.0)
            state_parts.append(idle_norm)

        # 2. 大廳呼叫子向量
        hall_up = [0.0] * self.num_floors
        hall_down = [0.0] * self.num_floors
        wait_up = [0.0] * self.num_floors
        wait_down = [0.0] * self.num_floors

        for floor in self.floors:
            f_idx = floor.floor_index
            waiting_up = [p for p in floor.waiting_queue if p.direction == 1 and p.state == PassengerState.WAITING]
            waiting_down = [p for p in floor.waiting_queue if p.direction == -1 and p.state == PassengerState.WAITING]

            if waiting_up:
                hall_up[f_idx] = 1.0
                longest_wait = max(p.get_wait_duration(self.current_time) for p in waiting_up)
                wait_up[f_idx] = min(longest_wait / 120.0, 1.0)

            if waiting_down:
                hall_down[f_idx] = 1.0
                longest_wait = max(p.get_wait_duration(self.current_time) for p in waiting_down)
                wait_down[f_idx] = min(longest_wait / 120.0, 1.0)

        state_parts.extend(hall_up)
        state_parts.extend(hall_down)
        state_parts.extend(wait_up)
        state_parts.extend(wait_down)

        # 3. 優先權子向量
        priority_req = [0.0] * self.num_floors
        priority_wait = [0.0] * self.num_floors

        for floor in self.floors:
            f_idx = floor.floor_index
            waiting_priority = [p for p in floor.waiting_queue if p.priority_level > 0 and p.state == PassengerState.WAITING]
            if waiting_priority:
                # 取得最高優先級並正規化 (1=輪椅, 2=醫護, 3=急診 -> 1/3, 2/3, 1.0)
                max_level = max(p.priority_level for p in waiting_priority)
                priority_req[f_idx] = max_level / 3.0
                
                longest_priority_wait = max(p.get_wait_duration(self.current_time) for p in waiting_priority)
                priority_wait[f_idx] = min(longest_priority_wait / 120.0, 1.0)

        state_parts.extend(priority_req)
        state_parts.extend(priority_wait)

        # 4. 全域特徵
        # time_of_day
        state_parts.append(min(self.current_time / max_time, 1.0))
        
        # traffic_intensity: 最近一分鐘抵達總數的移動平均 (除以 10.0 正規化)
        intensity = len(self.arrival_history) / 10.0
        state_parts.append(min(intensity, 1.0))
        
        # active_emergency_count: 當前在等待或運送中的急診人數 (Level 3)
        active_emergency = 0
        for floor in self.floors:
            active_emergency += sum(1 for p in floor.waiting_queue if p.priority_level == 3)
        for elev in self.elevators:
            active_emergency += sum(1 for p in elev.passengers if p.priority_level == 3)
            
        state_parts.append(min(active_emergency / 5.0, 1.0))

        return np.array(state_parts, dtype=np.float32)
