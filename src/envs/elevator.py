from enum import Enum
from typing import List, Optional
from dataclasses import dataclass, field
from src.envs.passenger import Passenger
from src.envs.event import Event, EventType

class ElevatorState(Enum):
    IDLE = "idle"
    ACCELERATE = "accelerate"
    CRUISE = "cruise"
    DECELERATE = "decelerate"
    DOOR_OPEN = "door_open"

class Elevator:
    """單台電梯的實體與運動狀態機模型"""

    def __init__(
        self,
        elevator_id: int,
        floor_heights: List[float],
        max_capacity: int = 12,
        rated_speed: float = 2.5,
        acceleration: float = 1.0,
        door_open_time: float = 1.0,
        door_close_time: float = 1.0,
        boarding_time_per_person: float = 1.5,
        door_extension_wheelchair: float = 3.0,
    ):
        self.elevator_id = elevator_id
        self.floor_heights = floor_heights
        self.max_capacity = max_capacity
        self.rated_speed = rated_speed
        self.acceleration = acceleration
        self.door_open_time = door_open_time
        self.door_close_time = door_close_time
        self.boarding_time_per_person = boarding_time_per_person
        self.door_extension_wheelchair = door_extension_wheelchair

        # 物理狀態
        self.current_position: float = 0.0  # 高度 (公尺)
        self.current_velocity: float = 0.0  # 速度 (m/s)
        self.current_direction: int = 0     # -1 = 下行, 0 = 靜止, 1 = 上行

        # 系統狀態
        self._state: ElevatorState = ElevatorState.IDLE
        self._pending_stops: List[int] = []
        self.passengers: List[Passenger] = []
        self.door_timer: float = 0.0
        self.time_since_idle: float = 0.0
        self.is_out_of_service: bool = False
        self.emergency_target: Optional[int] = None

        # 內部追蹤用時間
        self.current_time: float = 0.0

    @property
    def current_floor(self) -> int:
        """回傳目前最接近的樓層索引"""
        return min(
            range(len(self.floor_heights)),
            key=lambda f: abs(self.floor_heights[f] - self.current_position)
        )

    @property
    def direction(self) -> int:
        return self.current_direction

    @property
    def current_load(self) -> int:
        return sum(p.space_occupied for p in self.passengers)

    @property
    def load_ratio(self) -> float:
        return self.current_load / self.max_capacity

    @property
    def is_door_open(self) -> bool:
        return self._state == ElevatorState.DOOR_OPEN

    @property
    def is_idle(self) -> bool:
        return self._state == ElevatorState.IDLE

    @property
    def pending_stops(self) -> List[int]:
        return self._pending_stops

    @property
    def state(self) -> ElevatorState:
        return self._state

    def assign_hall_call(self, floor: int, direction: int) -> None:
        """指派大廳呼叫"""
        if floor not in self._pending_stops:
            self._pending_stops.append(floor)

    def assign_emergency(self, floor: int) -> None:
        """指派緊急任務 (Level 3 急診)，設定直達目標"""
        self.emergency_target = floor
        if floor not in self._pending_stops:
            self._pending_stops.append(floor)
        # 如果門開著，強制縮短開門時間，使電梯能立即關門出發執行緊急任務
        if self._state == ElevatorState.DOOR_OPEN:
            self.door_timer = min(self.door_timer, self.door_close_time)

    def clear_stops(self) -> List[int]:
        """清除所有停靠站並回傳"""
        old_stops = self._pending_stops.copy()
        self._pending_stops.clear()
        self.emergency_target = None
        return old_stops

    def _get_next_target_floor(self) -> Optional[int]:
        """依 SCAN 策略或緊急直達目標決定下一個停靠樓層"""
        if self.emergency_target is not None:
            return self.emergency_target

        if not self._pending_stops:
            return None

        current_y = self.current_position

        # 靜止狀態：前往距離最近的停靠站
        if self.current_direction == 0:
            return min(self._pending_stops, key=lambda f: abs(self.floor_heights[f] - current_y))

        # 上行狀態
        if self.current_direction == 1:
            # 尋找前方或同樓層的停靠站
            floors_ahead = [f for f in self._pending_stops if self.floor_heights[f] >= current_y - 1e-5]
            if floors_ahead:
                return min(floors_ahead, key=lambda f: self.floor_heights[f])
            # 若前方無停靠站，則尋找後方停靠站（會觸發轉向）
            floors_behind = [f for f in self._pending_stops if self.floor_heights[f] < current_y]
            if floors_behind:
                return max(floors_behind, key=lambda f: self.floor_heights[f])

        # 下行狀態
        if self.current_direction == -1:
            # 尋找前方或同樓層的停靠站
            floors_ahead = [f for f in self._pending_stops if self.floor_heights[f] <= current_y + 1e-5]
            if floors_ahead:
                return max(floors_ahead, key=lambda f: self.floor_heights[f])
            # 若前方無停靠站，則尋找後方停靠站（會觸發轉向）
            floors_behind = [f for f in self._pending_stops if self.floor_heights[f] > current_y]
            if floors_behind:
                return min(floors_behind, key=lambda f: self.floor_heights[f])

        return None

    def open_doors(self, base_time: float) -> None:
        """手動開啟電梯門 (設定計時器)"""
        self._state = ElevatorState.DOOR_OPEN
        self.door_timer = base_time

    def update(self, dt: float) -> List[Event]:
        """推進電梯物理與狀態機一個時間步長 dt (通常為 1.0 秒)"""
        self.current_time += dt
        events = []
        is_opposite = False

        # 1. 如果處於 IDLE 狀態，先嘗試切換至移動或開門狀態，避免延遲一幀才開始動作
        if self._state == ElevatorState.IDLE:
            next_target = self._get_next_target_floor()
            if next_target is not None:
                self.time_since_idle = 0.0
                target_y = self.floor_heights[next_target]
                if target_y > self.current_position:
                    self.current_direction = 1
                    self._state = ElevatorState.ACCELERATE
                elif target_y < self.current_position:
                    self.current_direction = -1
                    self._state = ElevatorState.ACCELERATE
                else:
                    # 已經在該樓層，直接開門
                    self._state = ElevatorState.DOOR_OPEN
                    self.door_timer = self.door_open_time + self.door_close_time
                    if next_target in self._pending_stops:
                        self._pending_stops.remove(next_target)
                    events.append(Event(EventType.ELEVATOR_ARRIVED, self.current_time, {
                        "elevator_id": self.elevator_id,
                        "floor": next_target
                    }))

        # 2. 依最新狀態執行物理或時間更新
        if self._state == ElevatorState.DOOR_OPEN:
            self.door_timer -= dt
            self.current_velocity = 0.0
            if self.door_timer <= 1e-5:
                self.door_timer = 0.0
                if self.emergency_target == self.current_floor:
                    self.emergency_target = None
                # 關門完成，確認下一個目標
                next_target = self._get_next_target_floor()
                if next_target is not None:
                    target_y = self.floor_heights[next_target]
                    if target_y > self.current_position:
                        self.current_direction = 1
                        self._state = ElevatorState.ACCELERATE
                    elif target_y < self.current_position:
                        self.current_direction = -1
                        self._state = ElevatorState.ACCELERATE
                    else:
                        # 已經在目標樓層 (例如在關門前又有同樓層呼叫)
                        if self.current_direction != 0:
                            # 如果電梯有建立方向，不要原地重複開門，避免因反向呼叫或滿載導致無限開關門
                            # 我們直接清除該停靠站，並尋找下一個非本樓層的目標
                            if next_target in self._pending_stops:
                                self._pending_stops.remove(next_target)
                            
                            # 重新取得下一個目標
                            next_target = self._get_next_target_floor()
                            if next_target is not None:
                                target_y = self.floor_heights[next_target]
                                if target_y > self.current_position:
                                    self.current_direction = 1
                                    self._state = ElevatorState.ACCELERATE
                                elif target_y < self.current_position:
                                    self.current_direction = -1
                                    self._state = ElevatorState.ACCELERATE
                                else:
                                    self.current_direction = 0
                                    self._state = ElevatorState.IDLE
                                    events.append(Event(EventType.ELEVATOR_IDLE, self.current_time, {"elevator_id": self.elevator_id}))
                            else:
                                self.current_direction = 0
                                self._state = ElevatorState.IDLE
                                events.append(Event(EventType.ELEVATOR_IDLE, self.current_time, {"elevator_id": self.elevator_id}))
                        else:
                            # 如果電梯是靜止/無方向的，可以原地重新開門
                            self._state = ElevatorState.DOOR_OPEN
                            self.door_timer = self.door_open_time + self.door_close_time
                            if next_target in self._pending_stops:
                                self._pending_stops.remove(next_target)
                else:
                    self.current_direction = 0
                    self._state = ElevatorState.IDLE
                    events.append(Event(EventType.ELEVATOR_IDLE, self.current_time, {"elevator_id": self.elevator_id}))

        elif self._state == ElevatorState.IDLE:
            self.current_velocity = 0.0
            self.time_since_idle += dt

        else:
            # 移動狀態: ACCELERATE, CRUISE, DECELERATE
            next_target = self._get_next_target_floor()
            
            # 如果目標樓層就是當前樓層，且電梯有建立行進方向，且電梯剛開始移動 (例如剛關門就被重複指派)
            # 我們直接清除該停靠站，不原地重複開門
            if self.current_position == self.floor_heights[next_target] and self.current_direction != 0:
                if next_target in self._pending_stops:
                    self._pending_stops.remove(next_target)
                if self.emergency_target == next_target:
                    self.emergency_target = None
                next_target = self._get_next_target_floor()

            if next_target is None:
                # 若無目標，減速至停止
                self._state = ElevatorState.DECELERATE
                target_y = self.current_position
            else:
                target_y = self.floor_heights[next_target]

            dist = abs(target_y - self.current_position)
            
            # 檢查目標是否在目前行進方向的反方向 (例如行進中被強行搶佔目標)
            if self.current_direction == 1 and target_y < self.current_position:
                is_opposite = True
            elif self.current_direction == -1 and target_y > self.current_position:
                is_opposite = True

            # 煞車距離計算: d = v^2 / 2a
            decel_dist = (self.current_velocity ** 2) / (2 * self.acceleration)

            # 狀態轉移判定
            if is_opposite:
                self._state = ElevatorState.DECELERATE
            elif self._state in (ElevatorState.ACCELERATE, ElevatorState.CRUISE) and dist <= decel_dist + 1e-5:
                self._state = ElevatorState.DECELERATE

            # 計算此步的運動學更新
            v_prev = self.current_velocity
            if self._state == ElevatorState.ACCELERATE:
                self.current_velocity = min(self.current_velocity + self.acceleration * dt, self.rated_speed)
                t_accel = (self.current_velocity - v_prev) / self.acceleration
                t_cruise = dt - t_accel
                step_move = 0.5 * (v_prev + self.current_velocity) * t_accel + self.current_velocity * t_cruise
                if self.current_velocity >= self.rated_speed:
                    self._state = ElevatorState.CRUISE

            elif self._state == ElevatorState.CRUISE:
                step_move = self.rated_speed * dt

            else:  # DECELERATE
                self.current_velocity = max(self.current_velocity - self.acceleration * dt, 0.0)
                t_decel = (v_prev - self.current_velocity) / self.acceleration
                step_move = 0.5 * (v_prev + self.current_velocity) * t_decel

            # 更新位置與防過衝判定
            if next_target is not None and not is_opposite and step_move >= dist:
                # 在此步驟內抵達目標樓層
                self.current_position = target_y
                self.current_velocity = 0.0
                self._state = ElevatorState.DOOR_OPEN
                self.door_timer = self.door_open_time + self.door_close_time
                if next_target in self._pending_stops:
                    self._pending_stops.remove(next_target)
                events.append(Event(EventType.ELEVATOR_ARRIVED, self.current_time, {
                    "elevator_id": self.elevator_id,
                    "floor": next_target
                }))
            else:
                self.current_position += self.current_direction * step_move
                self.current_position = max(0.0, min(self.current_position, self.floor_heights[-1]))
                
                if self.current_velocity == 0.0 and self._state == ElevatorState.DECELERATE:
                    if is_opposite or next_target is None:
                        # 若無目標或因反向目標而減速至靜止，不進行傳送，而是原地轉為 IDLE 狀態
                        self._state = ElevatorState.IDLE
                        self.current_direction = 0
                        events.append(Event(EventType.ELEVATOR_IDLE, self.current_time, {"elevator_id": self.elevator_id}))
                    else:
                        # 正常減速到 0 仍未到，強行到達並對齊
                        self.current_position = target_y
                        self._state = ElevatorState.DOOR_OPEN
                        self.door_timer = self.door_open_time + self.door_close_time
                        if next_target in self._pending_stops:
                            self._pending_stops.remove(next_target)
                        events.append(Event(EventType.ELEVATOR_ARRIVED, self.current_time, {
                            "elevator_id": self.elevator_id,
                            "floor": next_target
                        }))

        return events

    def manual_update(self, action: int, dt: float) -> float:
        """
        手動更新電梯物理狀態與運動學 (Motor Control)
        0: STOP/IDLE, 1: MOVE_UP, 2: MOVE_DOWN, 3: OPEN_DOOR
        """
        penalty = 0.0
        self.current_time += dt

        # 狀態與動作的約束處理
        if action == 3:  # OPEN_DOOR
            if self.current_velocity > 0.0:
                # 行進中開門 - 無效且危險
                penalty += -2.0
                # 視同 STOP，減速
                v_prev = self.current_velocity
                self.current_velocity = max(0.0, self.current_velocity - self.acceleration * dt)
                self.current_position += self.current_direction * 0.5 * (v_prev + self.current_velocity) * dt
                self._state = ElevatorState.DECELERATE
            else:
                # 靜止，檢查是否對齊樓層
                nearest_floor = min(range(len(self.floor_heights)), key=lambda f: abs(self.floor_heights[f] - self.current_position))
                if abs(self.floor_heights[nearest_floor] - self.current_position) > 0.1:
                    # 未對齊樓層開門 - 無效
                    penalty += -2.0
                    self._state = ElevatorState.IDLE
                    self.current_direction = 0
                else:
                    self.current_position = self.floor_heights[nearest_floor]
                    self._state = ElevatorState.DOOR_OPEN
                    self.door_timer = self.door_open_time + self.door_close_time
        else:
            # 關閉門 (如果原本是開門狀態，現在選擇了移動或停止，門立即關閉)
            if self._state == ElevatorState.DOOR_OPEN:
                self._state = ElevatorState.IDLE
                self.current_direction = 0
                self.door_timer = 0.0

            if action == 0:  # STOP/IDLE
                if self.current_velocity > 0.0:
                    v_prev = self.current_velocity
                    self.current_velocity = max(0.0, self.current_velocity - self.acceleration * dt)
                    self.current_position += self.current_direction * 0.5 * (v_prev + self.current_velocity) * dt
                    self._state = ElevatorState.DECELERATE
                else:
                    self._state = ElevatorState.IDLE
                    self.current_direction = 0
                    self.time_since_idle += dt

            elif action == 1:  # MOVE_UP
                if self.current_position >= self.floor_heights[-1]:
                    penalty += -2.0
                    self.current_velocity = 0.0
                    self.current_direction = 0
                    self._state = ElevatorState.IDLE
                else:
                    # 如果原本下行，需先減速到 0
                    if self.current_direction == -1 and self.current_velocity > 0.0:
                        v_prev = self.current_velocity
                        self.current_velocity = max(0.0, self.current_velocity - self.acceleration * dt)
                        self.current_position += self.current_direction * 0.5 * (v_prev + self.current_velocity) * dt
                        self._state = ElevatorState.DECELERATE
                        if self.current_velocity == 0.0:
                            self.current_direction = 0
                    else:
                        self.current_direction = 1
                        v_prev = self.current_velocity
                        self.current_velocity = min(self.rated_speed, self.current_velocity + self.acceleration * dt)
                        self.current_position += self.current_direction * 0.5 * (v_prev + self.current_velocity) * dt
                        self._state = ElevatorState.ACCELERATE if self.current_velocity < self.rated_speed else ElevatorState.CRUISE

            elif action == 2:  # MOVE_DOWN
                if self.current_position <= 0.0:
                    penalty += -2.0
                    self.current_velocity = 0.0
                    self.current_direction = 0
                    self._state = ElevatorState.IDLE
                else:
                    # 如果原本上行，需先減速到 0
                    if self.current_direction == 1 and self.current_velocity > 0.0:
                        v_prev = self.current_velocity
                        self.current_velocity = max(0.0, self.current_velocity - self.acceleration * dt)
                        self.current_position += self.current_direction * 0.5 * (v_prev + self.current_velocity) * dt
                        self._state = ElevatorState.DECELERATE
                        if self.current_velocity == 0.0:
                            self.current_direction = 0
                    else:
                        self.current_direction = -1
                        v_prev = self.current_velocity
                        self.current_velocity = min(self.rated_speed, self.current_velocity + self.acceleration * dt)
                        self.current_position += self.current_direction * 0.5 * (v_prev + self.current_velocity) * dt
                        self._state = ElevatorState.ACCELERATE if self.current_velocity < self.rated_speed else ElevatorState.CRUISE

            # 減速到 0 後自動對齊最近樓層
            if self.current_velocity == 0.0 and self._state != ElevatorState.DOOR_OPEN:
                nearest_floor = min(range(len(self.floor_heights)), key=lambda f: abs(self.floor_heights[f] - self.current_position))
                self.current_position = self.floor_heights[nearest_floor]
                self._state = ElevatorState.IDLE
                self.current_direction = 0

        # 防過衝邊界保護
        self.current_position = max(0.0, min(self.current_position, self.floor_heights[-1]))
        return penalty
