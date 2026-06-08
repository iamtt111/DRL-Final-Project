from enum import Enum
from dataclasses import dataclass
from typing import Optional

class PassengerState(Enum):
    WAITING = "waiting"
    BOARDING = "boarding"
    IN_TRANSIT = "in_transit"
    ARRIVED = "arrived"

@dataclass
class Passenger:
    id: int                     # 唯一識別碼
    arrival_time: float         # 到達大廳呼叫的時間
    origin_floor: int           # 起始樓層
    destination_floor: int      # 目的樓層
    priority_level: int         # 0=一般, 1=輪椅, 2=醫護, 3=急診
    state: PassengerState = PassengerState.WAITING
    
    # 指標追蹤
    wait_start_time: float = 0.0      # 開始等待的時間
    board_time: Optional[float] = None # 上車時間
    arrive_time: Optional[float] = None# 到達時間
    
    def __post_init__(self):
        if self.wait_start_time == 0.0:
            self.wait_start_time = self.arrival_time

    def get_wait_duration(self, current_time: float) -> float:
        """已等待的時間 (需要傳入當前模擬時間)"""
        if self.board_time is not None:
            return self.board_time - self.wait_start_time
        return current_time - self.wait_start_time

    @property
    def direction(self) -> int:
        """移動方向: +1 上行, -1 下行"""
        return 1 if self.destination_floor > self.origin_floor else -1

    @property
    def space_occupied(self) -> int:
        """此優先權乘客所佔用的電梯空間容量 (單位)"""
        if self.priority_level == 1:       # ♿輪椅族：佔 2 人空間
            return 2
        else:                              # 👥普通 / 醫護 / 急診：佔 1 人空間
            return 1
