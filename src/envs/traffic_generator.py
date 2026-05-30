from typing import List
import numpy as np
from src.envs.passenger import Passenger

class PoissonTrafficGenerator:
    """基礎 Poisson 隨機乘客產生器"""

    def __init__(self, config: dict):
        self.num_floors = config.get("num_floors", 16)
        self.duration_minutes = config.get("duration_minutes", 10)
        self.total_passengers = config.get("total_passengers", 170)
        self.flow_distribution = config.get("flow_distribution", {
            "incoming": 0.35,
            "outgoing": 0.35,
            "interfloor": 0.30
        })

        total_seconds = self.duration_minutes * 60
        self.arrival_rate = self.total_passengers / total_seconds if total_seconds > 0 else 0.0
        self.rng = np.random.default_rng()

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def generate(self, current_time: float, dt: float) -> List[Passenger]:
        """依 Poisson 分布產生在此時間步 dt 內到達的乘客"""
        num_arrivals = self.rng.poisson(self.arrival_rate * dt)
        passengers = []

        for _ in range(num_arrivals):
            rand_val = self.rng.random()
            inc = self.flow_distribution.get("incoming", 0.35)
            out = self.flow_distribution.get("outgoing", 0.35)

            if rand_val < inc:
                # Incoming: 大廳 (0) -> 其他樓層 (1..N-1)
                origin = 0
                destination = self.rng.integers(1, self.num_floors)
            elif rand_val < inc + out:
                # Outgoing: 其他樓層 (1..N-1) -> 大廳 (0)
                origin = self.rng.integers(1, self.num_floors)
                destination = 0
            else:
                # Interfloor: 其他樓層間 (1..N-1) -> 其他樓層間 (1..N-1)
                origin = self.rng.integers(1, self.num_floors)
                destination = self.rng.integers(1, self.num_floors)
                while destination == origin:
                    destination = self.rng.integers(1, self.num_floors)

            p = Passenger(
                id=-1,  # 稍後由 HospitalTrafficGenerator 指派
                arrival_time=current_time,
                origin_floor=origin,
                destination_floor=destination,
                priority_level=0
            )
            passengers.append(p)

        return passengers


class HospitalTrafficGenerator:
    """醫院情境隨機交通流與優先權事件產生器"""

    def __init__(self, config: dict):
        self.config = config
        self.num_floors = config.get("building", {}).get("num_floors", 16)

        # 讀取基本交通流參數 (預設為 mixed_traffic)
        traffic_config = config.get("traffic", {})
        self.base_generator = PoissonTrafficGenerator({
            "num_floors": self.num_floors,
            "duration_minutes": traffic_config.get("duration_minutes", 10),
            "total_passengers": traffic_config.get("total_passengers", 100),
            "flow_distribution": traffic_config.get("flow_distribution", {
                "incoming": 0.35,
                "outgoing": 0.35,
                "interfloor": 0.30
            })
        })

        # 優先事件發生率 (每秒發生率)
        priority_config = config.get("priority_events", {})
        self.emergency_rate = priority_config.get("emergency_rate", 0.005)   # Level 3 急診
        self.staff_rate = priority_config.get("staff_rate", 0.01)           # Level 2 醫護
        self.wheelchair_rate = priority_config.get("wheelchair_rate", 0.015) # Level 1 輪椅

        self.passenger_counter = 0
        self.rng = np.random.default_rng()

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.base_generator.reset(rng)
        self.passenger_counter = 0

    def generate_arrivals(self, current_time: float, dt: float) -> List[Passenger]:
        """產生所有到達乘客 (包含一般與優先級乘客)"""
        # 1. 產生一般乘客
        passengers = self.base_generator.generate(current_time, dt)
        for p in passengers:
            p.id = self.passenger_counter
            self.passenger_counter += 1

        # 2. 隨機產生急診事件 (Level 3)
        num_emergency = self.rng.poisson(self.emergency_rate * dt)
        for _ in range(num_emergency):
            p = self._create_priority_passenger(current_time, level=3)
            passengers.append(p)

        # 3. 隨機產生醫護呼叫 (Level 2)
        num_staff = self.rng.poisson(self.staff_rate * dt)
        for _ in range(num_staff):
            p = self._create_priority_passenger(current_time, level=2)
            passengers.append(p)

        # 4. 隨機產生輪椅乘客 (Level 1)
        num_wheelchair = self.rng.poisson(self.wheelchair_rate * dt)
        for _ in range(num_wheelchair):
            p = self._create_priority_passenger(current_time, level=1)
            passengers.append(p)

        return passengers

    def _create_priority_passenger(self, current_time: float, level: int) -> Passenger:
        """建立隨機起迄樓層的優先權乘客"""
        origin = self.rng.integers(0, self.num_floors)
        destination = self.rng.integers(0, self.num_floors)
        while destination == origin:
            destination = self.rng.integers(0, self.num_floors)

        p = Passenger(
            id=self.passenger_counter,
            arrival_time=current_time,
            origin_floor=origin,
            destination_floor=destination,
            priority_level=level
        )
        self.passenger_counter += 1
        return p
