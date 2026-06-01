import numpy as np
from typing import Tuple, Optional

class NearestCarAgent:
    """
    Nearest Car (最近電梯優先) 規則式調度代理人
    作為系統效能評估的 Lower-bound Baseline
    """

    def __init__(self, env = None):
        self.env = env

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
        current_call = None
    ) -> Tuple[int, Optional[np.ndarray]]:
        """
        根據當前待處理呼叫，選擇距離最近且方向相容的電梯
        符合 Stable-Baselines3 Agent 預測介面
        """
        # 1. 取得當前指派的大廳外呼信號
        if current_call is None and self.env is not None and self.env.pending_assignments:
            current_call = self.env.pending_assignments[0]

        if current_call is None:
            # 若無待指派呼叫，預設指派給電梯 0
            return 0, None

        best_elevator_id = 0
        best_score = float('inf')

        # 3. 最近車輛調度邏輯
        elevators = self.env.building.elevators if self.env is not None else []
        for elev in elevators:
            e_id = elev.elevator_id
            
            # 檢查物理狀態過濾不可指派電梯：滿載或停用/故障
            if elev.current_load >= elev.max_capacity:
                continue
            if elev.is_out_of_service:
                continue

            distance = abs(elev.current_floor - current_call.floor)
            
            # 方向相容加權
            direction_penalty = 0
            if elev.current_direction != 0 and elev.current_direction != current_call.direction:
                direction_penalty = 10  # 方向不同加罰 10 樓的距離成本

            score = distance + direction_penalty
            if score < best_score:
                best_score = score
                best_elevator_id = e_id

        # 確保在極端情況下 (例如全部電梯均滿載或停用)，仍指派預設電梯 0
        if best_score == float('inf'):
            best_elevator_id = 0

        return best_elevator_id, None
