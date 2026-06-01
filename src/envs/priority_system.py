from src.envs.building import Building
from src.envs.elevator import Elevator

class PrioritySystem:
    """處理三級優先權事件與急診搶佔機制"""

    def __init__(self, config: dict):
        self.config = config

    def check_and_apply_preemption(self, building: Building, emergency_floor: int) -> bool:
        """
        當新的 Level 3 急診乘客產生時，尋找最優電梯進行搶佔指派，並重新分配受影響的外呼任務
        """
        # 1. 尋找候選電梯 (排除故障及正在處理急診的電梯)
        candidates = []
        for elev in building.elevators:
            if elev.is_out_of_service:
                continue
            if elev.emergency_target is not None:
                continue
            candidates.append(elev)

        if not candidates:
            return False

        # 2. 計算每台候選電梯到急診樓層的估計到達時間 (ETA)
        target_height = building.floor_heights[emergency_floor]
        best_elev = None
        min_eta = float('inf')

        for elev in candidates:
            eta = self._estimate_travel_time(elev, target_height)
            if eta < min_eta:
                min_eta = eta
                best_elev = elev

        if best_elev is None:
            return False

        # 3. 搶佔與重新分配
        # 取得目前車廂內乘客的目的地 (內部呼叫，不可清除)
        internal_destinations = {p.destination_floor for p in best_elev.passengers}
        
        # 找出需要清除的大廳外呼停靠站 (即非內呼的停靠站)
        hall_stops_to_redistribute = [f for f in best_elev.pending_stops if f not in internal_destinations]
        
        # 重新設定電梯的停靠站：只保留車廂內乘客的目的地，並加入急診目標
        best_elev.clear_stops()
        for f in internal_destinations:
            best_elev.assign_hall_call(f, 0)
        
        best_elev.assign_emergency(emergency_floor)

        # 4. 移除硬編碼重新分配邏輯：
        # 被搶佔的大廳外呼任務將不直接指派給其他電梯。
        # 它們會在接下來的模擬更新中，由環境的 _update_pending_assignments 自動掃描並重新加入動態競標佇列中。
        return True

    def _estimate_travel_time(self, elev: Elevator, target_height: float) -> float:
        """估計電梯到達目標高度的時間 (考慮當前速度)"""
        dist = abs(target_height - elev.current_position)
        v = elev.current_velocity
        a = elev.acceleration
        v_max = elev.rated_speed

        t_acc = (v_max - v) / a if v < v_max else 0.0
        d_acc = 0.5 * (v + v_max) * t_acc
        
        if dist <= d_acc:
            return 2 * (dist / a) ** 0.5 if a > 0 else 0.0
        else:
            d_cruise = dist - d_acc
            t_cruise = d_cruise / v_max
            return t_acc + t_cruise
