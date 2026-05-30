import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Tuple, Optional
from src.envs.building import Building, HallCall
from src.envs.traffic_generator import HospitalTrafficGenerator
from src.envs.priority_system import PrioritySystem
from src.rewards.reward_functions import calculate_reward
from src.utils.config_loader import load_config

class HospitalElevatorEnv(gym.Env):
    """
    智慧醫院電梯群控與優先調度 Gymnasium 環境 (符合 Gymnasium API)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: dict = None, render_mode: str = None):
        super().__init__()
        self.config = config or load_config()
        self.render_mode = render_mode

        # 基礎組態讀取
        building_config = self.config.get("building", {})
        self.num_floors = building_config.get("num_floors", 16)
        
        elevator_config = self.config.get("elevator", {})
        self.num_elevators = elevator_config.get("num_elevators", 4)
        
        # 建立建築物與交通產生器
        # 我們將全域組態傳給它們
        self.building = Building(self.config)
        self.traffic_gen = HospitalTrafficGenerator(self.config)
        self.priority_system = PrioritySystem(self.config)

        # 模擬參數
        self.dt = 1.0  # 模擬步長 (秒)
        self.max_time = 600.0  # 每個 Episode 的模擬長度為 10 分鐘 (600秒)

        # Gymnasium 空間定義
        # 183 維的扁平化狀態向量
        state_dim = self.num_elevators * (self.num_floors + 5) + 4 * self.num_floors + 2 * self.num_floors + 3
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.num_elevators)

        # 當前等待指派的大廳呼叫佇列 (事件驅動派梯的核心)
        self.pending_assignments: List[HallCall] = []

        # 隨機數產生器
        self.rng = np.random.default_rng()

        # 視覺化渲染器 (Phase 2 Scaffolding)
        self.renderer = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """重置環境"""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        # 重置子系統
        self.building.reset(self.rng)
        self.traffic_gen.reset(self.rng)
        self.pending_assignments.clear()

        # 推進模擬直到至少有一個大廳呼叫等待指派，或者 Episode 結束
        terminated = False
        step_events = []

        while not self.pending_assignments and not terminated:
            # 獲取當前模擬時間
            current_sim_time = self.building.current_time + self.dt
            
            # 生成新乘客
            new_passengers = self.traffic_gen.generate_arrivals(current_sim_time, self.dt)
            for p in new_passengers:
                if p.priority_level == 3:
                    # 進行搶佔判定
                    self.priority_system.check_and_apply_preemption(self.building, p.origin_floor)
                self.building.add_passenger(p)

            # 更新物理模擬
            events = self.building.update(self.dt)
            step_events.extend(events)

            # 掃描並更新待指派佇列
            self._update_pending_assignments()

            if self.render_mode == "human":
                self.render()
                import time
                time.sleep(0.05)

            if self.building.current_time >= self.max_time:
                terminated = True

        obs = self.building.get_state_vector(max_time=self.max_time)
        info = {
            "current_time": self.building.current_time,
            "pending_assignments_count": len(self.pending_assignments),
            "step_events": step_events
        }

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """執行指派動作並前進模擬"""
        # 1. 執行動作：將當前最前面的呼叫指派給 action 對應的電梯
        if self.pending_assignments:
            call = self.pending_assignments.pop(0)
            self.building.elevators[action].assign_hall_call(call.floor, call.direction)

        # 2. 推進模擬直至下一個指派事件或 Episode 結束
        terminated = False
        step_events = []

        while not self.pending_assignments and not terminated:
            current_sim_time = self.building.current_time + self.dt
            
            # 生成新乘客
            new_passengers = self.traffic_gen.generate_arrivals(current_sim_time, self.dt)
            for p in new_passengers:
                if p.priority_level == 3:
                    self.priority_system.check_and_apply_preemption(self.building, p.origin_floor)
                self.building.add_passenger(p)

            # 更新物理模擬
            events = self.building.update(self.dt)
            step_events.extend(events)

            # 掃描並更新待指派佇列
            self._update_pending_assignments()

            if self.render_mode == "human":
                self.render()
                import time
                time.sleep(0.05)

            if self.building.current_time >= self.max_time:
                terminated = True

        # 3. 計算步長獎勵
        reward, components = calculate_reward(self.building, step_events)

        # 4. 生成觀察值與 info 字典
        obs = self.building.get_state_vector(max_time=self.max_time)
        info = {
            "current_time": self.building.current_time,
            "pending_assignments_count": len(self.pending_assignments),
            "step_events": step_events,
            "reward_components": components
        }

        # 5. 渲染模式處理
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, info

    def _update_pending_assignments(self) -> None:
        """掃描所有樓層的等待佇列，尋找尚未指派給任何電梯的大廳外呼信號"""
        active_calls = self.building.get_pending_hall_calls()

        # 收集所有電梯目前已排定的停靠樓層
        assigned_floors = set()
        for elev in self.building.elevators:
            assigned_floors.update(elev.pending_stops)

        for call in active_calls:
            # 如果呼叫的樓層不在任何電梯的 pending_stops 中，且尚未加入 self.pending_assignments
            if call.floor not in assigned_floors:
                if call not in self.pending_assignments:
                    self.pending_assignments.append(call)

    def action_masks(self) -> np.ndarray:
        """回傳動作遮罩 (MaskablePPO 規定為 shape=(Ne,) 且可用動作為 True)"""
        mask = np.ones(self.num_elevators, dtype=bool)
        for i, elev in enumerate(self.building.elevators):
            # 已滿載的電梯不可被指派
            if elev.current_load >= elev.max_capacity:
                mask[i] = False
            # 故障電梯不可被指派
            if elev.is_out_of_service:
                mask[i] = False
        return mask

    def get_action_mask(self) -> np.ndarray:
        """相容性別名方法"""
        return self.action_masks()

    def load_scenario(self, scenario_name: str) -> None:
        """動態切換大廳乘客交通流模式與到達率"""
        import os
        import yaml
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        scenario_path = os.path.join(base_dir, "configs", "scenarios", f"{scenario_name}.yaml")
        
        if os.path.exists(scenario_path):
            with open(scenario_path, "r", encoding="utf-8") as f:
                scenario_config = yaml.safe_load(f)
                # 更新環境與交通產生器的組態
                self.config["traffic"] = scenario_config
                self.traffic_gen = HospitalTrafficGenerator(self.config)

    def render(self):
        """渲染模擬畫面"""
        if self.render_mode is None:
            return
            
        if self.renderer is None:
            from src.visualization.pygame_renderer import PygameRenderer
            self.renderer = PygameRenderer(self.building, self.max_time)
            
        return self.renderer.render(self.building)

    def close(self):
        """關閉環境"""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
