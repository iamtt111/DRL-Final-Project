import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from src.envs.building import Building, HallCall
from src.envs.traffic_generator import HospitalTrafficGenerator
from src.envs.priority_system import PrioritySystem
from src.rewards.reward_functions import calculate_reward
from src.utils.config_loader import load_config
from src.envs.passenger import PassengerState

class HospitalElevatorMAEnv:
    """
    智慧醫院電梯群控與優先調度 多代理人環境 (Bidding-Based Cooperative Multi-Agent Env)
    每個電梯為一個 Agent，進行外呼派單競標。
    """

    def __init__(self, config: Dict[str, Any] = None, render_mode: str = None):
        self.config = config or load_config()
        self.render_mode = render_mode

        # 基礎配置讀取
        building_config = self.config.get("building", {})
        self.num_floors = building_config.get("num_floors", 16)
        
        elevator_config = self.config.get("elevator", {})
        self.num_elevators = elevator_config.get("num_elevators", 4)
        
        # 建立模擬核心
        self.building = Building(self.config)
        self.traffic_gen = HospitalTrafficGenerator(self.config)
        self.priority_system = PrioritySystem(self.config)

        self.dt = 1.0
        self.max_time = 600.0

        # 當前等待指派的大廳呼叫佇列
        self.pending_assignments: List[HallCall] = []
        self.rng = np.random.default_rng()
        self.renderer = None

        # 定義觀察與動作空間
        # 每個 agent 觀察維度：4 (任務特徵) + 10 (電梯自身狀態) + 3 * (電梯數-1) (其他電梯狀態)
        self.local_obs_dim = 4 + 10 + 3 * (self.num_elevators - 1)
        
        # 觀察空間
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(self.local_obs_dim,), dtype=np.float32
        )
        # 動作空間：連續的外呼競標分數 [0.0, 1.0]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[int, np.ndarray], Dict[int, Any]]:
        """重置環境"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.building.reset(self.rng)
        self.traffic_gen.reset(self.rng)
        self.pending_assignments.clear()

        terminated = False
        step_events = []

        # 推進模擬直到有待指派呼叫，或者時間終了
        while not self.pending_assignments and not terminated:
            current_sim_time = self.building.current_time + self.dt
            new_passengers = self.traffic_gen.generate_arrivals(current_sim_time, self.dt)
            for p in new_passengers:
                if p.priority_level == 3:
                    self.priority_system.check_and_apply_preemption(self.building, p.origin_floor)
                self.building.add_passenger(p)

            events = self.building.update(self.dt)
            step_events.extend(events)
            self._update_pending_assignments()

            if self.render_mode == "human":
                self.render()
                import time
                time.sleep(0.05)

            if self.building.current_time >= self.max_time:
                terminated = True

        obs = {i: self.get_agent_obs(i) for i in range(self.num_elevators)}
        info = {
            "current_time": self.building.current_time,
            "pending_assignments_count": len(self.pending_assignments),
            "step_events": step_events
        }
        infos = {i: info for i in range(self.num_elevators)}

        return obs, infos

    def step(self, actions: Dict[int, Any]) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict[int, bool], Dict[int, Any]]:
        """執行競標派單並推進模擬"""
        # 1. 動作遮罩過濾：故障或滿載電梯的競標分數無效
        mask = self.action_masks()
        valid_bids = {}
        for idx in range(self.num_elevators):
            # actions[idx] 可以是 scalar, numpy array, 或是 list
            act = actions.get(idx, 0.0)
            if isinstance(act, (np.ndarray, list)):
                bid = float(act[0])
            else:
                bid = float(act)

            if not mask[idx]:
                bid = -1e9
            valid_bids[idx] = bid

        # 選擇競標值最高的電梯
        best_elevator_id = int(np.argmax([valid_bids[idx] for idx in range(self.num_elevators)]))

        # 執行指派
        if self.pending_assignments:
            call = self.pending_assignments.pop(0)
            self.building.elevators[best_elevator_id].assign_hall_call(call.floor, call.direction)

        # 2. 推進模擬直至下一個指派信號或 Episode 結束
        terminated = False
        step_events = []

        while not self.pending_assignments and not terminated:
            current_sim_time = self.building.current_time + self.dt
            new_passengers = self.traffic_gen.generate_arrivals(current_sim_time, self.dt)
            for p in new_passengers:
                if p.priority_level == 3:
                    self.priority_system.check_and_apply_preemption(self.building, p.origin_floor)
                self.building.add_passenger(p)

            events = self.building.update(self.dt)
            step_events.extend(events)
            self._update_pending_assignments()

            if self.render_mode == "human":
                self.render()
                import time
                time.sleep(0.05)

            if self.building.current_time >= self.max_time:
                terminated = True

        # 3. 計算共享步長獎勵
        reward_config = self.config.get("reward", {})
        weights = reward_config.get("weights", None)
        reward, components = calculate_reward(self.building, step_events, weights=weights)
        rewards = {i: reward for i in range(self.num_elevators)}

        # 4. 生成下一狀態
        obs = {i: self.get_agent_obs(i) for i in range(self.num_elevators)}
        terminations = {i: terminated for i in range(self.num_elevators)}
        truncations = {i: False for i in range(self.num_elevators)}
        
        info = {
            "current_time": self.building.current_time,
            "pending_assignments_count": len(self.pending_assignments),
            "step_events": step_events,
            "reward_components": components
        }
        infos = {i: info for i in range(self.num_elevators)}

        if self.render_mode == "human":
            self.render()

        return obs, rewards, terminations, truncations, infos

    def get_agent_obs(self, agent_id: int) -> np.ndarray:
        """
        為單個電梯 Agent 生成局部觀察特徵向量
        """
        elev = self.building.elevators[agent_id]
        
        # 1. 任務特徵 (Call Features)
        if self.pending_assignments:
            call = self.pending_assignments[0]
            call_floor_norm = call.floor / (self.num_floors - 1)
            call_dir = float(call.direction)

            # 計算該呼叫的等待時間與最高優先級
            floor = self.building.floors[call.floor]
            waiting_passengers = [p for p in floor.waiting_queue if p.state == PassengerState.WAITING and p.direction == call.direction]
            if waiting_passengers:
                max_priority = max(p.priority_level for p in waiting_passengers) / 3.0
                max_wait = min(1.0, max(p.get_wait_duration(self.building.current_time) for p in waiting_passengers) / 120.0)
            else:
                max_priority = 0.0
                max_wait = 0.0
        else:
            call_floor_norm = 0.0
            call_dir = 0.0
            max_priority = 0.0
            max_wait = 0.0

        call_feats = [call_floor_norm, call_dir, max_priority, max_wait]

        # 2. 電梯自身狀態 (Self Elevator State)
        pos_norm = elev.current_position / self.building.max_height if self.building.max_height > 0 else 0.0
        vel_norm = elev.current_velocity / elev.rated_speed if elev.rated_speed > 0 else 0.0
        dir_val = float(elev.current_direction)
        doors_open = 1.0 if elev.state.value == "door_open" else 0.0
        
        from src.envs.elevator import ElevatorState
        is_moving = 1.0 if elev.state in (ElevatorState.ACCELERATE, ElevatorState.CRUISE, ElevatorState.DECELERATE) else 0.0
        load_norm = elev.current_load / elev.max_capacity if elev.max_capacity > 0 else 0.0
        
        if self.pending_assignments:
            call = self.pending_assignments[0]
            dist_norm = abs(elev.current_position - self.building.floor_heights[call.floor]) / self.building.max_height
            compat = 1.0 if (elev.current_direction == 0 or elev.current_direction == call.direction) else 0.0
        else:
            dist_norm = 0.0
            compat = 1.0
            
        out_of_service = 1.0 if elev.is_out_of_service else 0.0
        preempted = 1.0 if elev.emergency_target is not None else 0.0
        stops_count_norm = len(elev.pending_stops) / self.num_floors

        self_feats = [
            pos_norm, vel_norm, dir_val, doors_open, is_moving,
            load_norm, dist_norm, compat, out_of_service, preempted, stops_count_norm
        ]
        # 只取前 10 個
        self_feats = self_feats[:10]

        # 3. 其他電梯狀態簡要特徵 (Other Elevators Feats)
        other_feats = []
        for idx in range(self.num_elevators):
            if idx == agent_id:
                continue
            other_elev = self.building.elevators[idx]
            # 距離當前呼叫的距離
            if self.pending_assignments:
                call = self.pending_assignments[0]
                oth_dist = abs(other_elev.current_position - self.building.floor_heights[call.floor]) / self.building.max_height
            else:
                oth_dist = 0.0
            oth_load = other_elev.current_load / other_elev.max_capacity if other_elev.max_capacity > 0 else 0.0
            oth_out = 1.0 if other_elev.is_out_of_service else 0.0
            other_feats.extend([oth_dist, oth_load, oth_out])

        obs = np.array(call_feats + self_feats + other_feats, dtype=np.float32)
        return obs

    def get_global_state(self) -> np.ndarray:
        """
        為 Centralized Critic 生成全局狀態向量
        """
        return self.building.get_state_vector(max_time=self.max_time)

    def _update_pending_assignments(self) -> None:
        """掃描所有樓層的等待佇列，尋找尚未指派給任何電梯的大廳外呼信號"""
        active_calls = self.building.get_pending_hall_calls()
        assigned_floors = set()
        for elev in self.building.elevators:
            assigned_floors.update(elev.pending_stops)

        for call in active_calls:
            if call.floor not in assigned_floors:
                if call not in self.pending_assignments:
                    self.pending_assignments.append(call)

    def action_masks(self) -> np.ndarray:
        """回傳可用電梯遮罩"""
        mask = np.ones(self.num_elevators, dtype=bool)
        for i, elev in enumerate(self.building.elevators):
            if elev.current_load >= elev.max_capacity:
                mask[i] = False
            if elev.is_out_of_service:
                mask[i] = False
        return mask

    def load_scenario(self, scenario_name: str) -> None:
        """動態切換大廳乘客交通流模式與到達率"""
        import os
        import yaml
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        scenario_path = os.path.join(base_dir, "configs", "scenarios", f"{scenario_name}.yaml")
        
        if os.path.exists(scenario_path):
            with open(scenario_path, "r", encoding="utf-8") as f:
                scenario_config = yaml.safe_load(f)
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
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
