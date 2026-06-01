import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from src.envs.building import Building, HallCall
from src.envs.traffic_generator import HospitalTrafficGenerator
from src.envs.priority_system import PrioritySystem
from src.utils.config_loader import load_config
from src.envs.passenger import PassengerState
from src.envs.elevator import ElevatorState
from src.envs.event import Event, EventType

class HospitalElevatorMAEnv:
    """
    智慧醫院電梯群控與優先調度 多代理人環境 (TRUE Decentralized Motor-Control Multi-Agent Env)
    每個電梯為一個 Agent，直接接收 Discrete(4) 電梯馬達控制動作：
    0: STOP/IDLE, 1: MOVE_UP, 2: MOVE_DOWN, 3: OPEN_DOOR
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

        self.dt = 1.0  # 步長固定為 1 秒
        self.max_time = 600.0

        # 用於相容性與大廳狀態掃描
        self.pending_assignments: List[HallCall] = []
        self.rng = np.random.default_rng()
        self.renderer = None

        # 定義局部觀察維度：83 維
        # 7 self + 16 internal destinations + 48 floor calls/priorities + 12 other elevators
        self.local_obs_dim = 7 + self.num_floors + 3 * self.num_floors + 4 * (self.num_elevators - 1)
        
        # 觀察與動作空間
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(self.local_obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[int, np.ndarray], Dict[int, Any]]:
        """重置環境"""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        self.building.reset(self.rng)
        self.traffic_gen.reset(self.rng)
        self.pending_assignments.clear()

        # 生成 t=0 時的第一批乘客
        new_passengers = self.traffic_gen.generate_arrivals(0.0, self.dt)
        for p in new_passengers:
            if p.priority_level == 3:
                self.priority_system.check_and_apply_preemption(self.building, p.origin_floor)
            self.building.add_passenger(p)

        obs = {i: self.get_agent_obs(i) for i in range(self.num_elevators)}
        info = {
            "current_time": 0.0,
            "step_events": []
        }
        infos = {i: info for i in range(self.num_elevators)}

        if self.render_mode == "human":
            self.render()

        return obs, infos

    def step(self, actions: Dict[int, Any]) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict[int, bool], Dict[int, Any]]:
        """執行時間驅動(1秒)物理前進並更新狀態"""
        # 支援字典或數組格式動作
        actions_dict = {}
        for idx in range(self.num_elevators):
            act = actions.get(idx, 0) if isinstance(actions, dict) else actions[idx]
            if isinstance(act, (np.ndarray, list)):
                actions_dict[idx] = int(act[0])
            else:
                actions_dict[idx] = int(act)

        # 1. 推進物理模擬 (呼叫 manual_update)
        penalties = {}
        for idx in range(self.num_elevators):
            elev = self.building.elevators[idx]
            # 進行物理更新並取得 invalid command penalty
            penalty = elev.manual_update(actions_dict[idx], self.dt)
            penalties[idx] = penalty

        # 2. 推進模擬時間
        self.building.current_time += self.dt
        terminated = self.building.current_time >= self.max_time

        # 3. 產生新到達乘客
        new_passengers = self.traffic_gen.generate_arrivals(self.building.current_time, self.dt)
        for p in new_passengers:
            if p.priority_level == 3:
                self.priority_system.check_and_apply_preemption(self.building, p.origin_floor)
            self.building.add_passenger(p)

        # 4. 乘客登機與下車邏輯
        step_events = []
        passenger_served_count = {i: 0 for i in range(self.num_elevators)}

        for idx in range(self.num_elevators):
            elev = self.building.elevators[idx]
            
            # 只有當電梯在 DOOR_OPEN 狀態且此步選擇了 OPEN_DOOR 動作時才更新進出
            if elev.state == ElevatorState.DOOR_OPEN and actions_dict[idx] == 3:
                current_floor_idx = elev.current_floor
                floor = self.building.floors[current_floor_idx]

                # (a) 乘客下車 (Deboarding)
                deboarding_passengers = [p for p in elev.passengers if p.destination_floor == current_floor_idx]
                for p in deboarding_passengers:
                    p.state = PassengerState.ARRIVED
                    p.arrive_time = self.building.current_time
                    elev.passengers.remove(p)
                    passenger_served_count[idx] += 1
                    
                    step_events.append(Event(EventType.PASSENGER_DELIVERED, self.building.current_time, {
                        "passenger_id": p.id,
                        "elevator_id": elev.elevator_id,
                        "floor": current_floor_idx,
                        "wait_time": p.get_wait_duration(self.building.current_time),
                        "transit_time": p.arrive_time - p.board_time,
                        "priority": p.priority_level
                    }))

                # 電梯清空時，重設前進方向
                if len(elev.passengers) == 0:
                    elev.current_direction = 0

                # 原地建立外呼方向
                if elev.current_direction == 0 and floor.waiting_queue:
                    first_waiting = floor.waiting_queue[0]
                    elev.current_direction = first_waiting.direction

                # (b) 乘客上車 (Boarding)
                boarding_passengers = []
                for p in list(floor.waiting_queue):
                    if len(elev.passengers) >= elev.max_capacity:
                        break
                    
                    if elev.emergency_target is not None:
                        if elev.emergency_target != current_floor_idx or p.priority_level != 3:
                            continue

                    is_emergency_boarding = (p.priority_level == 3 and elev.emergency_target == current_floor_idx)
                    if elev.current_direction == 0 or p.direction == elev.current_direction or is_emergency_boarding:
                        p.state = PassengerState.IN_TRANSIT
                        p.board_time = self.building.current_time
                        elev.passengers.append(p)
                        floor.waiting_queue.remove(p)
                        boarding_passengers.append(p)
                        passenger_served_count[idx] += 1

                        if is_emergency_boarding:
                            elev.emergency_target = p.destination_floor

                        step_events.append(Event(EventType.PASSENGER_BOARDED, self.building.current_time, {
                            "passenger_id": p.id,
                            "elevator_id": elev.elevator_id,
                            "floor": current_floor_idx,
                            "wait_time": p.get_wait_duration(self.building.current_time),
                            "priority": p.priority_level
                        }))

                # 更新剩餘停靠點與緊急狀態
                elev._pending_stops = list(set(p.destination_floor for p in elev.passengers))
                if elev.emergency_target == current_floor_idx:
                    elev.emergency_target = None

                # (c) 發送外呼完成事件
                if boarding_passengers:
                    served_dir = elev.current_direction
                    still_waiting = any(p.direction == served_dir for p in floor.waiting_queue)
                    if not still_waiting:
                        step_events.append(Event(EventType.HALL_CALL_SERVED, self.building.current_time, {
                            "floor": current_floor_idx,
                            "direction": served_dir,
                            "elevator_id": elev.elevator_id
                        }))

            # 若此步選擇了開門動作(3)卻沒有任何人上下車，施加空開門懲罰以防止無效刷開門
            if actions_dict[idx] == 3 and passenger_served_count[idx] == 0:
                penalties[idx] += -1.0

        # 5. 計算密集獎勵 (Dense Reward Shaping)
        total_invalid_penalty = sum(penalties.values())
        
        energy_penalty = 0.0
        for idx in range(self.num_elevators):
            elev = self.building.elevators[idx]
            if elev.current_velocity > 0.0:
                energy_penalty += -0.05
                
        event_rewards = 0.0
        for event in step_events:
            if event.type.value == "passenger_boarded":
                p_level = event.data["priority"]
                if p_level == 3:
                    event_rewards += 15.0
                elif p_level == 2:
                    event_rewards += 5.0
                elif p_level == 1:
                    event_rewards += 3.0
                else:
                    event_rewards += 1.0
            elif event.type.value == "passenger_delivered":
                p_level = event.data["priority"]
                if p_level == 3:
                    event_rewards += 30.0
                elif p_level == 2:
                    event_rewards += 10.0
                elif p_level == 1:
                    event_rewards += 6.0
                else:
                    event_rewards += 2.0

        waiting_penalty = 0.0
        # 大廳等待乘客懲罰
        for floor in self.building.floors:
            for p in floor.waiting_queue:
                if p.state == PassengerState.WAITING:
                    if p.priority_level == 3:
                        waiting_penalty += -1.0
                    elif p.priority_level == 2:
                        waiting_penalty += -0.3
                    elif p.priority_level == 1:
                        waiting_penalty += -0.15
                    else:
                        waiting_penalty += -0.05

        # 車廂內等待乘客懲罰
        for elev in self.building.elevators:
            for p in elev.passengers:
                if p.priority_level == 3:
                    waiting_penalty += -0.5
                elif p.priority_level == 2:
                    waiting_penalty += -0.15
                elif p.priority_level == 1:
                    waiting_penalty += -0.1
                else:
                    waiting_penalty += -0.03

        step_reward = total_invalid_penalty + energy_penalty + event_rewards + waiting_penalty
        rewards = {i: step_reward for i in range(self.num_elevators)}

        # 6. 生成下一觀察值與 info
        obs = {i: self.get_agent_obs(i) for i in range(self.num_elevators)}
        terminations = {i: terminated for i in range(self.num_elevators)}
        truncations = {i: False for i in range(self.num_elevators)}

        components = {
            "invalid_action_penalty": total_invalid_penalty,
            "energy_penalty": energy_penalty,
            "event_rewards": event_rewards,
            "waiting_penalty": waiting_penalty
        }
        
        info = {
            "current_time": self.building.current_time,
            "step_events": step_events,
            "reward_components": components
        }
        infos = {i: info for i in range(self.num_elevators)}

        if self.render_mode == "human":
            self.render()

        return obs, rewards, terminations, truncations, infos

    def get_agent_obs(self, agent_id: int) -> np.ndarray:
        """
        為單個電梯 Agent 生成 83 維的局部觀察特徵向量
        """
        elev = self.building.elevators[agent_id]
        
        # 1. 自身狀態 (Self State, dim=7)
        pos_norm = elev.current_position / self.building.max_height if self.building.max_height > 0 else 0.0
        vel_norm = elev.current_velocity / elev.rated_speed if elev.rated_speed > 0 else 0.0
        dir_val = float(elev.current_direction)
        door_open = 1.0 if elev.state.value == "door_open" else 0.0
        load_ratio = elev.current_load / elev.max_capacity if elev.max_capacity > 0 else 0.0
        out_of_service = 1.0 if elev.is_out_of_service else 0.0
        preempted = 1.0 if elev.emergency_target is not None else 0.0
        
        self_feats = [pos_norm, vel_norm, dir_val, door_open, load_ratio, out_of_service, preempted]

        # 2. 車內乘客目的地 (Internal Destinations, dim=16)
        dest_feats = [0.0] * self.num_floors
        for p in elev.passengers:
            dest_feats[p.destination_floor] = 1.0

        # 3. 各樓層的外呼與等待狀態 (Lobby Calls & Priorities, dim=48)
        hall_up = [0.0] * self.num_floors
        hall_down = [0.0] * self.num_floors
        priority_feats = [0.0] * self.num_floors
        
        for floor in self.building.floors:
            f_idx = floor.floor_index
            waiting_up = [p for p in floor.waiting_queue if p.direction == 1 and p.state == PassengerState.WAITING]
            waiting_down = [p for p in floor.waiting_queue if p.direction == -1 and p.state == PassengerState.WAITING]
            waiting_priority = [p for p in floor.waiting_queue if p.priority_level > 0 and p.state == PassengerState.WAITING]
            
            if waiting_up:
                hall_up[f_idx] = 1.0
            if waiting_down:
                hall_down[f_idx] = 1.0
            if waiting_priority:
                max_priority = max(p.priority_level for p in waiting_priority)
                priority_feats[f_idx] = max_priority / 3.0
                
        lobby_feats = hall_up + hall_down + priority_feats

        # 4. 其他電梯狀態 (Other Elevators, dim=12)
        other_feats = []
        for idx in range(self.num_elevators):
            if idx == agent_id:
                continue
            other_elev = self.building.elevators[idx]
            oth_pos = other_elev.current_position / self.building.max_height if self.building.max_height > 0 else 0.0
            oth_dir = float(other_elev.current_direction)
            oth_door = 1.0 if other_elev.state.value == "door_open" else 0.0
            oth_load = other_elev.current_load / other_elev.max_capacity if other_elev.max_capacity > 0 else 0.0
            other_feats.extend([oth_pos, oth_dir, oth_door, oth_load])

        obs = np.array(self_feats + dest_feats + lobby_feats + other_feats, dtype=np.float32)
        return obs

    def get_global_state(self) -> np.ndarray:
        """
        為 Centralized Critic 生成全域狀態向量 (dim=183)
        """
        return self.building.get_state_vector(max_time=self.max_time)

    def action_masks(self) -> np.ndarray:
        """
        回傳每個代理人的可用動作遮罩 (num_elevators, 4)
        0: STOP/IDLE, 1: MOVE_UP, 2: MOVE_DOWN, 3: OPEN_DOOR
        """
        mask = np.ones((self.num_elevators, 4), dtype=bool)
        for idx in range(self.num_elevators):
            elev = self.building.elevators[idx]
            
            # MOVE_UP 頂樓不可用
            if elev.current_position >= self.building.floor_heights[-1]:
                mask[idx, 1] = False
                
            # MOVE_DOWN 底樓不可用
            if elev.current_position <= 0.0:
                mask[idx, 2] = False
                
            # OPEN_DOOR 移動中不可用，且未對齊樓層不可用
            if elev.current_velocity > 0.0:
                mask[idx, 3] = False
            else:
                nearest_floor = min(range(len(self.building.floor_heights)), key=lambda f: abs(self.building.floor_heights[f] - elev.current_position))
                if abs(self.building.floor_heights[nearest_floor] - elev.current_position) > 0.1:
                    mask[idx, 3] = False
                    
            # 故障中電梯除 STOP/IDLE 外，全部動作皆無效
            if elev.is_out_of_service:
                mask[idx, 1] = False
                mask[idx, 2] = False
                mask[idx, 3] = False
                
        return mask

    def load_scenario(self, scenario_name: str) -> None:
        """動態切換交通情境"""
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
