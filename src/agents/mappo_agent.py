import os
import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from typing import Tuple, Optional, Any, Dict

class MAPPOActor(nn.Module):
    """
    MAPPO Parameter-Sharing Actor Network in PyTorch
    Maps local observation to a Gaussian distribution for bidding.
    """
    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Bids must be in [0, 1]
        )
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        mu = self.mu_head(x)
        std = torch.exp(self.log_std).expand_as(mu)
        return mu, std

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, std = self.forward(obs)
        if deterministic:
            action = mu
            log_prob = torch.zeros_like(mu)
        else:
            normal = dist.Normal(mu, std)
            action = normal.sample()
            # Clamp action to [0, 1]
            action = torch.clamp(action, 0.0, 1.0)
            log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, std = self.forward(obs)
        normal = dist.Normal(mu, std)
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy


class MAPPOCritic(nn.Module):
    """
    MAPPO Centralized Critic Network in PyTorch
    Maps global/centralized state to state value V(S).
    """
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def get_mappo_obs_for_elevator(env: Any, agent_id: int) -> np.ndarray:
    """
    通用輔助函數：從任何電梯環境 (單/多代理人) 中為特定電梯計算 MAPPO 局部觀察向量
    """
    building = env.building
    num_floors = env.num_floors
    num_elevators = env.num_elevators
    max_height = building.max_height
    elev = building.elevators[agent_id]
    
    # 1. 任務特徵 (Call Features)
    if env.pending_assignments:
        call = env.pending_assignments[0]
        call_floor_norm = call.floor / (num_floors - 1)
        call_dir = float(call.direction)

        # 計算該呼叫的等待時間與最高優先級
        floor = building.floors[call.floor]
        from src.envs.passenger import PassengerState
        waiting_passengers = [p for p in floor.waiting_queue if p.state == PassengerState.WAITING and p.direction == call.direction]
        if waiting_passengers:
            max_priority = max(p.priority_level for p in waiting_passengers) / 3.0
            max_wait = min(1.0, max(p.get_wait_duration(building.current_time) for p in waiting_passengers) / 120.0)
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
    pos_norm = elev.current_position / max_height if max_height > 0 else 0.0
    vel_norm = elev.current_velocity / elev.rated_speed if elev.rated_speed > 0 else 0.0
    dir_val = float(elev.current_direction)
    doors_open = 1.0 if elev.state.value == "door_open" else 0.0
    
    from src.envs.elevator import ElevatorState
    is_moving = 1.0 if elev.state in (ElevatorState.ACCELERATE, ElevatorState.CRUISE, ElevatorState.DECELERATE) else 0.0
    load_norm = elev.current_load / elev.max_capacity if elev.max_capacity > 0 else 0.0
    
    if env.pending_assignments:
        call = env.pending_assignments[0]
        dist_norm = abs(elev.current_position - building.floor_heights[call.floor]) / max_height
        compat = 1.0 if (elev.current_direction == 0 or elev.current_direction == call.direction) else 0.0
    else:
        dist_norm = 0.0
        compat = 1.0
        
    out_of_service = 1.0 if elev.is_out_of_service else 0.0
    preempted = 1.0 if elev.emergency_target is not None else 0.0

    self_feats = [
        pos_norm, vel_norm, dir_val, doors_open, is_moving,
        load_norm, dist_norm, compat, out_of_service, preempted
    ]

    # 3. 其他電梯狀態簡要特徵 (Other Elevators Feats)
    other_feats = []
    for idx in range(num_elevators):
        if idx == agent_id:
            continue
        other_elev = building.elevators[idx]
        if env.pending_assignments:
            call = env.pending_assignments[0]
            oth_dist = abs(other_elev.current_position - building.floor_heights[call.floor]) / max_height
        else:
            oth_dist = 0.0
        oth_load = other_elev.current_load / other_elev.max_capacity if other_elev.max_capacity > 0 else 0.0
        oth_out = 1.0 if other_elev.is_out_of_service else 0.0
        other_feats.extend([oth_dist, oth_load, oth_out])

    obs = np.array(call_feats + self_feats + other_feats, dtype=np.float32)
    return obs


class MAPPOAgent:
    """
    MAPPO 代理人封裝類別
    支援在訓練與推論 (Demo/Evaluation) 時載入權重並進行預測。
    相容於單代理人 Gym 迴圈與多代理人 Gym 迴圈。
    """
    def __init__(self, model_path: Optional[str] = None, env: Any = None, obs_dim: int = 23, state_dim: int = 183):
        self.env = env
        self.num_elevators = env.num_elevators if env is not None else 4
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        
        # 實例化 actor 網路
        self.actor = MAPPOActor(obs_dim=self.obs_dim)
        
        if model_path is not None:
            self.load(model_path)

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True
    ) -> Tuple[int, Optional[np.ndarray]]:
        """
        推論方法：接收全域/局部觀測值，回傳競標勝出電梯的動作 ID (相容單代理人 API)
        """
        # 如果環境不可用，無法計算局部觀察，直接回傳預設動作
        if self.env is None:
            return 0, None

        # 取得動作遮罩
        mask = self.env.action_masks()

        # 針對每一台電梯計算其局部觀察值與競標分數
        valid_bids = {}
        for idx in range(self.num_elevators):
            if not mask[idx]:
                valid_bids[idx] = -1e9
                continue
            
            # 取得該電梯 Agent 的局部觀察值
            local_obs = get_mappo_obs_for_elevator(self.env, idx)
            local_obs_t = torch.FloatTensor(local_obs).unsqueeze(0)
            
            with torch.no_grad():
                bid, _ = self.actor.get_action(local_obs_t, deterministic=deterministic)
                valid_bids[idx] = float(bid.numpy()[0, 0])

        # 選擇競標值最高的電梯
        action = int(np.argmax([valid_bids[i] for i in range(self.num_elevators)]))
        return action, None

    def save(self, path: str) -> None:
        """儲存 actor 與 critic 的權重"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.actor.state_dict(), path)

    def load(self, path: str) -> None:
        """載入 actor 權重"""
        if os.path.exists(path):
            self.actor.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            self.actor.eval()
            print(f"MAPPO Actor model successfully loaded from {path}")
        else:
            print(f"Warning: MAPPO Model path {path} does not exist. Using random policy.")
