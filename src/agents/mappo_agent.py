import os
import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from typing import Tuple, Optional, Any, Dict
from src.envs.passenger import PassengerState

class MAPPOActor(nn.Module):
    """
    MAPPO Parameter-Sharing Actor Network in PyTorch
    Maps local observation to a Categorical distribution over 4 discrete motor commands:
    [0: STOP/IDLE, 1: MOVE_UP, 2: MOVE_DOWN, 3: OPEN_DOOR]
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
        self.action_head = nn.Linear(hidden_dim // 2, 4)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.action_head(self.net(obs))

    def get_action(self, obs: torch.Tensor, action_masks: Optional[torch.Tensor] = None, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)
        
        if action_masks is not None:
            # 將無效動作的 logit 設為極小值以遮罩
            masked_logits = logits.clone()
            # 確保遮罩為 boolean tensor
            masked_logits[~action_masks] = -1e9
            probs = dist.Categorical(logits=masked_logits)
        else:
            probs = dist.Categorical(logits=logits)
            
        if deterministic:
            # 選擇機率最大的動作
            action = torch.argmax(probs.probs, dim=-1, keepdim=True)
        else:
            action = probs.sample().unsqueeze(-1)
            
        log_prob = probs.log_prob(action.squeeze(-1)).unsqueeze(-1)
        return action, log_prob

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor, action_masks: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)
        
        if action_masks is not None:
            masked_logits = logits.clone()
            masked_logits[~action_masks] = -1e9
            probs = dist.Categorical(logits=masked_logits)
        else:
            probs = dist.Categorical(logits=logits)
            
        log_prob = probs.log_prob(action.squeeze(-1).long()).unsqueeze(-1)
        entropy = probs.entropy().unsqueeze(-1)
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
    通用輔助函數：從環境中為特定電梯計算 83 維 MAPPO 局部觀察向量
    """
    if hasattr(env, "get_agent_obs"):
        return env.get_agent_obs(agent_id)
        
    building = env.building
    num_floors = env.num_floors
    num_elevators = env.num_elevators
    elev = building.elevators[agent_id]
    
    pos_norm = elev.current_position / building.max_height if building.max_height > 0 else 0.0
    vel_norm = elev.current_velocity / elev.rated_speed if elev.rated_speed > 0 else 0.0
    dir_val = float(elev.current_direction)
    door_open = 1.0 if elev.state.value == "door_open" else 0.0
    load_ratio = elev.current_load / elev.max_capacity if elev.max_capacity > 0 else 0.0
    out_of_service = 1.0 if elev.is_out_of_service else 0.0
    preempted = 1.0 if elev.emergency_target is not None else 0.0
    
    self_feats = [pos_norm, vel_norm, dir_val, door_open, load_ratio, out_of_service, preempted]

    dest_feats = [0.0] * num_floors
    for p in elev.passengers:
        dest_feats[p.destination_floor] = 1.0

    hall_up = [0.0] * num_floors
    hall_down = [0.0] * num_floors
    priority_feats = [0.0] * num_floors
    
    for floor in building.floors:
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

    other_feats = []
    for idx in range(num_elevators):
        if idx == agent_id:
            continue
        other_elev = building.elevators[idx]
        oth_pos = other_elev.current_position / building.max_height if building.max_height > 0 else 0.0
        oth_dir = float(other_elev.current_direction)
        oth_door = 1.0 if other_elev.state.value == "door_open" else 0.0
        oth_load = other_elev.current_load / other_elev.max_capacity if other_elev.max_capacity > 0 else 0.0
        other_feats.extend([oth_pos, oth_dir, oth_door, oth_load])

    return np.array(self_feats + dest_feats + lobby_feats + other_feats, dtype=np.float32)


class MAPPOAgent:
    """
    MAPPO 代理人封裝類別
    """
    def __init__(self, model_path: Optional[str] = None, env: Any = None, obs_dim: int = 83, state_dim: int = 183):
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
        observation: Any,
        state: Optional[np.ndarray] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True
    ) -> Tuple[Any, Optional[np.ndarray]]:
        """
        推論方法：支援單代理人/多代理人局部觀測，回傳動作
        """
        if self.env is None:
            if isinstance(observation, dict):
                return {i: 0 for i in range(self.num_elevators)}, None
            return 0, None

        # 獲取動作遮罩
        masks = self.env.action_masks()  # (num_elevators, 4)

        if isinstance(observation, dict):
            actions = {}
            for idx, local_obs in observation.items():
                local_obs_t = torch.FloatTensor(local_obs).unsqueeze(0)
                mask_t = torch.BoolTensor(masks[idx]).unsqueeze(0)
                with torch.no_grad():
                    action, _ = self.actor.get_action(local_obs_t, action_masks=mask_t, deterministic=deterministic)
                    actions[idx] = int(action.numpy()[0, 0])
            return actions, None
        else:
            # 單一電梯或 legacy 單代理人呼叫
            local_obs_t = torch.FloatTensor(observation).unsqueeze(0)
            mask_t = torch.BoolTensor(masks[0]).unsqueeze(0)
            with torch.no_grad():
                action, _ = self.actor.get_action(local_obs_t, action_masks=mask_t, deterministic=deterministic)
                act_val = int(action.numpy()[0, 0])
            return act_val, None

    def save(self, path: str) -> None:
        """儲存 actor 的權重"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.actor.state_dict(), path)

    def load(self, path: str) -> None:
        """載入 actor 權重"""
        if os.path.exists(path):
            try:
                self.actor.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                self.actor.eval()
                print(f"MAPPO Actor model successfully loaded from {path}")
            except Exception as e:
                print(f"Warning: Failed to load MAPPO model from {path} due to mismatch (likely legacy weights): {e}.")
                print("Using randomly initialized policy instead.")
                self.actor.eval()
        else:
            print(f"Warning: MAPPO Model path {path} does not exist. Using random policy.")
