import os
import numpy as np
from typing import Tuple, Optional, Any

def deterministic_hash(j: int, c: int, idx: int, limit: int) -> int:
    """進程無關、確定性的整數三元組 FNV-1a 哈希函數"""
    h = 2166136261
    for val in (j, c, idx):
        val_unsigned = val & 0xffffffff
        h = h ^ val_unsigned
        h = (h * 16777619) & 0xffffffff
    return h % limit

class TileCoder:
    """基於確定性哈希的 Tile Coding 特徵提取器"""

    def __init__(self, num_tilings: int = 8, bins_per_coord: int = 8, feature_dim: int = 1048576):
        self.num_tilings = num_tilings
        self.bins_per_coord = bins_per_coord
        self.feature_dim = feature_dim

    def get_features(self, state: np.ndarray) -> np.ndarray:
        """將連續的狀態向量轉換為稀疏二值特徵向量 (相容性與測試用途)"""
        features = np.zeros(self.feature_dim, dtype=np.float32)
        d = len(state)

        for j in range(self.num_tilings):
            offset = j / self.num_tilings
            for c in range(d):
                val = state[c]
                val = max(-1.0, min(1.0, val))
                idx = int(np.floor(val * self.bins_per_coord - offset))
                h = deterministic_hash(j, c, idx, self.feature_dim)
                features[h] = 1.0
        return features

    def get_active_indices(self, state: np.ndarray) -> list:
        """僅返回被激活的特徵索引列表 (大幅降低計算與記憶體開銷)"""
        indices = []
        d = len(state)
        for j in range(self.num_tilings):
            offset = j / self.num_tilings
            for c in range(d):
                val = state[c]
                val = max(-1.0, min(1.0, val))
                idx = int(np.floor(val * self.bins_per_coord - offset))
                h = deterministic_hash(j, c, idx, self.feature_dim)
                indices.append(h)
        return indices


class SarsaAgent:
    """
    SARSA(λ) 搭配 Tile Coding 的傳統強化學習對照組代理人 (稀疏優化版)
    """

    def __init__(
        self,
        num_elevators: int = 4,
        feature_dim: int = 1048576,
        learning_rate: float = 0.5,
        gamma: float = 0.9,
        lambda_trace: float = 0.5,
        epsilon: float = 0.08,
        env: Any = None
    ):
        self.num_elevators = num_elevators
        self.feature_dim = feature_dim
        self.alpha = learning_rate
        self.gamma = gamma
        self.lambda_trace = lambda_trace
        self.epsilon = epsilon
        self.env = env

        # 初始化權重
        self.weights = np.zeros((self.num_elevators, self.feature_dim), dtype=np.float32)
        
        # 稀疏資格迹：每台電梯一個 dict (格式為 {feature_index: trace_value})
        self.eligibility_traces = [{} for _ in range(self.num_elevators)]

        self.tile_coder = TileCoder(feature_dim=self.feature_dim)

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True
    ) -> Tuple[int, Optional[np.ndarray]]:
        """ε-greedy 策略動作選擇 (支援 Action Masking)"""
        # 取得動作遮罩
        mask = None
        if self.env is not None:
            mask = self.env.action_masks()

        # ε-greedy 探索
        if not deterministic and np.random.random() < self.epsilon:
            if mask is not None and np.any(mask):
                valid_actions = np.where(mask)[0]
                return int(np.random.choice(valid_actions)), None
            else:
                return np.random.randint(self.num_elevators), None

        # 稀疏特徵提取與 Q 值加和計算
        active_indices = self.tile_coder.get_active_indices(observation)
        q_values = np.zeros(self.num_elevators)
        for a in range(self.num_elevators):
            q_values[a] = np.sum(self.weights[a, active_indices])

        # 應用動作遮罩
        if mask is not None:
            q_values[~mask] = -1e9

        action = int(np.argmax(q_values))
        return action, None

    def update(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        s_prime: np.ndarray,
        a_prime: int,
        done: bool
    ) -> float:
        """更新權重與資格迹，並回傳 TD Error (稀疏優化版)"""
        active_s = self.tile_coder.get_active_indices(s)
        
        # 計算 Q(s, a)
        q_sa = np.sum(self.weights[a, active_s])

        # 計算 Q(s', a')
        if not done:
            active_sp = self.tile_coder.get_active_indices(s_prime)
            q_sp_ap = np.sum(self.weights[a_prime, active_sp])
        else:
            q_sp_ap = 0.0

        # TD Error δ = R + γ * Q(s', a') - Q(s, a)
        td_error = r + self.gamma * q_sp_ap - q_sa

        # 稀疏衰減資格迹 (Decay traces)
        decay = self.gamma * self.lambda_trace
        for action in range(self.num_elevators):
            trace = self.eligibility_traces[action]
            for idx in list(trace.keys()):
                trace[idx] *= decay
                if trace[idx] < 1e-4:
                    del trace[idx]

        # 更新目前動作的資格迹 (Replacing Trace)
        trace_a = self.eligibility_traces[a]
        for idx in active_s:
            trace_a[idx] = 1.0

        # 學習步長 (Alpha) 需除以總被激活特徵數 (num_tilings * state_dim) 以避免更新發散
        step_size = self.alpha / (self.tile_coder.num_tilings * len(s))

        # 僅對非零的資格迹進行權重更新，大幅提高運算效率
        for action in range(self.num_elevators):
            trace = self.eligibility_traces[action]
            if trace:
                indices = list(trace.keys())
                vals = np.array(list(trace.values()), dtype=np.float32)
                self.weights[action, indices] += step_size * td_error * vals

        if done:
            # 結束時重置所有資格迹
            for action in range(self.num_elevators):
                self.eligibility_traces[action].clear()

        return float(td_error)

    def save(self, path: str) -> None:
        """儲存權重為 .npz 檔"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, weights=self.weights)

    def load(self, path: str) -> None:
        """載入權重"""
        if os.path.exists(path):
            data = np.load(path)
            self.weights = data["weights"]
