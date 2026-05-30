import os
import numpy as np
from typing import Tuple, Optional, Any

def deterministic_hash(j: int, c: int, idx: int, limit: int) -> int:
    """進程無關、確定性的整數三元組 FNV-1a 哈希函數"""
    h = 2166136261
    for val in (j, c, idx):
        # 轉換為 32 位無符號整數進行運算
        val_unsigned = val & 0xffffffff
        h = h ^ val_unsigned
        h = (h * 16777619) & 0xffffffff
    return h % limit

class TileCoder:
    """基於確定性哈希的 Tile Coding 特徵提取器"""

    def __init__(self, num_tilings: int = 8, bins_per_coord: int = 8, feature_dim: int = 8192):
        self.num_tilings = num_tilings
        self.bins_per_coord = bins_per_coord
        self.feature_dim = feature_dim

    def get_features(self, state: np.ndarray) -> np.ndarray:
        """將連續的狀態向量轉換為稀疏二值特徵向量"""
        features = np.zeros(self.feature_dim, dtype=np.float32)
        d = len(state)

        for j in range(self.num_tilings):
            # 每層 tiling 的平移偏移量
            offset = j / self.num_tilings
            for c in range(d):
                val = state[c]
                # 限制值域在 [-1.0, 1.0]
                val = max(-1.0, min(1.0, val))

                # 計算所屬 bin 索引
                idx = int(np.floor(val * self.bins_per_coord - offset))

                # 確定性哈希映射到特徵維度
                h = deterministic_hash(j, c, idx, self.feature_dim)
                features[h] = 1.0
        return features


class SarsaAgent:
    """
    SARSA(λ) 搭配 Tile Coding 的傳統強化學習對照組代理人
    """

    def __init__(
        self,
        num_elevators: int = 4,
        feature_dim: int = 8192,
        learning_rate: float = 0.6,
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

        # 初始化權重與資格迹
        self.weights = np.zeros((self.num_elevators, self.feature_dim), dtype=np.float32)
        self.eligibility_traces = np.zeros((self.num_elevators, self.feature_dim), dtype=np.float32)

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

        # 計算 Q(s, a) = weights * phi
        phi = self.tile_coder.get_features(observation)
        q_values = np.dot(self.weights, phi)

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
        """更新權重與資格迹，並回傳 TD Error"""
        phi_s = self.tile_coder.get_features(s)
        
        # 計算 Q(s, a)
        q_sa = np.dot(self.weights[a], phi_s)

        # 計算 Q(s', a')
        if not done:
            phi_sp = self.tile_coder.get_features(s_prime)
            q_sp_ap = np.dot(self.weights[a_prime], phi_sp)
        else:
            q_sp_ap = 0.0

        # TD Error δ = R + γ * Q(s', a') - Q(s, a)
        td_error = r + self.gamma * q_sp_ap - q_sa

        # 更新資格迹 (Replacing Trace)
        self.eligibility_traces *= self.gamma * self.lambda_trace
        self.eligibility_traces[a] = np.maximum(self.eligibility_traces[a], phi_s)

        # 更新權重
        # 分母為 Tile Coding 中的特徵疊加數目以維持穩定的學習步長
        num_active_features = self.tile_coder.num_tilings * len(s)
        step_size = self.alpha / num_active_features

        for action in range(self.num_elevators):
            self.weights[action] += step_size * td_error * self.eligibility_traces[action]

        if done:
            # 結束時重置資格迹
            self.eligibility_traces.fill(0.0)

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
