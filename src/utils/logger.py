import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class RewardTrackingCallback(BaseCallback):
    """
    自訂 Stable-Baselines3 回呼函式，用來在 TensorBoard 中記錄各項多目標獎勵分量，防止獎勵黑客 (Reward Hacking)。
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.wait_time_penalties = []
        self.energy_penalties = []
        self.emergency_penalties = []
        self.bonuses = []

    def _on_step(self) -> bool:
        # 從環境的 info dict 中讀取 reward_components
        # 由於 SB3 支援多環境 (VecEnv)，infos 是一個 list 結構
        for info in self.locals.get("infos", []):
            if "reward_components" in info:
                comp = info["reward_components"]
                self.wait_time_penalties.append(comp.get("wait_time_penalty", 0.0))
                self.energy_penalties.append(comp.get("energy_penalty", 0.0))
                self.emergency_penalties.append(comp.get("emergency_penalty", 0.0))
                self.bonuses.append(comp.get("bonus", 0.0))

        # 每 100 步計算一次均值並記錄至 TensorBoard
        if self.n_calls % 100 == 0:
            if self.wait_time_penalties:
                self.logger.record("rewards/wait_time_penalty", np.mean(self.wait_time_penalties))
                self.logger.record("rewards/energy_penalty", np.mean(self.energy_penalties))
                self.logger.record("rewards/emergency_penalty", np.mean(self.emergency_penalties))
                self.logger.record("rewards/bonus", np.mean(self.bonuses))
                
                # 清除快取，避免記憶體溢出
                self.wait_time_penalties.clear()
                self.energy_penalties.clear()
                self.emergency_penalties.clear()
                self.bonuses.clear()
                
        return True
