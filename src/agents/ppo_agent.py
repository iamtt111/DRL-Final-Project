import numpy as np
from typing import Tuple, Optional, Any
from sb3_contrib import MaskablePPO

class PPOAgent:
    """
    Stable-Baselines3 sb3_contrib.MaskablePPO 代理人封裝
    支援在訓練與推論時載入 action_masks
    """

    def __init__(self, model_path: Optional[str] = None, env: Any = None):
        self.env = env
        if model_path is not None:
            self.model = MaskablePPO.load(model_path)
        else:
            self.model = None

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True
    ) -> Tuple[int, Optional[np.ndarray]]:
        """
        利用 MaskablePPO 預測下一個電梯指派動作
        """
        # 取得動作遮罩
        mask = None
        if self.env is not None:
            mask = self.env.action_masks()

        if self.model is not None:
            action, next_state = self.model.predict(
                observation,
                action_masks=mask,
                deterministic=deterministic
            )
            return int(action), next_state
        else:
            # 若未載入模型，隨機從可用的電梯中指派
            if mask is not None and np.any(mask):
                valid_actions = np.where(mask)[0]
                action = int(np.random.choice(valid_actions))
            else:
                action = 0
            return action, None
