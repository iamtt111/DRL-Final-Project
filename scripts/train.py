import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from src.envs.elevator_env import HospitalElevatorEnv
from src.utils.config_loader import load_config
from src.utils.logger import RewardTrackingCallback

def main():
    parser = argparse.ArgumentParser(description="Train MaskablePPO agent for Hospital Elevator EGCS")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    args = parser.parse_args()

    # 1. 載入組態與超參數
    config = load_config(args.config)
    ppo_config = config.get("ppo", {})

    total_timesteps = args.timesteps or ppo_config.get("total_timesteps", 1000000)

    # 2. 建立向量化環境與評估環境
    vec_env = make_vec_env(lambda: HospitalElevatorEnv(config=config), n_envs=1)
    eval_env = HospitalElevatorEnv(config=config)

    # 3. 建立評估與獎勵監控 Callbacks
    os.makedirs("models/ppo", exist_ok=True)
    os.makedirs("logs/eval", exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/ppo",
        log_path="logs/eval",
        eval_freq=max(1, ppo_config.get("eval_freq", 10000)),
        n_eval_episodes=ppo_config.get("n_eval_episodes", 20),
        deterministic=True,
        render=False
    )
    
    reward_callback = RewardTrackingCallback()
    callbacks = CallbackList([eval_callback, reward_callback])

    # 4. 解析神經網路層參數與激活函數
    policy_kwargs = ppo_config.get("policy_kwargs", {})
    if isinstance(policy_kwargs, dict):
        policy_kwargs = policy_kwargs.copy()
        act_fn = policy_kwargs.get("activation_fn", "ReLU")
        if act_fn == "ReLU":
            policy_kwargs["activation_fn"] = torch.nn.ReLU
        elif act_fn == "Tanh":
            policy_kwargs["activation_fn"] = torch.nn.Tanh

    # 5. 實例化 MaskablePPO 模型
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=ppo_config.get("learning_rate", 3e-4),
        n_steps=ppo_config.get("n_steps", 2048),
        batch_size=ppo_config.get("batch_size", 64),
        n_epochs=ppo_config.get("n_epochs", 10),
        gamma=ppo_config.get("gamma", 0.99),
        gae_lambda=ppo_config.get("gae_lambda", 0.95),
        clip_range=ppo_config.get("clip_range", 0.2),
        ent_coef=ppo_config.get("ent_coef", 0.01),
        vf_coef=ppo_config.get("vf_coef", 0.5),
        max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
        policy_kwargs=policy_kwargs,
        tensorboard_log="logs/ppo",
        verbose=1
    )

    # 6. 開始訓練
    print(f"Starting MaskablePPO training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    
    # 7. 儲存最終模型
    model.save("models/ppo/final_model")
    print("Training completed successfully! Final model saved at models/ppo/final_model.zip")

if __name__ == "__main__":
    main()
