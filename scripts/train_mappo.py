import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from typing import Dict, Any
from src.envs.elevator_ma_env import HospitalElevatorMAEnv
from src.agents.mappo_agent import MAPPOActor, MAPPOCritic
from src.utils.config_loader import load_config

class MAPPOBuffer:
    """
    用於 MAPPO 集中訓練、分佈執行的 Trajectory 緩衝區
    """
    def __init__(self, num_agents: int, obs_dim: int, state_dim: int, capacity: int):
        self.obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, num_agents, 1), dtype=np.float32)
        self.log_probs = np.zeros((capacity, num_agents, 1), dtype=np.float32)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.values = np.zeros((capacity, 1), dtype=np.float32)
        self.returns = np.zeros((capacity, 1), dtype=np.float32)
        self.advantages = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.max_size = capacity

    def store(self, obs: Dict[int, np.ndarray], act: Dict[int, float], log_prob: Dict[int, float], state: np.ndarray, reward: float, done: bool, val: float):
        if self.ptr >= self.max_size:
            return
        for i in range(len(obs)):
            self.obs[self.ptr, i] = obs[i]
            self.actions[self.ptr, i] = act[i]
            self.log_probs[self.ptr, i] = log_prob[i]
        self.states[self.ptr] = state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = val
        self.ptr += 1

    def compute_returns_and_advantages(self, last_val: float, gamma: float = 0.99, lam: float = 0.95):
        gae = 0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_val = last_val
            else:
                next_val = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_val * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def clear(self):
        self.ptr = 0

    def get_batches(self, batch_size: int):
        indices = np.arange(self.ptr)
        np.random.shuffle(indices)
        
        # 將資料轉換為 PyTorch 變數
        obs_t = torch.FloatTensor(self.obs[:self.ptr])
        actions_t = torch.FloatTensor(self.actions[:self.ptr])
        log_probs_t = torch.FloatTensor(self.log_probs[:self.ptr])
        states_t = torch.FloatTensor(self.states[:self.ptr])
        returns_t = torch.FloatTensor(self.returns[:self.ptr])
        advantages_t = torch.FloatTensor(self.advantages[:self.ptr])
        
        # 進行標準化優勢函數以穩定 PPO 更新
        adv_mean = advantages_t.mean()
        adv_std = advantages_t.std() + 1e-8
        advantages_t = (advantages_t - adv_mean) / adv_std

        for start in range(0, self.ptr, batch_size):
            batch_indices = indices[start:start + batch_size]
            yield (
                obs_t[batch_indices],
                actions_t[batch_indices],
                log_probs_t[batch_indices],
                states_t[batch_indices],
                returns_t[batch_indices],
                advantages_t[batch_indices]
            )


def evaluate(env: HospitalElevatorMAEnv, actor: MAPPOActor, n_episodes: int = 10) -> float:
    """在評估環境中測試當前策略的平均回報"""
    total_rewards = []
    num_agents = env.num_elevators
    
    for _ in range(n_episodes):
        obs, infos = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            actions = {}
            for i in range(num_agents):
                obs_t = torch.FloatTensor(obs[i]).unsqueeze(0)
                with torch.no_grad():
                    bid, _ = actor.get_action(obs_t, deterministic=True)
                    actions[i] = bid.numpy()[0]
                    
            obs, rewards, terminations, truncations, infos = env.step(actions)
            episode_reward += rewards[0]  # 由於所有代理人共享獎勵，取 0 的獎勵即為全局獎勵
            done = all(terminations.values())
            
        total_rewards.append(episode_reward)
        
    return float(np.mean(total_rewards))


def main():
    parser = argparse.ArgumentParser(description="Train MAPPO agent for Hospital Elevator EGCS")
    parser.add_argument("--config", type=str, default="configs/train_mappo.yaml", help="Path to MAPPO config file")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    args = parser.parse_args()

    # 1. 載入組態與超參數
    config = load_config(args.config)
    mappo_config = config.get("mappo", {})

    total_timesteps = args.timesteps or mappo_config.get("total_timesteps", 1000000)
    lr = mappo_config.get("learning_rate", 3e-4)
    gamma = mappo_config.get("gamma", 0.99)
    lam = mappo_config.get("gae_lambda", 0.95)
    clip_range = mappo_config.get("clip_range", 0.2)
    ent_coef = mappo_config.get("ent_coef", 0.01)
    vf_coef = mappo_config.get("vf_coef", 0.5)
    max_grad_norm = mappo_config.get("max_grad_norm", 0.5)
    hidden_dim = mappo_config.get("hidden_dim", 256)
    n_epochs = mappo_config.get("n_epochs", 10)
    batch_size = mappo_config.get("batch_size", 64)
    buffer_capacity = mappo_config.get("buffer_capacity", 2048)
    eval_freq = mappo_config.get("eval_freq", 10000)
    n_eval_episodes = mappo_config.get("n_eval_episodes", 10)

    # 2. 建立訓練與評估環境
    env = HospitalElevatorMAEnv(config=config)
    eval_env = HospitalElevatorMAEnv(config=config)
    
    # 重置環境以初始化電梯，確保 get_state_vector() 能取到正確的維度
    env.reset()
    eval_env.reset()

    obs_dim = env.local_obs_dim
    state_dim = env.building.get_state_vector().shape[0]
    num_agents = env.num_elevators

    # 3. 建立神經網路與優化器
    actor = MAPPOActor(obs_dim=obs_dim, hidden_dim=hidden_dim)
    
    # 載入預訓練的 Actor 權重 (冷啟動)
    pretrain_path = "models/mappo/pretrain_actor.pt"
    if os.path.exists(pretrain_path):
        actor.load_state_dict(torch.load(pretrain_path, map_location=torch.device('cpu')))
        print(f"Successfully loaded pretrained MAPPO Actor weights from {pretrain_path}")
    else:
        print("No pretrained MAPPO Actor weights found. Training from scratch.")
        
    critic = MAPPOCritic(state_dim=state_dim, hidden_dim=hidden_dim)
    
    actor_opt = optim.Adam(actor.parameters(), lr=lr)
    critic_opt = optim.Adam(critic.parameters(), lr=lr)

    # 4. 建立緩衝區
    buffer = MAPPOBuffer(num_agents=num_agents, obs_dim=obs_dim, state_dim=state_dim, capacity=buffer_capacity)

    # 建立模型與日誌存檔路徑
    os.makedirs("models/mappo", exist_ok=True)

    print(f"MAPPO config loaded successfully! Obs dim: {obs_dim}, State dim: {state_dim}, Num Elevators: {num_agents}")
    print(f"Starting training for {total_timesteps} timesteps...")

    current_timesteps = 0
    best_eval_reward = -float('inf')
    last_eval_step = 0

    obs, infos = env.reset()

    while current_timesteps < total_timesteps:
        # Collect rollouts
        for _ in range(buffer_capacity):
            actions = {}
            log_probs = {}
            
            # 對於每台電梯計算觀察值與輸出 Action Bid
            for idx in range(num_agents):
                local_obs = obs[idx]
                obs_t = torch.FloatTensor(local_obs).unsqueeze(0)
                
                with torch.no_grad():
                    bid, log_prob = actor.get_action(obs_t, deterministic=False)
                    actions[idx] = float(bid.numpy()[0, 0])
                    log_probs[idx] = float(log_prob.numpy()[0, 0])

            # 獲取集中式 Critic 的狀態值估計
            global_state = env.get_global_state()
            state_t = torch.FloatTensor(global_state).unsqueeze(0)
            with torch.no_grad():
                val = float(critic(state_t).numpy()[0, 0])

            # 執行步驟
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            reward = rewards[0]
            done = all(terminations.values())

            # 儲存到 buffer
            buffer.store(obs, actions, log_probs, global_state, reward, done, val)
            
            obs = next_obs
            current_timesteps += 1

            if done:
                obs, infos = env.reset()
                # 重新計算最後一狀態值並結算
                state_t = torch.FloatTensor(env.get_global_state()).unsqueeze(0)
                with torch.no_grad():
                    last_val = float(critic(state_t).numpy()[0, 0])
                buffer.compute_returns_and_advantages(last_val, gamma, lam)
                break
        else:
            # Buffer 填滿但 Episode 尚未結束
            state_t = torch.FloatTensor(env.get_global_state()).unsqueeze(0)
            with torch.no_grad():
                last_val = float(critic(state_t).numpy()[0, 0])
            buffer.compute_returns_and_advantages(last_val, gamma, lam)

        # 進行 Policy 與 Critic 更新
        if buffer.ptr > 0:
            actor.train()
            critic.train()
            
            total_act_loss = 0.0
            total_crit_loss = 0.0
            total_entropy = 0.0
            update_count = 0
            
            for _ in range(n_epochs):
                for batch in buffer.get_batches(batch_size):
                    batch_obs, batch_actions, batch_log_probs, batch_states, batch_returns, batch_advantages = batch
                    
                    # 1. 集中式 Critic 更新
                    values = critic(batch_states)
                    critic_loss = nn.MSELoss()(values, batch_returns)
                    
                    critic_opt.zero_grad()
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                    critic_opt.step()
                    
                    # 2. 分佈式 Actor 更新
                    # 我們將 agents 展開 (B * NumAgents, features) 來進行參數共享更新
                    B = batch_obs.size(0)
                    batch_obs_flat = batch_obs.view(B * num_agents, obs_dim)
                    batch_actions_flat = batch_actions.view(B * num_agents, 1)
                    batch_log_probs_flat = batch_log_probs.view(B * num_agents, 1)
                    
                    # 重複擴展 advantages 到所有 agent 上
                    batch_advantages_flat = batch_advantages.repeat_interleave(num_agents, dim=0)
                    
                    new_log_probs_flat, entropy_flat = actor.evaluate_actions(batch_obs_flat, batch_actions_flat)
                    
                    ratio = torch.exp(new_log_probs_flat - batch_log_probs_flat)
                    surr1 = ratio * batch_advantages_flat
                    surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * batch_advantages_flat
                    
                    actor_loss = -torch.min(surr1, surr2).mean() - ent_coef * entropy_flat.mean()
                    
                    actor_opt.zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                    actor_opt.step()
                    
                    total_act_loss += actor_loss.item()
                    total_crit_loss += critic_loss.item()
                    total_entropy += entropy_flat.mean().item()
                    update_count += 1

            # 列印訓練狀態
            avg_act = total_act_loss / update_count
            avg_crit = total_crit_loss / update_count
            avg_ent = total_entropy / update_count
            mean_rew = np.mean(buffer.rewards[:buffer.ptr]) * 600.0  # 估算大約 Episode 獎勵
            print(f"Step: {current_timesteps}/{total_timesteps} | Mean Rew: {mean_rew:.2f} | Act Loss: {avg_act:.4f} | Crit Loss: {avg_crit:.4f} | Ent: {avg_ent:.3f}")
            
            buffer.clear()

        # 5. 定期評估與存檔最佳模型
        if current_timesteps - last_eval_step >= eval_freq:
            last_eval_step = current_timesteps
            actor.eval()
            print("Running periodic evaluation...")
            eval_reward = evaluate(eval_env, actor, n_episodes=n_eval_episodes)
            print(f"Evaluation Mean Reward at Step {current_timesteps}: {eval_reward:.2f}")
            
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save(actor.state_dict(), "models/mappo/best_model.pt")
                print(f"--> Saved new best MAPPO model! (Reward: {eval_reward:.2f})")

    # 6. 儲存最終模型
    torch.save(actor.state_dict(), "models/mappo/final_model.pt")
    print("MAPPO Training completed successfully! Final model saved at models/mappo/final_model.pt")


if __name__ == "__main__":
    main()
