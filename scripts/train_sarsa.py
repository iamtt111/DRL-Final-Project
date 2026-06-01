import os
import argparse
import numpy as np
from src.envs.elevator_env import HospitalElevatorEnv
from src.agents.sarsa_agent import SarsaAgent

def train_sarsa(episodes=5000, save_path="models/sarsa/sarsa_weights.npz"):
    env = HospitalElevatorEnv()
    agent = SarsaAgent(env=env)
    
    # 設置 Epsilon 探索衰減參數
    start_epsilon = 1.0
    end_epsilon = 0.05
    
    print(f"Training SARSA(λ) for {episodes} episodes...")
    episode_rewards = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        
        # 線性衰減 Epsilon 到 0.05，在 90% 訓練步數時達到最小值
        decay_limit = int(episodes * 0.9)
        if ep < decay_limit:
            agent.epsilon = start_epsilon - ep / decay_limit * (start_epsilon - end_epsilon)
        else:
            agent.epsilon = end_epsilon
            
        action, _ = agent.predict(obs, deterministic=False)
        done = False
        total_reward = 0.0
        
        while not done:
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # ε-greedy 下一步動作
            next_action, _ = agent.predict(next_obs, deterministic=False)
            
            # 更新 SARSA 權重
            agent.update(obs, action, reward, next_obs, next_action, done)
            
            obs = next_obs
            action = next_action
            
        episode_rewards.append(total_reward)
        
        if (ep + 1) % 100 == 0:
            avg_rew = np.mean(episode_rewards[-100:])
            print(f"Episode {ep+1}/{episodes} | Avg Reward (last 100): {avg_rew:.2f} | Epsilon: {agent.epsilon:.4f}")
            
    agent.save(save_path)
    print(f"SARSA training completed. Weights saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SARSA(λ) agent with Tile Coding")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of training episodes")
    parser.add_argument("--save-path", type=str, default="models/sarsa/sarsa_weights.npz", help="Save path for trained weights")
    args = parser.parse_args()
    
    train_sarsa(episodes=args.episodes, save_path=args.save_path)
