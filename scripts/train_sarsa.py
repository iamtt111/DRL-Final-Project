import os
import numpy as np
from src.envs.elevator_env import HospitalElevatorEnv
from src.agents.sarsa_agent import SarsaAgent

def train_sarsa(episodes=30, save_path="models/sarsa/sarsa_weights.npz"):
    env = HospitalElevatorEnv()
    agent = SarsaAgent(env=env)
    
    print(f"Training SARSA(λ) for {episodes} episodes...")
    for ep in range(episodes):
        obs, info = env.reset()
        action, _ = agent.predict(obs, deterministic=False)
        done = False
        
        while not done:
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # ε-greedy next action
            next_action, _ = agent.predict(next_obs, deterministic=False)
            
            # Update SARSA
            agent.update(obs, action, reward, next_obs, next_action, done)
            
            obs = next_obs
            action = next_action
            
        if (ep + 1) % 5 == 0:
            print(f"Episode {ep+1}/{episodes} completed")
            
    agent.save(save_path)
    print(f"SARSA training completed. Weights saved to {save_path}")

if __name__ == "__main__":
    train_sarsa()
