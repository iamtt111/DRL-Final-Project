import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.envs.elevator_ma_env import HospitalElevatorMAEnv
from src.agents.rule_based import NearestCarAgent
from src.agents.mappo_agent import MAPPOActor, get_mappo_obs_for_elevator
from src.utils.config_loader import load_config

def collect_expert_data(env: HospitalElevatorMAEnv, num_decision_steps: int = 50000):
    """
    執行 Nearest Car 專家策略並收集對應的 (observations, expert_actions) 數據
    """
    expert = NearestCarAgent(env=env)
    
    obs_data = []
    action_data = []
    
    collected_steps = 0
    episodes = 0
    
    print(f"Starting data collection... Target decision steps: {num_decision_steps}")
    
    while collected_steps < num_decision_steps:
        obs, infos = env.reset()
        done = False
        episodes += 1
        
        while not done and collected_steps < num_decision_steps:
            # 1. 專家預測離散派梯決策 (選取哪一台電梯)
            # 由於 NearestCarAgent 與單/多代理人通用，我們直接預測
            expert_action, _ = expert.predict(None)
            
            # 2. 獲取當前決策時刻 4 部電梯的局部觀察值 (obs_dim = 23)
            # 在決策時刻，將 4 部電梯的 local observations 記錄下來
            step_obs = []
            for idx in range(env.num_elevators):
                local_obs = get_mappo_obs_for_elevator(env, idx)
                step_obs.append(local_obs)
                
            obs_data.append(step_obs)
            action_data.append(expert_action)
            collected_steps += 1
            
            # 3. 轉化為多代理人環境的投標動作 (被選中的設為 1.0, 其餘 0.0)
            actions = {}
            for idx in range(env.num_elevators):
                actions[idx] = [1.0] if idx == expert_action else [0.0]
                
            # 執行步驟
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            obs = next_obs
            done = all(terminations.values())
            
        if episodes % 10 == 0 or collected_steps >= num_decision_steps:
            print(f"Collected {collected_steps}/{num_decision_steps} steps (Episodes: {episodes})")
            
    return np.array(obs_data, dtype=np.float32), np.array(action_data, dtype=np.int64)

def pretrain_actor():
    # 1. 載入預設組態
    config = load_config()
    env = HospitalElevatorMAEnv(config=config)
    
    # 2. 收集專家資料
    num_steps = 50000
    obs_np, actions_np = collect_expert_data(env, num_decision_steps=num_steps)
    
    print(f"Data collection completed. Obs shape: {obs_np.shape}, Actions shape: {actions_np.shape}")
    
    # 3. 建立 PyTorch Dataset & DataLoader
    obs_t = torch.FloatTensor(obs_np)      # shape: (N, 4, 23)
    actions_t = torch.LongTensor(actions_np) # shape: (N,)
    
    dataset = torch.utils.data.TensorDataset(obs_t, actions_t)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    # 4. 實例化 MAPPOActor 並進行預訓練
    obs_dim = env.local_obs_dim
    actor = MAPPOActor(obs_dim=obs_dim)
    optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    num_epochs = 20
    print(f"Starting Behavior Cloning pretraining for {num_epochs} epochs...")
    
    actor.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_obs, batch_actions in train_loader:
            B = batch_obs.size(0)
            
            # 將批次中 4 台電梯的觀察值展開以輸入參數共享的 Actor
            # batch_obs_flat shape: (B * 4, 23)
            batch_obs_flat = batch_obs.view(B * 4, obs_dim)
            
            # 獲取連續的 bid 分數 -> shape: (B * 4, 1)
            mu, _ = actor(batch_obs_flat)
            
            # 將 bid 分數重新折疊回電梯維度 -> shape: (B, 4)
            bids = mu.view(B, 4)
            
            # 計算交叉熵損失，迫使專家選中的電梯擁有最高的分數
            loss = criterion(bids, batch_actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * B
            
            # 預測準確度統計 (選 bids 最大的那台)
            preds = torch.argmax(bids, dim=1)
            correct += (preds == batch_actions).sum().item()
            total += B
            
        avg_loss = epoch_loss / total
        accuracy = correct / total
        print(f"Epoch {epoch+1:02d}/{num_epochs:02d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy * 100:.2f}%")
        
    # 5. 儲存模型權重
    os.makedirs("models/mappo", exist_ok=True)
    save_path = "models/mappo/pretrain_actor.pt"
    torch.save(actor.state_dict(), save_path)
    print(f"Pretraining completed! Pretrained actor weights saved at: {save_path}")

if __name__ == "__main__":
    pretrain_actor()
