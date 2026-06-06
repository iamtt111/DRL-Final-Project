import argparse
import time
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pygame
from src.envs.elevator_env import HospitalElevatorEnv
from src.agents.ppo_agent import PPOAgent
from src.agents.sarsa_agent import SarsaAgent
from src.agents.rule_based import NearestCarAgent
from src.envs.passenger import Passenger
from src.utils.config_loader import load_config

def main():
    parser = argparse.ArgumentParser(description="EGCS Interactive Demo with Pygame")
    parser.add_argument("--agent", type=str, default="rule", choices=["ppo", "sarsa", "rule", "rule_based", "mappo"], help="Agent type")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained model weights")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--scenario", type=str, default="morning_peak", choices=["morning_peak", "evening_peak", "mixed_traffic"], help="Traffic scenario name")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # 確保開啟 Pygame 渲染模式
    env = HospitalElevatorEnv(config=config, render_mode="human")
    env.load_scenario(args.scenario)
    obs, info = env.reset(seed=42)

    # 載入選定的代理人 (對應不同的 args.agent)
    if args.agent == "ppo":
        agent = PPOAgent(model_path=args.model_path, env=env)
    elif args.agent == "sarsa":
        agent = SarsaAgent(env=env)
        if args.model_path:
            agent.load(args.model_path)
    elif args.agent == "mappo":
        from src.agents.mappo_agent import MAPPOAgent
        agent = MAPPOAgent(model_path=args.model_path, env=env)
    else:
        # 支持 "rule" 和 "rule_based" 指派為 NearestCarAgent
        agent = NearestCarAgent(env=env)

    print(f"Starting visual demo using {args.agent} agent...")
    print("Initializing Pygame window... Simulation will start in 3 seconds.")
    time.sleep(3)
    print("Press ESC or close the Pygame window to exit.")
    
    # 全局狀態機與畫面更新控制器
    current_state = "RUNNING"
    injected_emergency = False
    clock = pygame.time.Clock()
    
    try:
        while True:
            # 1. 統一幀率控制器 (每個 Step 限制在每秒 10 幀，提供流暢且可讀的移動速度)
            clock.tick(10)

            # 2. 統一事件處理迴圈，確保使用者可以隨時退出 (按 ESC 或點擊關閉按鈕)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt

            # 3. 狀態機控制
            if current_state == "RUNNING":
                # 於第 20 模擬秒強行注入一個 Level 3 急診任務，展示搶佔與轉向路徑
                current_sim_time = env.building.current_time
                if current_sim_time >= 20.0 and not injected_emergency:
                    print("\n[DEMO] *** 門診警報：於 12 樓注入急診病床 (Level 3) ***")
                    
                    emg_passenger = Passenger(
                        id=999,
                        arrival_time=current_sim_time,
                        origin_floor=12,
                        destination_floor=1,  # 送至 1 樓手術室
                        priority_level=3
                    )
                    
                    preempted = env.priority_system.check_and_apply_preemption(env.building, 12)
                    if preempted:
                        print("[DEMO] 搶佔成功！已指派最近的電梯直達 12 樓，其原本任務已被重新分配。")
                    else:
                        print("[DEMO] 搶佔失敗，所有電梯皆在處理急診或故障。將作為普通呼叫進行指派。")
                        
                    env.building.add_passenger(emg_passenger)
                    env._update_pending_assignments()
                    injected_emergency = True

                # 獲取代理人決策並推進物理環境
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # 若模擬完成，轉移狀態至 SUMMARY，此時不跳出循環而是原地凍結渲染
                if terminated or truncated:
                    current_state = "SUMMARY"
                    print("\n[DEMO] 模擬結束。切換至 SUMMARY 狀態，保持 Pygame 視窗開啟以供檢視 KPI 面板。")
                    print("[DEMO] 按 ESC 鍵或點擊視窗關閉按鈕即可退出。")

            elif current_state == "SUMMARY":
                # 4. 凍結模擬更新，但持續渲染最終畫面和 KPI 面板
                env.render()

    except KeyboardInterrupt:
        print("Demo stopped by user.")
    finally:
        env.close()
        print("Demo closed.")

if __name__ == "__main__":
    main()
