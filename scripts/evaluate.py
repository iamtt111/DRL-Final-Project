import argparse
import numpy as np
import yaml
import os
from src.envs.elevator_env import HospitalElevatorEnv
from src.agents.ppo_agent import PPOAgent
from src.agents.rule_based import NearestCarAgent
from src.agents.sarsa_agent import SarsaAgent
from src.utils.config_loader import load_config

def evaluate_policy(env, agent, n_episodes: int = 20) -> dict:
    """評估指定代理人在指定環境下的性能"""
    all_awt = []
    all_pwt = []
    all_ert = []
    all_nss = []
    all_eni = []
    all_scr = []
    all_ecr = []
    all_lbi = []
    all_passengers = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        
        # 暫存此 episode 的事件與指標
        passengers_delivered = []
        elevator_starts = [0] * len(env.building.elevators)
        prev_states = [None] * len(env.building.elevators)
        
        # 模擬事件迴圈
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 收集此步產出的乘客運送事件
            step_events = info.get("step_events", [])
            for event in step_events:
                if event.type.value == "passenger_delivered":
                    passengers_delivered.append(event.data)
                    all_passengers.append({
                        "wait_time": float(event.data["wait_time"]),
                        "priority": int(event.data["priority"])
                    })
                elif event.type.value == "elevator_arrived":
                    elev_id = event.data["elevator_id"]
                    elevator_starts[elev_id] += 1

        # 計算本集 KPI
        # 1. 等待時間 (AWT) 與優先權等待時間 (PWT)
        if passengers_delivered:
            awt = np.mean([p["wait_time"] for p in passengers_delivered])
            pwt_list = [p["wait_time"] for p in passengers_delivered if p["priority"] > 0]
            pwt = np.mean(pwt_list) if pwt_list else 0.0
            
            # 急診回應時間 (ERT)
            ert_list = [p["wait_time"] for p in passengers_delivered if p["priority"] == 3]
            ert = np.mean(ert_list) if ert_list else 0.0
            
            # 急診完成率 (ECR - 30秒內送達)
            ecr = sum(1 for p in passengers_delivered if p["priority"] == 3 and p["wait_time"] <= 30.0) / len(ert_list) * 100.0 if ert_list else 100.0
        else:
            awt, pwt, ert, ecr = 0.0, 0.0, 0.0, 100.0

        # 2. 啟停次數 (NSS)
        nss = sum(elevator_starts)
        
        # 3. 能耗指數 (ENI - 簡化計算為運行總高度)
        # 我們直接統計每台電梯在物理運動下的移動總距離
        eni = sum(abs(elev.current_position) for elev in env.building.elevators) / 3.0 # 除以樓層高度

        # 4. 負載均衡度 (LBI - 車廂人數標準差)
        lbi = np.std([elev.current_load for elev in env.building.elevators])

        all_awt.append(awt)
        all_pwt.append(pwt)
        all_ert.append(ert)
        all_nss.append(nss)
        all_eni.append(eni)
        all_ecr.append(ecr)
        all_lbi.append(lbi)

    return {
        "awt": float(np.mean(all_awt)),
        "pwt": float(np.mean(all_pwt)),
        "ert": float(np.mean(all_ert)),
        "nss": float(np.mean(all_nss)),
        "eni": float(np.mean(all_eni)),
        "ecr": float(np.mean(all_ecr)),
        "lbi": float(np.mean(all_lbi)),
        "passengers": all_passengers,
        "raw": {
            "awt": all_awt,
            "pwt": all_pwt,
            "ert": all_ert,
            "nss": all_nss,
            "eni": all_eni,
            "ecr": all_ecr,
            "lbi": all_lbi
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Elevator EGCS policy")
    parser.add_argument("--agent", type=str, required=True, choices=["ppo", "sarsa", "rule", "mappo"], help="Agent type")
    parser.add_argument("--model-path", type=str, default=None, help="Path to trained model weight zip/npz")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes")
    parser.add_argument("--config", type=str, default=None, help="Config yaml path")
    args = parser.parse_args()

    config = load_config(args.config)
    env = HospitalElevatorEnv(config=config)

    # 載入代理人
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
        agent = NearestCarAgent(env=env)

    print(f"Evaluating {args.agent} agent over {args.episodes} episodes...")
    metrics = evaluate_policy(env, agent, n_episodes=args.episodes)

    print("\n================ Evaluation Results ================")
    print(f"全體平均等待時間 (AWT): {metrics['awt']:.2f} s")
    print(f"優先乘客等待時間 (PWT): {metrics['pwt']:.2f} s")
    print(f"急診平均回應時間 (ERT): {metrics['ert']:.2f} s")
    print(f"急診回應完成率   (ECR): {metrics['ecr']:.1f} %")
    print(f"總啟停次數       (NSS): {metrics['nss']:.1f} 次")
    print(f"能耗指數         (ENI): {metrics['eni']:.1f} 樓層·次")
    print(f"負載均衡度       (LBI): {metrics['lbi']:.2f} 人")
    print("====================================================")

if __name__ == "__main__":
    main()
