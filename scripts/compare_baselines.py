import argparse
import numpy as np
import json
import os
from src.envs.elevator_env import HospitalElevatorEnv
from src.agents.ppo_agent import PPOAgent
from src.agents.sarsa_agent import SarsaAgent
from src.agents.rule_based import NearestCarAgent
from src.utils.config_loader import load_config
from src.utils.metrics import generate_comparative_stats
from scripts.evaluate import evaluate_policy

def main():
    parser = argparse.ArgumentParser(description="Benchmark MaskablePPO, SARSA(λ), and Nearest Car")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes per scenario")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    env = HospitalElevatorEnv(config=config)

    scenarios = ["morning_peak", "evening_peak", "mixed_traffic", "disaster_crisis"]
    
    # 定義模型路徑
    ppo_model_path = "models/ppo/best_model.zip"
    if not os.path.exists(ppo_model_path):
        ppo_model_path = "models/ppo/final_model.zip"
        if not os.path.exists(ppo_model_path):
            ppo_model_path = None
            print("Warning: No PPO model found, running untrained PPOAgent (fallback random)")

    sarsa_model_path = "models/sarsa/sarsa_weights.npz"
    if not os.path.exists(sarsa_model_path):
        sarsa_model_path = None
        print("Warning: No SARSA weights found, running untrained SarsaAgent")

    mappo_model_path = "models/mappo/best_model.pt"
    if not os.path.exists(mappo_model_path):
        mappo_model_path = "models/mappo/final_model.pt"
        if not os.path.exists(mappo_model_path):
            mappo_model_path = None
            print("Warning: No MAPPO model found, running untrained MAPPOAgent (fallback random)")

    # 初始化代理人
    ppo_agent = PPOAgent(model_path=ppo_model_path, env=env)
    sarsa_agent = SarsaAgent(env=env)
    if sarsa_model_path:
        sarsa_agent.load(sarsa_model_path)
    rule_agent = NearestCarAgent(env=env)
    
    from src.agents.mappo_agent import MAPPOAgent
    mappo_agent = MAPPOAgent(model_path=mappo_model_path, env=env)

    agents = {
        "MaskablePPO": ppo_agent,
        "SARSA(λ)": sarsa_agent,
        "Nearest Car": rule_agent,
        "MAPPO": mappo_agent
    }

    results = {}

    for scenario in scenarios:
        print(f"\n================ Running scenario: {scenario} ================")
        env.load_scenario(scenario)
        results[scenario] = {}
        
        for name, agent in agents.items():
            print(f"Benchmarking {name} over {args.episodes} episodes...")
            metrics = evaluate_policy(env, agent, n_episodes=args.episodes)
            results[scenario][name] = metrics

    # 輸出統計顯著性檢定
    print("\n\n================ STATISTICAL SIGNIFICANCE TESTS (PPO/MAPPO vs Baselines) ================")
    summary_report = {}
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario}")
        summary_report[scenario] = {}
        
        for eval_name in ["MaskablePPO", "MAPPO"]:
            summary_report[scenario][eval_name] = {}
            eval_awt = results[scenario][eval_name]["raw"]["awt"]
            eval_ert = results[scenario][eval_name]["raw"]["ert"]
            
            for baseline in ["SARSA(λ)", "Nearest Car"]:
                base_awt = results[scenario][baseline]["raw"]["awt"]
                base_ert = results[scenario][baseline]["raw"]["ert"]
                
                # AWT 檢定
                stats_awt = generate_comparative_stats(eval_awt, base_awt)
                # ERT 檢定
                stats_ert = generate_comparative_stats(eval_ert, base_ert)
                
                summary_report[scenario][eval_name][baseline] = {
                    "awt": stats_awt,
                    "ert": stats_ert
                }

                print(f"  {eval_name} vs {baseline}:")
                print(f"    全體等待 (AWT) 均值差: {stats_awt['mean_diff']:.2f}s, p-value: {stats_awt['p_value']:.4e}, Cohen's d: {stats_awt['cohens_d']:.2f}")
                print(f"    急診回應 (ERT) 均值差: {stats_ert['mean_diff']:.2f}s, p-value: {stats_ert['p_value']:.4e}, Cohen's d: {stats_ert['cohens_d']:.2f}")

    # 儲存結果至 JSON
    os.makedirs("docs", exist_ok=True)
    with open("docs/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "scenarios": results,
            "significance": summary_report
        }, f, indent=4)
        
    print("\nBenchmark results and statistical reports saved to docs/benchmark_results.json")

    # 呼叫圖表自動生成 (Phase 4.3)
    try:
        from src.visualization.charts import generate_all_plots
        print("\nGenerating evaluation charts...")
        generate_all_plots("docs/benchmark_results.json")
    except Exception as e:
        print(f"Failed to generate charts: {e}")

if __name__ == "__main__":
    main()
