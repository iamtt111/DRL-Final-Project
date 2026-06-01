import argparse
import numpy as np
import json
import os
import datetime
from src.envs.elevator_env import HospitalElevatorEnv
from src.agents.sarsa_agent import SarsaAgent
from src.agents.rule_based import NearestCarAgent
from src.utils.config_loader import load_config
from src.utils.metrics import generate_comparative_stats
from scripts.evaluate import evaluate_policy

def main():
    parser = argparse.ArgumentParser(description="Benchmark SARSA(λ), Nearest Car, and MAPPO")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes per scenario")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    env = HospitalElevatorEnv(config=config)
    from src.envs.elevator_ma_env import HospitalElevatorMAEnv
    ma_env = HospitalElevatorMAEnv(config=config)

    scenarios = ["morning_peak", "evening_peak", "mixed_traffic", "disaster_crisis"]

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
    sarsa_agent = SarsaAgent(env=env)
    if sarsa_model_path:
        sarsa_agent.load(sarsa_model_path)
    rule_agent = NearestCarAgent(env=env)
    
    from src.agents.mappo_agent import MAPPOAgent
    mappo_agent = MAPPOAgent(model_path=mappo_model_path, env=ma_env)

    agents = {
        "SARSA(λ)": sarsa_agent,
        "Nearest Car": rule_agent,
        "MAPPO": mappo_agent
    }

    results = {}

    for scenario in scenarios:
        print(f"\n================ Running scenario: {scenario} ================")
        results[scenario] = {}
        
        for name, agent in agents.items():
            print(f"Benchmarking {name} over {args.episodes} episodes...")
            eval_env = ma_env if name == "MAPPO" else env
            eval_env.load_scenario(scenario)
            metrics = evaluate_policy(eval_env, agent, n_episodes=args.episodes)
            results[scenario][name] = metrics

    # 輸出統計顯著性檢定
    print("\n\n================ STATISTICAL SIGNIFICANCE TESTS (PPO/MAPPO vs Baselines) ================")
    summary_report = {}
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario}")
        summary_report[scenario] = {}
        
        for eval_name in ["MAPPO"]:
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
    plot_paths = {}
    try:
        from src.visualization.charts import generate_all_plots
        print("\nGenerating evaluation charts...")
        plot_paths = generate_all_plots("docs/benchmark_results.json")
    except Exception as e:
        print(f"Failed to generate charts: {e}")

    # Generate Markdown Tables
    markdown_tables = []
    for scenario in scenarios:
        sc_data = results.get(scenario, {})
        
        # Build Table
        table = []
        table.append(f"### Scenario: {scenario.replace('_', ' ').title()}")
        table.append("| Metric | MAPPO | SARSA(λ) | Nearest Car |")
        table.append("| :--- | :---: | :---: | :---: |")
        
        metrics_keys = [
            ("AWT (s)", "awt", ".2f"),
            ("ERT (s)", "ert", ".2f"),
            ("ECR (%)", "ecr", ".2f"),
            ("NSS (times)", "nss", ".2f")
        ]
        
        for label, key, fmt in metrics_keys:
            mappo_val = f"{sc_data.get('MAPPO', {}).get(key, 0.0):{fmt}}" if 'MAPPO' in sc_data else "N/A"
            sarsa_val = f"{sc_data.get('SARSA(λ)', {}).get(key, 0.0):{fmt}}" if 'SARSA(λ)' in sc_data else "N/A"
            nearest_val = f"{sc_data.get('Nearest Car', {}).get(key, 0.0):{fmt}}" if 'Nearest Car' in sc_data else "N/A"
            table.append(f"| {label} | {mappo_val} | {sarsa_val} | {nearest_val} |")
            
        table_str = "\n".join(table)
        markdown_tables.append(table_str)
        
    print("\n================ BENCHMARK METRICS SUMMARY TABLES ================")
    for table_str in markdown_tables:
        print(table_str)
        print()
        
    # Generate/Update docs/evaluation_report.md
    report_content = []
    report_content.append("# Scientific Evaluation Report")
    report_content.append(f"*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    report_content.append("\n## Executive Summary")
    report_content.append("This report outlines the scientific evaluation and comparative benchmarking of the Multi-Agent PPO (MAPPO), SARSA(λ), and Nearest Car elevator control algorithms across multiple traffic distribution scenarios.")
    
    report_content.append("\n## Scenario Metrics Comparison")
    for table_str in markdown_tables:
        report_content.append(table_str)
        report_content.append("")
        
    report_content.append("## Visualizations")
    
    if plot_paths:
        if "awt" in plot_paths:
            report_content.append("### 1. Normal vs. Emergency Waiting Time")
            rel_path = plot_paths["awt"].replace("docs/", "")
            report_content.append(f"![Normal vs. Emergency WT]({rel_path})")
            report_content.append("")
            
        if "radar" in plot_paths:
            report_content.append("### 2. Multi-Objective Performance Radar")
            rel_path = plot_paths["radar"].replace("docs/", "")
            report_content.append(f"![Radar Plot]({rel_path})")
            report_content.append("")
            
        if "tradeoff" in plot_paths:
            report_content.append("### 3. Medical AWT vs. ERT Trade-off")
            rel_path = plot_paths["tradeoff"].replace("docs/", "")
            report_content.append(f"![Tradeoff Plot]({rel_path})")
            report_content.append("")
            
        if "boxplot" in plot_paths:
            report_content.append("### 4. Waiting Time Distribution by Passenger Priority")
            rel_path = plot_paths["boxplot"].replace("docs/", "")
            report_content.append(f"![Priority Boxplot]({rel_path})")
            report_content.append("")
            
        if "training" in plot_paths:
            report_content.append("### 5. MAPPO Training Convergence")
            rel_path = plot_paths["training"].replace("docs/", "")
            report_content.append(f"![Training Convergence]({rel_path})")
            report_content.append("")
    else:
        report_content.append("*Warning: No evaluation charts generated.*")
        
    with open("docs/evaluation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_content))
    print("Evaluation report saved to docs/evaluation_report.md")

if __name__ == "__main__":
    main()
