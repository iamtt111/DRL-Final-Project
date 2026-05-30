import pytest
from src.envs.elevator_env import HospitalElevatorEnv
from src.agents.rule_based import NearestCarAgent
from scripts.evaluate import evaluate_policy

def test_evaluate_policy_runs():
    env = HospitalElevatorEnv()
    agent = NearestCarAgent(env)
    
    # 評估 2 個 Episode，驗證管線是否能正常運作並輸出正確格式
    metrics = evaluate_policy(env, agent, n_episodes=2)
    
    assert "awt" in metrics
    assert "pwt" in metrics
    assert "ert" in metrics
    assert "nss" in metrics
    assert "eni" in metrics
    assert "ecr" in metrics
    assert "lbi" in metrics
    assert "raw" in metrics
    
    assert len(metrics["raw"]["awt"]) == 2
