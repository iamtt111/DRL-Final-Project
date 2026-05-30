import pytest
import numpy as np
import tempfile
import os
from src.agents.sarsa_agent import TileCoder, SarsaAgent, deterministic_hash

def test_deterministic_hash():
    h1 = deterministic_hash(1, 2, 3, 100)
    h2 = deterministic_hash(1, 2, 3, 100)
    assert h1 == h2
    assert 0 <= h1 < 100

def test_tile_coder():
    tc = TileCoder(num_tilings=4, bins_per_coord=5, feature_dim=100)
    state = np.random.uniform(-1, 1, 10)
    phi = tc.get_features(state)
    assert phi.shape == (100,)
    # 每個維度在每個 tiling 中只會啟用 1 個特徵，總特徵啟用數上限為 num_tilings * len(state) = 40
    active = np.sum(phi == 1.0)
    assert 0 < active <= 40

def test_sarsa_agent_update():
    agent = SarsaAgent(num_elevators=4, feature_dim=100, learning_rate=0.5, gamma=0.9, lambda_trace=0.5)
    s = np.random.uniform(-1, 1, 10)
    a = 1
    r = 2.0
    s_prime = np.random.uniform(-1, 1, 10)
    a_prime = 2
    
    # 初始權重應該均為 0
    assert np.all(agent.weights == 0.0)
    
    # 執行更新
    td_error = agent.update(s, a, r, s_prime, a_prime, done=False)
    assert td_error == 2.0  # 因為 Q 均為 0，td_error = r + 0 - 0 = r
    
    # 更新後，權重不應均為 0
    assert not np.all(agent.weights == 0.0)

def test_sarsa_save_load():
    agent = SarsaAgent(num_elevators=4, feature_dim=100)
    agent.weights[1, 10] = 5.0
    
    # 使用臨時檔案測試儲存與載入
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sarsa_weights.npz")
        agent.save(path)
        
        new_agent = SarsaAgent(num_elevators=4, feature_dim=100)
        assert new_agent.weights[1, 10] == 0.0
        new_agent.load(path)
        assert new_agent.weights[1, 10] == 5.0
