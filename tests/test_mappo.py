import pytest
import numpy as np
import torch
from src.envs.elevator_ma_env import HospitalElevatorMAEnv
from src.agents.mappo_agent import MAPPOAgent, MAPPOActor, MAPPOCritic

def test_mappo_env_reset_and_step():
    # Instantiate environment
    env = HospitalElevatorMAEnv()
    
    # Reset
    obs, infos = env.reset(seed=42)
    assert isinstance(obs, dict)
    assert len(obs) == env.num_elevators
    for agent_id, o in obs.items():
        assert o.shape == (env.local_obs_dim,)
        assert o.dtype == np.float32

    # Step with continuous bids
    actions = {i: [0.5 + 0.1 * i] for i in range(env.num_elevators)}
    next_obs, rewards, terminations, truncations, next_infos = env.step(actions)
    
    assert isinstance(next_obs, dict)
    assert len(next_obs) == env.num_elevators
    assert isinstance(rewards, dict)
    assert len(rewards) == env.num_elevators
    assert isinstance(terminations, dict)
    assert len(terminations) == env.num_elevators
    
    # Assert shared reward
    first_reward = rewards[0]
    for r in rewards.values():
        assert r == first_reward

def test_mappo_agent_predict():
    # Mock environment
    env = HospitalElevatorMAEnv()
    env.reset(seed=42)
    
    # Instantiate agent
    agent = MAPPOAgent(env=env)
    
    # Predict on random observation
    dummy_obs = np.zeros(env.local_obs_dim, dtype=np.float32)
    action, _ = agent.predict(dummy_obs, deterministic=True)
    
    assert isinstance(action, int)
    assert 0 <= action < env.num_elevators

def test_mappo_networks():
    obs_dim = 23
    state_dim = 243
    
    actor = MAPPOActor(obs_dim=obs_dim)
    critic = MAPPOCritic(state_dim=state_dim)
    
    dummy_obs_t = torch.zeros((5, obs_dim))
    dummy_state_t = torch.zeros((5, state_dim))
    
    mu, std = actor(dummy_obs_t)
    assert mu.shape == (5, 1)
    assert std.shape == (5, 1)
    
    action, log_prob = actor.get_action(dummy_obs_t, deterministic=False)
    assert action.shape == (5, 1)
    assert log_prob.shape == (5, 1)
    
    val = critic(dummy_state_t)
    assert val.shape == (5, 1)
