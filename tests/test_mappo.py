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

    # Verify action masks shape and type
    masks = env.action_masks()
    assert masks.shape == (env.num_elevators, 4)
    assert masks.dtype == bool

    # Step with Discrete(4) motor commands
    actions = {i: 0 for i in range(env.num_elevators)}  # All choose STOP/IDLE
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
    
    # Instantiate agent (obs_dim=83, state_dim=183)
    agent = MAPPOAgent(env=env, obs_dim=env.local_obs_dim)
    
    # Predict on multi-agent observations dict
    obs, _ = env.reset()
    actions, _ = agent.predict(obs, deterministic=True)
    
    assert isinstance(actions, dict)
    assert len(actions) == env.num_elevators
    for i in range(env.num_elevators):
        assert isinstance(actions[i], int)
        assert 0 <= actions[i] < 4

    # Predict on single elevator observation
    dummy_obs = np.zeros(env.local_obs_dim, dtype=np.float32)
    action, _ = agent.predict(dummy_obs, deterministic=True)
    
    assert isinstance(action, int)
    assert 0 <= action < 4

def test_mappo_networks():
    obs_dim = 83
    state_dim = 183
    
    actor = MAPPOActor(obs_dim=obs_dim)
    critic = MAPPOCritic(state_dim=state_dim)
    
    dummy_obs_t = torch.zeros((5, obs_dim))
    dummy_state_t = torch.zeros((5, state_dim))
    dummy_masks_t = torch.ones((5, 4), dtype=torch.bool)
    
    logits = actor(dummy_obs_t)
    assert logits.shape == (5, 4)
    
    action, log_prob = actor.get_action(dummy_obs_t, action_masks=dummy_masks_t, deterministic=False)
    assert action.shape == (5, 1)
    assert log_prob.shape == (5, 1)
    
    # Check action values are valid discrete values
    assert torch.all(action >= 0) and torch.all(action < 4)
    
    log_prob, entropy = actor.evaluate_actions(dummy_obs_t, action, action_masks=dummy_masks_t)
    assert log_prob.shape == (5, 1)
    assert entropy.shape == (5, 1)
    
    val = critic(dummy_state_t)
    assert val.shape == (5, 1)
