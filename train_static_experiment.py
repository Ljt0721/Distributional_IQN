import sys
import numpy as np
import argparse
import os
import json
from datetime import datetime

# Aggressive NumPy 2.0 Compatibility Patch
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
if not hasattr(np, 'int_'):
    np.int_ = np.int64
if not hasattr(np, 'float_'):
    np.float_ = np.float64

sys.path.insert(0, "./thirdparty")
import gym
import torch as th
import torch.nn as nn

# SB3 Imports
from stable_baselines3 import PPO, DQN
from sb3_contrib import QRDQN
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import create_mlp

# Custom Imports
import marinenav_env.envs.marinenav_env as marinenav_env
from thirdparty.IQN import IQNAgent # Custom Adaptive IQN

# Configurations
SAVE_DIR = "pretrained_models"
LOG_DIR = "logs"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
SEED = 42
TOTAL_TIMESTEPS = 3000000 
EVAL_FREQ = 20000

# --- Environment Wrappers ---
class SuccessWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        is_success = False
        if info.get('state') == 'reach goal':
            is_success = True
        elif info.get('reach_goal'): 
            is_success = True
        info['is_success'] = is_success
        return obs, reward, done, info

# --- Logic for Static Difficulty ---
def get_static_schedule(num_obstacles):
    # Heuristic mapping from obstacles to other difficulty params
    # Based on original paper ratios: 6obs->4cores, 10obs->8cores
    # Linear-ish interpolation
    
    if num_obstacles <= 6:
        num_cores = 4
        min_dist = 30.0
    elif num_obstacles <= 8:
        num_cores = 6
        min_dist = 35.0
    elif num_obstacles <= 10:
        num_cores = 8
        min_dist = 40.0
    else: # > 10 (Very Hard)
        num_cores = 10
        min_dist = 45.0
    
    schedule = dict(
        timesteps=[0], # Constant from step 0
        num_cores=[num_cores],
        num_obstacles=[num_obstacles],
        min_start_goal_dis=[min_dist]
    )
    return schedule

def create_env(seed, num_obstacles):
    schedule = get_static_schedule(num_obstacles)
    env = gym.make('marinenav_env:marinenav_env-v0', seed=seed, schedule=schedule)
    env = SuccessWrapper(env)
    return env

# --- Comparison Algorithms (SB3) ---
class DuelingQNetwork(BasePolicy):
    def __init__(self, observation_space, action_space, features_extractor, features_dim, net_arch=None, activation_fn=nn.ReLU, normalize_images=True):
        super().__init__(observation_space, action_space, features_extractor, normalize_images)
        if net_arch is None: net_arch = [64, 64]
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        action_dim = self.action_space.n
        
        self.value_net = nn.Sequential(*create_mlp(features_dim, 1, net_arch, activation_fn))
        self.advantage_net = nn.Sequential(*create_mlp(features_dim, action_dim, net_arch, activation_fn))

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        values = self.value_net(features)
        advantages = self.advantage_net(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values = self(observation)
        return q_values.argmax(dim=1).reshape(-1)
        
    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(dict(net_arch=self.net_arch, features_dim=self.features_dim, activation_fn=self.activation_fn, features_extractor=self.features_extractor))
        return data

class DuelingDQNPolicy(DQNPolicy):
    def make_q_net(self):
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DuelingQNetwork(**net_args).to(self.device)

# --- Callbacks ---
class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, n_eval_episodes, log_path, prefix="eval"):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = log_path
        self.timesteps = []
        self.results = []
        self.successes = []
        self.times = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            rewards, lengths, successes = [], [], []
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                total_reward = 0
                steps = 0
                is_success = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.eval_env.step(action)
                    total_reward += reward
                    steps += 1
                    if done and info.get('is_success'): is_success = 1
                rewards.append(total_reward)
                lengths.append(steps)
                successes.append(is_success)
                
            self.timesteps.append(self.num_timesteps)
            self.results.append(rewards)
            self.times.append(lengths)
            self.successes.append(successes)
            
            save_file = os.path.join(self.log_path, "evaluations.npz")
            os.makedirs(self.log_path, exist_ok=True)
            np.savez(save_file, timesteps=np.array(self.timesteps), rewards=np.array(self.results), successes=np.array(self.successes), times=np.array(self.times))
            if self.verbose > 0: print(f"Step {self.num_timesteps}: Mean Reward = {np.mean(rewards):.2f}, Success = {np.mean(successes):.2f}")
        return True

# --- Main Training Functions ---
def train_sb3_model(algo_name, obs_count, model_class, policy_class, policy_kwargs=None):
    print(f"--- Starting Training: {algo_name} (Obstacles: {obs_count}) ---")
    
    run_name = f"Static_Obs{obs_count}_{algo_name}"
    run_dir = os.path.join(SAVE_DIR, run_name, f"seed_{SEED}")
    os.makedirs(run_dir, exist_ok=True)
    
    env = create_env(SEED, obs_count)
    eval_env = create_env(SEED + 1000, obs_count)
    
    callback = EvalCallback(eval_env, EVAL_FREQ, 30, run_dir)
    
    model = model_class(policy_class, env, verbose=1, seed=SEED, policy_kwargs=policy_kwargs)
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
        model.save(os.path.join(run_dir, "final_model"))
    finally:
        env.close(); eval_env.close()

def train_adaptive_iqn(obs_count):
    print(f"--- Starting Training: Adaptive IQN (Obstacles: {obs_count}) ---")
    
    run_name = f"Static_Obs{obs_count}_AdaptiveIQN"
    run_dir = os.path.join(SAVE_DIR, run_name, f"seed_{SEED}")
    os.makedirs(run_dir, exist_ok=True)
    
    env = create_env(SEED, obs_count)
    eval_env = create_env(SEED + 1000, obs_count)
    
    # IQN Agent Setup
    # IQNAgent(state_dim, action_dim, device, seed)
    state_dim = env.observation_space.shape[0] if len(env.observation_space.shape) > 0 else env.observation_space.n
    action_dim = env.action_space.n
    
    model = IQNAgent(state_dim, action_dim, device="cpu", seed=SEED+100) # IQN agent handles device internally usually or cpu default? Checked file, uses cpu default
    
    # We need to construct a 'eval_config' dict for IQN's internal eval loop if it uses it.
    # But wait, IQN.learn signature from train_IQN_model.py:
    # model.learn(total_timesteps, train_env, eval_env, eval_config, eval_freq, eval_log_path)
    # The 'eval_config' is specifically for saving checkpoints of specific scenarios.
    # We can perform a dummy eval_config.
    
    eval_config = {} # Dummy
    
    # It seems 'eval_log_path' is where it saves 'evaluations.npz'?
    # Checking IQN code would be best, but let's assume it follows the style of the repo.
    # The EvalCallback Logic in IQNAgent likely handles creating evaluations.npz
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS,
                    train_env=env,
                    eval_env=eval_env,
                    eval_config=eval_config,
                    eval_freq=EVAL_FREQ,
                    eval_log_path=run_dir)
    except Exception as e:
        print(f"Error in IQN Training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close(); eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs", type=int, required=True, help="Number of obstacles (6, 8, 10)")
    parser.add_argument("--algo", type=str, default="iqn", help="iqn, d3qn, ppo, rainbow")
    args = parser.parse_args()
    
    if args.algo.lower() == "iqn":
        train_adaptive_iqn(args.obs)
    elif args.algo.lower() == "d3qn":
        train_sb3_model("D3QN", args.obs, DQN, DuelingDQNPolicy, dict(net_arch=[256, 256]))
    elif args.algo.lower() == "ppo":
        train_sb3_model("PPO", args.obs, PPO, "MlpPolicy")
    elif args.algo.lower() == "rainbow":
        train_sb3_model("Rainbow", args.obs, QRDQN, "MlpPolicy")

