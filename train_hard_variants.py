import sys
import numpy as np

# Aggressive NumPy 2.0 Compatibility Patch
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
if not hasattr(np, 'int_'):
    np.int_ = np.int64
if not hasattr(np, 'float_'):
    np.float_ = np.float64

sys.path.insert(0, "./thirdparty")
import gym
import os
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO, DQN
from sb3_contrib import QRDQN
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import create_mlp
import marinenav_env.envs.marinenav_env as marinenav_env

print(f"NumPy bool8 patched: {hasattr(np, 'bool8')}")

# Define save directories
SAVE_DIR = "pretrained_models"
LOG_DIR = "logs"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Training Parameters
TOTAL_TIMESTEPS = 3000000
EVAL_FREQ = 10000
N_EVAL_EPISODES = 30
SEED = 42

class SuccessWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # marinenav_env provides info['state'] = 'reach goal' or boolean
        is_success = False
        if info.get('state') == 'reach goal':
            is_success = True
        elif info.get('reach_goal'): 
            is_success = True
        
        info['is_success'] = is_success
        return obs, reward, done, info

class EvalCallback(BaseCallback):
    """
    Custom callback to evaluate policy and save results in .npz format 
    compatible with plot_paper_results.py
    """
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
            episode_rewards = []
            episode_lengths = []
            episode_successes = []
            
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
                    
                    if done and info.get('is_success'):
                        is_success = 1
                
                episode_rewards.append(total_reward)
                episode_lengths.append(steps)
                episode_successes.append(is_success)
                
            self.timesteps.append(self.num_timesteps)
            self.results.append(episode_rewards)
            self.times.append(episode_lengths)
            self.successes.append(episode_successes)
            
            # Save .npz
            save_file = os.path.join(self.log_path, "evaluations.npz")
            os.makedirs(self.log_path, exist_ok=True)
            np.savez(save_file,
                     timesteps=np.array(self.timesteps),
                     rewards=np.array(self.results),
                     successes=np.array(self.successes),
                     times=np.array(self.times))
            
            if self.verbose > 0:
                mean_reward = np.mean(episode_rewards)
                mean_success = np.mean(episode_successes)
                print(f"Step {self.num_timesteps}: Mean Reward = {mean_reward:.2f}, Success Rate = {mean_success:.2f}")

        return True

def create_env(seed, difficulty="hard"):
    # Standard schedule for reference
    standard_schedule = dict(timesteps=[0, 1000000],
                             num_cores=[4, 8],
                             num_obstacles=[6, 10],
                             min_start_goal_dis=[30.0, 35.0])
    
    # HARD schedule: More obstacles, more vortices
    hard_schedule = dict(timesteps=[0, 1000000],
                             num_cores=[8, 12],     # Increased
                             num_obstacles=[10, 20],# Increased
                             min_start_goal_dis=[35.0, 40.0]) # Increased slightly

    schedule = hard_schedule if difficulty == "hard" else standard_schedule
    
    env = gym.make('marinenav_env:marinenav_env-v0', seed=seed, schedule=schedule)
    env = SuccessWrapper(env)
    return env

def train_model(algo_name, model_class, policy_class, policy_kwargs=None, difficulty="hard"):
    print(f"--- Starting Training: {algo_name} [{difficulty.upper()}] ---")
    
    # Paths: e.g. pretrained_models/Hard_D3QN/seed_42
    prefix = "Hard_" if difficulty == "hard" else ""
    algo_dir = os.path.join(SAVE_DIR, f"{prefix}{algo_name}", f"seed_{SEED}")
    os.makedirs(algo_dir, exist_ok=True)
    
    # Envs
    env = create_env(SEED, difficulty)
    eval_env = create_env(SEED + 1000, difficulty) # Separate seed for eval
    
    # Callback
    eval_callback = EvalCallback(eval_env, EVAL_FREQ, N_EVAL_EPISODES, algo_dir)
    
    # Model
    print(f"Initializing {algo_name}...")
    model = model_class(policy_class, env, verbose=1, seed=SEED, 
                        policy_kwargs=policy_kwargs)
    
    # Train
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
        model.save(os.path.join(algo_dir, "final_model"))
        print(f"--- Finished Training: {algo_name} ---")
    except KeyboardInterrupt:
        print(f"Training interrupted for {algo_name}. Saving current model...")
        model.save(os.path.join(algo_dir, "interrupted_model"))
    finally:
        env.close()
        eval_env.close()

class DuelingQNetwork(BasePolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        features_extractor,
        features_dim,
        net_arch=None,
        activation_fn=nn.ReLU,
        normalize_images=True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.normalize_images = normalize_images
        action_dim = self.action_space.n

        value_net_layers = create_mlp(self.features_dim, 1, self.net_arch, self.activation_fn)
        self.value_net = nn.Sequential(*value_net_layers)
        
        advantage_net_layers = create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn)
        self.advantage_net = nn.Sequential(*advantage_net_layers)

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
        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

class DuelingDQNPolicy(DQNPolicy):
    def make_q_net(self):
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DuelingQNetwork(**net_args).to(self.device)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("algo", type=str, help="Algorithm to train (d3qn, ppo, rainbow, all)")
    args = parser.parse_args()
    
    target = args.algo.lower()
    difficulty = "hard"

    # --- 1. D3QN (Dueling Double DQN) ---
    if target == "all" or target == "d3qn":
        d3qn_kwargs = dict(net_arch=[256, 256])
        train_model("D3QN", DQN, DuelingDQNPolicy, policy_kwargs=d3qn_kwargs, difficulty=difficulty)
    
    # --- 2. PPO ---
    if target == "all" or target == "ppo":
        train_model("PPO", PPO, "MlpPolicy", difficulty=difficulty)

    # --- 3. Rainbow (approx via QR-DQN) ---
    if target == "all" or target == "rainbow":
        train_model("Rainbow", QRDQN, "MlpPolicy", difficulty=difficulty)
