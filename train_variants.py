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
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.monitor import Monitor
import marinenav_env.envs.marinenav_env as marinenav_env

print(f"NumPy bool8 patched: {hasattr(np, 'bool8')}")

# Define save directories
SAVE_DIR = "pretrained_models"
LOG_DIR = "logs"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Training Parameters
# 50k steps is enough to show the "process" curve as requested.
# If you want full convergence, increase this.
TOTAL_TIMESTEPS = 50000 
SEED = 42

class SuccessWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # marinenav_env provides info['state'] = 'reach goal'
        is_success = False
        if info.get('state') == 'reach goal':
            is_success = True
        elif info.get('reach_goal'): # Some env versions use boolean
            is_success = True
        
        info['is_success'] = is_success
        return obs, reward, done, info

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

        # We construct separate MLPs for Value and Advantage streams
        # Note: features_dim -> output of extractor (e.g. FlattenExtractor has dim=input_size)
        
        # Value stream (outputs scalar)
        # Using create_mlp from SB3 helps handle layers correctly
        value_net_layers = create_mlp(self.features_dim, 1, self.net_arch, self.activation_fn)
        self.value_net = nn.Sequential(*value_net_layers)
        
        # Advantage stream (outputs action_dim)
        advantage_net_layers = create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn)
        self.advantage_net = nn.Sequential(*advantage_net_layers)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        values = self.value_net(features)
        advantages = self.advantage_net(features)
        # Dueling architecture: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values = self(observation)
        # Greedy action
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
        # Override to create DuelingQNetwork instead of QNetwork
        # We need to make sure we use the same arguments that were prepared in self.net_args
        # (Since we inherit from DQNPolicy which prepares net_args in __init__)
        
        # Depending on SB3 version, self.net_args might or might not be fully populated before _build is called.
        # But _build calls make_q_net.
        # Let's inspect DQNPolicy again... It calls _build in __init__.
        # And _build calls make_q_net.
        # So at this point self.net_args should have been set.
        
        # However, we must call _update_features_extractor to ensure features_extractor is built/passed if needed.
        # DQNPolicy.make_q_net does: 
        # net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        # return QNetwork(**net_args).to(self.device)
        
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DuelingQNetwork(**net_args).to(self.device)

def create_env(seed):
    training_schedule = dict(timesteps=[0, 1000000],
                             num_cores=[4, 8],
                             num_obstacles=[6, 10],
                             min_start_goal_dis=[30.0, 35.0])
    
    env = gym.make('marinenav_env:marinenav_env-v0', seed=seed, schedule=training_schedule)
    return env

def train_d3qn():
    print("Training D3QN (Dueling Double DQN) with custom Policy...")
    env = create_env(SEED)
    
    # Standard architecture for this paper seems to be MLP [256, 256]
    policy_kwargs = dict(net_arch=[256, 256])
    
    model = DQN(DuelingDQNPolicy, env, verbose=1, seed=SEED, policy_kwargs=policy_kwargs)
    
    print(f"Starting training for {TOTAL_TIMESTEPS} steps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    
    save_path = os.path.join(SAVE_DIR, "D3QN", "best_model")
    model.save(save_path)
    print(f"D3QN Trained and Saved to {save_path}")
    env.close()

if __name__ == "__main__":
    def create_monitored_env(seed, log_name):
        log_file = os.path.join(LOG_DIR, log_name) # Monitor needs base path
        env = create_env(seed)
        env = SuccessWrapper(env)
        # We must add custom keys to Monitor to save them
        env = Monitor(env, log_file, info_keywords=("is_success",))
        return env

    print("Training PPO with logging...")
    env_ppo = create_monitored_env(SEED, "PPO_log")
    model_ppo = PPO("MlpPolicy", env_ppo, verbose=1, seed=SEED)
    model_ppo.learn(total_timesteps=TOTAL_TIMESTEPS)
    model_ppo.save(os.path.join(SAVE_DIR, "PPO", "best_model"))
    env_ppo.close()

    print("Training D3QN with logging...")
    env_d3qn = create_monitored_env(SEED, "D3QN_log")
    policy_kwargs = dict(net_arch=[256, 256])
    model_d3qn = DQN(DuelingDQNPolicy, env_d3qn, verbose=1, seed=SEED, policy_kwargs=policy_kwargs)
    model_d3qn.learn(total_timesteps=TOTAL_TIMESTEPS)
    model_d3qn.save(os.path.join(SAVE_DIR, "D3QN", "best_model"))
    env_d3qn.close()

    # train_d3qn()
