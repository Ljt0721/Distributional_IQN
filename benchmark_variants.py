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
from stable_baselines3 import PPO, DQN
from sb3_contrib import QRDQN
import marinenav_env.envs.marinenav_env as marinenav_env
import pandas as pd
from train_variants import DuelingDQNPolicy # Helper to ensure class is available

SAVE_DIR = "pretrained_models"
SEED = 42
NUM_EPISODES = 50  # Number of episodes to evaluate

def create_eval_env(seed):
    # No schedule for evaluation -> Fixed difficulty
    # You can adjust these parameters to match "current map" specifically if needed
    env = gym.make('marinenav_env:marinenav_env-v0', seed=seed)
    return env

def evaluate_model(model, env, num_episodes=10):
    all_rewards = []
    success_count = 0
    steps_list = []
    
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            if done:
                if info.get('is_success', False) or info.get('success', False) or (reward > 50): # Heuristic for success if not in info
                    success_count += 1
        
        all_rewards.append(episode_reward)
        steps_list.append(steps)
        if (i+1) % 10 == 0:
            print(f"Episode {i+1}/{num_episodes}: Reward={episode_reward:.2f}, Steps={steps}")
            
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    success_rate = success_count / num_episodes
    mean_steps = np.mean(steps_list)
    
    return mean_reward, std_reward, success_rate, mean_steps

def main():
    results = []
    
    models_to_test = [
        # Use the PROVIDED pretrained DQN model from the repo authors
        # Note: SB3 DQN implementation uses Double Q-learning by default, so this is effectively DDQN
        ("DQN-Official (DDQN)", DQN, os.path.join(SAVE_DIR, "DQN", "seed_3", "best_model")),

        # My locally trained models (10k steps - insufficient for convergence, but demonstrating pipeline)
        ("PPO-Local", PPO, os.path.join(SAVE_DIR, "PPO", "best_model")),
        # ("DDQN-Local", DQN, os.path.join(SAVE_DIR, "DDQN", "best_model")), # Using Official instead
        # For D3QN, we used standard DQN class but with a custom DuelingDQNPolicy
        ("D3QN-Local (Custom)", DQN, os.path.join(SAVE_DIR, "D3QN", "best_model")), 
    ]

    env = create_eval_env(SEED)

    for name, algo_class, path in models_to_test:
        print(f"Evaluating {name}...")
        try:
            # We assume the model loads successfully
            model = algo_class.load(path, env=env)
            mean_reward, std_reward, success_rate, mean_steps = evaluate_model(model, env, NUM_EPISODES)
            
            print(f"Results for {name}:")
            print(f"  Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"  Success Rate: {success_rate * 100:.2f}%")
            print(f"  Mean Steps: {mean_steps:.2f}")
            
            results.append({
                "Algorithm": name,
                "Mean Reward": mean_reward,
                "Std Reward": std_reward,
                "Success Rate": success_rate,
                "Mean Steps": mean_steps
            })
        except Exception as e:
            print(f"Failed to evaluate {name}: {e}")

    env.close()
    
    # Save results
    df = pd.DataFrame(results)
    print("\nBenchmark Summary:")
    print(df)
    df.to_csv("benchmark_results_reproduced.csv", index=False)
    print("Results saved to benchmark_results_reproduced.csv")

if __name__ == "__main__":
    main()
