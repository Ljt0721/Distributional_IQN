
import sys
import os
import shutil
import warnings
import numpy as np
import json
import time as t_module
from datetime import datetime

# Convert to absolute path to avoid import errors
sys.path.insert(0, os.path.join(os.getcwd(), "thirdparty"))

from stable_baselines3 import DQN
from thirdparty import IQNAgent
import APF
import BA
import marinenav_env.envs.marinenav_env as marinenav_env
import copy

# Suppress warnings
warnings.filterwarnings("ignore")

def evaluation_IQN(first_observation, agent, test_env, adaptive:bool=False, cvar=1.0):
    observation = first_observation
    cumulative_reward = 0.0
    length = 0
    done = False
    energy = 0.0
    
    quantiles_data = []
    taus_data = []

    cvars = []
    computation_times = []

    while not done and length < 1000:
        action = None
        if adaptive:
            start = t_module.time()
            (action, quantiles, taus), cvar = agent.act_adaptive_eval(observation)
            end = t_module.time()
            computation_times.append(end-start)
            cvars.append(cvar)
        else:
            start = t_module.time()
            action, quantiles, taus = agent.act_eval(observation, cvar=cvar)
            end = t_module.time()
            computation_times.append(end-start)
            cvars.append(cvar)

        quantiles_data.append(quantiles)
        taus_data.append(taus)
        
        observation, reward, done, info = test_env.step(int(action))
        cumulative_reward += test_env.discount ** length * reward
        length += 1
        energy += test_env.robot.compute_action_energy_cost(int(action))
        
    success = True if info["state"] == "reach goal" else False
    out_of_area = True if info["state"] == "out of boundary" else False
    time = test_env.robot.dt * test_env.robot.N * length

    ep_data = test_env.episode_data()
    # Serialize numpy arrays for JSON
    ep_data["robot"]["actions_cvars"] = [float(c) for c in cvars]
    # Simplify for storage - we don't need full quantile history for metric aggregation
    # ep_data["robot"]["actions_quantiles"] = [x.tolist() for x in quantiles_data]
    # ep_data["robot"]["actions_taus"] = [x.tolist() for x in taus_data]

    return ep_data, success, time, energy, out_of_area, computation_times

def evaluation_DQN(first_observation, agent, test_env):
    observation = first_observation
    cumulative_reward = 0.0
    length = 0
    done = False
    energy = 0.0

    computation_times = []

    while not done and length < 1000:
        start = t_module.time()
        action, _ = agent.predict(observation, deterministic=True)
        end = t_module.time()
        computation_times.append(end-start)
        observation, reward, done, info = test_env.step(int(action))
        cumulative_reward += test_env.discount ** length * reward
        length += 1
        energy += test_env.robot.compute_action_energy_cost(int(action))
        
    success = True if info["state"] == "reach goal" else False
    out_of_area = True if info["state"] == "out of boundary" else False
    time = test_env.robot.dt * test_env.robot.N * length

    return test_env.episode_data(), success, time, energy, out_of_area, computation_times

def evaluation_classical(first_observation, agent, test_env):
    observation = first_observation
    cumulative_reward = 0.0
    length = 0
    done = False
    energy = 0.0

    computation_times = []
    
    while not done and length < 1000:
        start = t_module.time()
        action = agent.act(observation)
        end = t_module.time()
        computation_times.append(end-start)
        observation, reward, done, info = test_env.step(int(action))
        cumulative_reward += test_env.discount ** length * reward
        length += 1
        energy += test_env.robot.compute_action_energy_cost(int(action))

    success = True if info["state"] == "reach goal" else False
    out_of_area = True if info["state"] == "out of boundary" else False
    time = test_env.robot.dt * test_env.robot.N * length

    return test_env.episode_data(), success, time, energy, out_of_area, computation_times

def exp_setup(envs, n_obs, n_cores):
    observations = []
    # Use a new seed for environment generation for each experiment to ensure variety
    # But keep consistent across agents
    
    # We call reset() on the first environment to generate a map
    # Then we copy that map config to others to ensure fair comparison
    
    first_env = envs[0]
    
    # Configure base settings
    first_env.reset_start_and_goal = False
    first_env.random_reset_state = False
    first_env.set_boundary = True
    first_env.obs_r_range = [1,3]
    first_env.start = np.array([5.0,5.0])
    first_env.goal = np.array([45.0,45.0])
    first_env.robot.N = 5
    first_env.num_cores = n_cores
    first_env.num_obs = n_obs
    
    obs0 = first_env.reset()
    observations.append(obs0)
    
    # Copy configuration to other envs
    # We manually set the obstacles and cores to be identical
    for i in range(1, len(envs)):
        env = envs[i]
        env.reset_start_and_goal = False
        env.random_reset_state = False
        env.set_boundary = True
        env.obs_r_range = [1,3]
        env.start = np.array([5.0,5.0])
        env.goal = np.array([45.0,45.0])
        env.robot.N = 5
        env.num_cores = n_cores
        env.num_obs = n_obs
        
        # Manually sync generated map elements
        env.cores = copy.deepcopy(first_env.cores)
        env.core_centers = copy.deepcopy(first_env.core_centers)
        env.obstacles = copy.deepcopy(first_env.obstacles)
        env.obs_centers = copy.deepcopy(first_env.obs_centers)
        
        # Reset robot state only
        current_v = env.get_velocity(env.start[0], env.start[1])
        env.robot.reset_state(env.start[0], env.start[1], current_velocity=current_v)
        env.episode_timesteps = 0
        
        observations.append(env.get_observation())
        
    return observations

def run_benchmark():
    # Settings
    experiment_count = 50  # Enough to be statistically significant
    device = "cpu"
    seed = 2024
    
    save_dir_iqn = "pretrained_models/IQN/seed_3"
    save_dir_dqn = "pretrained_models/DQN/seed_3"
    model_file_dqn = "latest_model.zip"
    
    print(f"Initializing Benchmark ({experiment_count} episodes)...")
    
    # 1. Initialize Environments
    # We need 4 identical environments for 4 agents
    envs = [marinenav_env.MarineNavEnv(seed + i) for i in range(4)]
    
    # 2. Initialize Agents
    # A. Adaptive IQN (Ours)
    print("Loading Adaptive IQN...")
    iqn_agent = IQNAgent(envs[0].get_state_space_dimension(),
                         envs[0].get_action_space_dimension(),
                         device=device,
                         seed=seed)
    iqn_agent.load_model(save_dir_iqn, device)
    
    # B. DQN (Baseline)
    print("Loading DQN...")
    dqn_agent = DQN.load(os.path.join(save_dir_dqn, model_file_dqn), device=device)
    
    # C. APF (Baseline)
    apf_agent = APF.APF_agent(envs[2].robot.a, envs[2].robot.w)
    
    # D. BA (Baseline)
    ba_agent = BA.BA_agent(envs[3].robot.a, envs[3].robot.w)
    
    agents = [iqn_agent, dqn_agent, apf_agent, ba_agent]
    names = ["IQN (Ours)", "DQN", "APF", "BA"]
    eval_funcs = [
        lambda o, a, e: evaluation_IQN(o, a, e, adaptive=True),
        lambda o, a, e: evaluation_DQN(o, a, e),
        lambda o, a, e: evaluation_classical(o, a, e),
        lambda o, a, e: evaluation_classical(o, a, e)
    ]
    
    # Storage
    results = {}
    for name in names:
        results[name] = {
            "success": [],
            "out_of_area": [],
            "time": [],
            "energy": [],
            "computation_times": []
        }
    
    # 3. Run Loop
    # Hard scene: 10 obs, 8 cores
    n_obs = 10
    n_cores = 8
    
    print("Starting loop...")
    for i in range(experiment_count):
        print(f"Running Episode {i+1}/{experiment_count}...")
        observations = exp_setup(envs, n_obs, n_cores)
        
        for j, name in enumerate(names):
            obs = observations[j]
            agent = agents[j]
            env = envs[j]
            func = eval_funcs[j]
            
            try:
                ep_data, success, time, energy, out_of_area, comp_times = func(obs, agent, env)
                
                results[name]["success"].append(int(success))
                results[name]["out_of_area"].append(int(out_of_area))
                
                # Only record metrics if success (standard practice for time/energy comparison)
                # Or record all but filter later. Let's record all for now but mark success.
                results[name]["time"].append(time)
                results[name]["energy"].append(energy)
                results[name]["computation_times"].extend(comp_times)
                
            except Exception as e:
                print(f"\nError running {name} in episode {i}: {e}")
                results[name]["success"].append(0)
                results[name]["out_of_area"].append(0)
                results[name]["time"].append(1000) # Max time penalty
                results[name]["energy"].append(0)

    print("\nBenchmark Complete.")
    
    # 4. Save Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"final_benchmark_results.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")

    # 5. Print Summary
    print("\n=== Benchmark Summary ===")
    print(f"{'Method':<20} | {'Success':<10} | {'Time (s)':<10} | {'Energy':<10}")
    print("-" * 60)
    
    for name in names:
        succ_arr = np.array(results[name]["success"])
        success_rate = np.mean(succ_arr) * 100
        
        # Filter for successful episodes for time/energy
        success_indices = np.where(succ_arr == 1)[0]
        
        if len(success_indices) > 0:
            time_arr = np.array(results[name]["time"])[success_indices]
            energy_arr = np.array(results[name]["energy"])[success_indices]
            avg_time = np.mean(time_arr)
            avg_energy = np.mean(energy_arr)
        else:
            avg_time = float('inf')
            avg_energy = float('inf')
            
        print(f"{name:<20} | {success_rate:.1f}%    | {avg_time:.2f}       | {avg_energy:.2f}")

if __name__ == "__main__":
    run_benchmark()
