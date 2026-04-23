import sys
import os
import shutil
import warnings
import numpy as np
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "thirdparty"))
sys.path.append(os.getcwd())

import marinenav_env.envs.marinenav_env as marinenav_env
from thirdparty import IQNAgent
from stable_baselines3 import DQN
import APF
import BA
import env_visualizer

def run_comparison():
    # Configuration
    seeds = [42, 101, 2024] # Different maps
    save_dir_iqn = "pretrained_models/IQN/seed_3"
    save_dir_dqn = "pretrained_models/DQN/seed_3/latest_model.zip"
    device = "cpu"
    
    # Common Env Params (matching training/eval config)
    obs_r_range = [1,3]
    start_pos = np.array([5.0,5.0])
    goal_pos = np.array([45.0,45.0])
    num_cores = 6
    num_obs = 8
    
    output_folder = "comparison_results"
    
    for seed in seeds:
        print(f"\nProcessing Seed {seed}...")
        
        # 1. Initialize Visualizer & Environment
        # We assume EnvVisualizer initializes the env with the seed
        ev = env_visualizer.EnvVisualizer(seed=seed, draw_traj=True, dpi=96)
        
        # Configure env settings
        ev.env.reset_start_and_goal = False
        ev.env.random_reset_state = False
        ev.env.set_boundary = True
        ev.env.obs_r_range = obs_r_range
        ev.env.start = start_pos
        ev.env.goal = goal_pos
        ev.env.robot.N = 5
        ev.env.num_cores = num_cores
        ev.env.num_obs = num_obs
        
        # Generate the map
        ev.env.reset()
        
        # 2. Initialize Agents
        print("Loading Agents...")
        
        # IQN
        iqn_agent = IQNAgent(ev.env.get_state_space_dimension(),
                             ev.env.get_action_space_dimension(),
                             device=device,
                             seed=seed)
        iqn_agent.load_model(save_dir_iqn, device)
        
        # DQN
        dqn_agent = DQN.load(save_dir_dqn, device=device)
        
        # APF
        apf_agent = APF.APF_agent(ev.env.robot.a, ev.env.robot.w)
        
        # BA
        ba_agent = BA.BA_agent(ev.env.robot.a, ev.env.robot.w)
        
        agents = {
            "DQN": dqn_agent,
            "Adaptive IQN": iqn_agent,
            "BA": ba_agent,
            "APF": apf_agent
        }
        
        all_actions = {}
        
        # 3. Run Simulation for each Agent
        for name, agent in agents.items():
            print(f"Running {name}...")
            
            # Reset Robot State Only (Preserve Map)
            current_v = ev.env.get_velocity(ev.env.start[0], ev.env.start[1])
            ev.env.robot.reset_state(ev.env.start[0], ev.env.start[1], current_velocity=current_v)
            ev.env.episode_timesteps = 0
            
            obs = ev.env.get_observation()
            done = False
            steps = 0
            max_steps = 1000
            actions = []
            
            while not done and steps < max_steps:
                # Agent Action Selection
                if name == "Adaptive IQN":
                    # Use act_adaptive_eval if available, else act_eval
                    # Calling act_adaptive_eval returns ((action, quantiles, taus), cvar)
                    # We check the method signature in run_experiments.py
                    # (action, quantiles, taus), cvar = agent.act_adaptive_eval(observation)
                    (action, _, _), _ = agent.act_adaptive_eval(obs)
                elif name == "DQN":
                    action, _ = agent.predict(obs, deterministic=True)
                elif name in ["APF", "BA"]:
                    action = agent.act(obs)
                
                # Ensure action is int
                action = int(action)
                actions.append(action)
                
                # Step
                obs, reward, done, info = ev.env.step(action)
                steps += 1
            
            print(f"  Finished {name}: {steps} steps, Result: {info.get('state')}")
            all_actions[name] = actions
            
        # 4. Draw Trajectory
        print("Generating Plot...")
        # ev.draw_trajectory uses 'trajectory_test.png' hardcoded
        # We pass only_ep_actions=False to use our dict
        ev.draw_trajectory(only_ep_actions=False, all_actions=all_actions)
        
        # 5. Move/Rename result
        src = "trajectory_test.png"
        dst = os.path.join(output_folder, f"comparison_seed_{seed}.png")
        if os.path.exists(src):
            if os.path.exists(dst):
                os.remove(dst)
            shutil.move(src, dst)
            print(f"Saved: {dst}")
        else:
            print("Error: Output file not found!")
            
    print("\nAll comparisons finished.")

if __name__ == "__main__":
    run_comparison()
