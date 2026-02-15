import sys
import os
import warnings
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

# Add 'thirdparty' to sys.path so that 'stable_baselines3' can be imported as a top-level package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "thirdparty"))

import numpy as np
import marinenav_env.envs.marinenav_env as marinenav_env
from thirdparty import IQNAgent
import env_visualizer

def run_test_and_visualize():
    # Configuration
    seed = 42
    save_dir = "pretrained_models/IQN/seed_3"
    device = "cpu"
    
    print("Initializing Visualizer...")
    # Initialize EnvVisualizer in 'draw_traj' mode (Mode 3 usually implies this usage)
    # Note: We need to pass dpi to match display or file output
    ev = env_visualizer.EnvVisualizer(seed=seed, draw_traj=True, dpi=96)
    
    # --- Customizing Environment ---
    # We need to apply the specific test settings to the visualizer's environment
    print("Configuring Environment...")
    ev.env.reset_start_and_goal = False
    ev.env.random_reset_state = False
    ev.env.set_boundary = True
    ev.env.obs_r_range = [1,3]
    ev.env.start = np.array([5.0,5.0])
    ev.env.goal = np.array([45.0,45.0])
    ev.env.robot.N = 5 
    ev.env.num_cores = 6
    ev.env.num_obs = 10
    
    # Re-reset to generate map with new settings
    ev.env.reset()
    
    # --- Run Agent ---
    print("Initializing IQN Agent...")
    agent = IQNAgent(ev.env.get_state_space_dimension(),
                     ev.env.get_action_space_dimension(),
                     device=device,
                     seed=seed)
    
    print(f"Loading model from {save_dir}...")
    agent.load_model(save_dir, device)

    print("Running Episode...")
    obs = ev.env.get_observation()
    done = False
    steps = 0
    max_steps = 1000
    
    actions_taken = []
    
    while not done and steps < max_steps:
        # Select action
        action, quantiles, taus = agent.act_eval(obs, cvar=1.0)
        
        # Store action for visualization
        actions_taken.append(int(action))
        
        # Step
        obs, reward, done, info = ev.env.step(action)
        steps += 1
        
    print(f"Episode finished. Steps: {steps}, Result: {info.get('state')}")

    # --- Draw Trajectory ---
    print("Generating Trajectory Plot...")
    
    # draw_trajectory expects a dictionary of agent_name -> action_list
    all_actions = {
        "IQN (Test)": actions_taken
    }
    
    # Calls internal plotting and saves to 'trajectory_test.png'
    # We set only_ep_actions=False to force it to use our provided all_actions
    ev.draw_trajectory(only_ep_actions=False, all_actions=all_actions)
    
    save_path = "trajectory_test.png"
    print(f"Plot saved to: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    run_test_and_visualize()
