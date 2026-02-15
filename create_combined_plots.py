import sys
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.patches import Circle

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "thirdparty"))

import marinenav_env.envs.marinenav_env as marinenav_env
from thirdparty import IQNAgent
from stable_baselines3 import DQN
import APF
import BA

def plot_current_field(ax, env):
    """Plot ocean current field as background"""
    x_pos = list(np.linspace(-2.5, env.width + 2.5, 110))
    y_pos = list(np.linspace(-2.5, env.height + 2.5, 110))
    
    pos_x = []
    pos_y = []
    arrow_x = []
    arrow_y = []
    speeds = np.zeros((len(x_pos), len(y_pos)))
    
    for m, x in enumerate(x_pos):
        for n, y in enumerate(y_pos):
            v = env.get_velocity(x, y)
            speed = np.clip(np.linalg.norm(v), 0.1, 10)
            pos_x.append(x)
            pos_y.append(y)
            arrow_x.append(v[0])
            arrow_y.append(v[1])
            speeds[n, m] = np.log(speed)
    
    cmap = cm.Blues(np.linspace(0, 1, 20))
    cmap = mpl.colors.ListedColormap(cmap[10:, :-1])
    
    ax.contourf(x_pos, y_pos, speeds, cmap=cmap, zorder=0)
    ax.quiver(pos_x, pos_y, arrow_x, arrow_y, width=0.001, zorder=1)

def run_agent_simulation(env, agent, agent_name):
    """Run simulation for one agent and return trajectory"""
    # Reset robot state only (preserve map)
    current_v = env.get_velocity(env.start[0], env.start[1])
    env.robot.reset_state(env.start[0], env.start[1], current_velocity=current_v)
    env.episode_timesteps = 0
    
    obs = env.get_observation()
    done = False
    steps = 0
    max_steps = 1000
    total_reward = 0
    
    # Store positions
    robot_positions = [np.array([env.robot.x, env.robot.y])]
    obstacle_positions_initial = [(obs.x, obs.y, obs.r) for obs in env.obstacles]
    
    while not done and steps < max_steps:
        # Agent action selection
        if agent_name == "Adaptive IQN":
            (action, _, _), _ = agent.act_adaptive_eval(obs)
        elif agent_name == "DQN":
            action, _ = agent.predict(obs, deterministic=True)
        elif agent_name in ["APF", "BA"]:
            action = agent.act(obs)
        
        action = int(action)
        
        # Step environment
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # Store positions
        robot_positions.append(np.array([env.robot.x, env.robot.y]))
    
    obstacle_positions_final = [(obs.x, obs.y, obs.r) for obs in env.obstacles]
    
    return np.array(robot_positions), obstacle_positions_initial, obstacle_positions_final, steps, info.get('state'), total_reward

def create_combined_plot(seed, output_folder):
    """Create a single plot with all four algorithms"""
    print(f"\n{'='*60}")
    print(f"Creating combined plot for Seed {seed}")
    print('='*60)
    
    # Initialize Environment
    env = marinenav_env.MarineNavEnv(seed=seed)
    
    # Configure environment
    env.reset_start_and_goal = False
    env.random_reset_state = False
    env.set_boundary = True
    env.obs_r_range = [1, 3]
    env.start = np.array([5.0, 5.0])
    env.goal = np.array([45.0, 45.0])
    env.robot.N = 5
    env.num_cores = 6
    env.num_obs = 8
    env.dynamic_obstacles = True
    
    # Reset environment
    env.reset()
    
    # Load agents
    save_dir_iqn = "pretrained_models/IQN/seed_3"
    save_dir_dqn = "pretrained_models/DQN/seed_3/latest_model.zip"
    device = "cpu"
    
    print("Loading agents...")
    
    # IQN
    iqn_agent = IQNAgent(env.get_state_space_dimension(),
                         env.get_action_space_dimension(),
                         device=device, seed=seed)
    iqn_agent.load_model(save_dir_iqn, device)
    
    # DQN
    dqn_agent = DQN.load(save_dir_dqn, device=device)
    
    # APF and BA
    apf_agent = APF.APF_agent(env.robot.a, env.robot.w)
    ba_agent = BA.BA_agent(env.robot.a, env.robot.w)
    
    agents = {
        "DQN": dqn_agent,
        "Adaptive IQN": iqn_agent,
        "BA": ba_agent,
        "APF": apf_agent
    }
    
    # Colors and styles for each agent
    styles = {
        "DQN": {'color': 'tab:orange', 'linestyle': 'solid', 'linewidth': 2},
        "Adaptive IQN": {'color': 'lime', 'linestyle': 'solid', 'linewidth': 2},
        "BA": {'color': 'red', 'linestyle': 'dashdot', 'linewidth': 2},
        "APF": {'color': 'blue', 'linestyle': 'dashdot', 'linewidth': 2}
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot background current field
    plot_current_field(ax, env)
    
    # Plot boundary
    boundary = np.array([[0, 0], [env.width, 0], [env.width, env.height], [0, env.height], [0, 0]])
    ax.plot(boundary[:, 0], boundary[:, 1], 'r-.', linewidth=3, zorder=2)
    
    # Run simulations and collect results
    all_results = {}
    obstacle_positions_initial = None
    obstacle_positions_final = None
    
    for agent_name, agent in agents.items():
        print(f"Running {agent_name}...")
        
        # Store initial environment state
        env_initial_obstacles = [(obs.x, obs.y, obs.r) for obs in env.obstacles]
        
        trajectory, obs_init, obs_final, steps, result, reward = run_agent_simulation(env, agent, agent_name)
        
        all_results[agent_name] = {
            'trajectory': trajectory,
            'steps': steps,
            'result': result,
            'reward': reward
        }
        
        # Store obstacle positions from first run
        if obstacle_positions_initial is None:
            obstacle_positions_initial = obs_init
            obstacle_positions_final = obs_final
        
        # Reset environment for next agent
        env.obstacles.clear()
        env.reset()
        
        print(f"  {agent_name}: {steps} steps, {result}, reward={reward:.2f}")
    
    # Plot initial obstacle positions (lighter)
    for x, y, r in obstacle_positions_initial:
        circle = Circle((x, y), r, color='magenta', alpha=0.2, zorder=3)
        ax.add_patch(circle)
    
    # Plot final obstacle positions (darker)
    for x, y, r in obstacle_positions_final:
        circle = Circle((x, y), r, color='magenta', alpha=0.7, zorder=4)
        ax.add_patch(circle)
    
    # Plot start and goal
    ax.plot(env.start[0], env.start[1], 'go', markersize=15, label='Start', zorder=10)
    ax.plot(env.goal[0], env.goal[1], 'y*', markersize=20, label='Goal', zorder=10)
    
    # Plot all trajectories
    for agent_name in ["DQN", "Adaptive IQN", "BA", "APF"]:
        result = all_results[agent_name]
        trajectory = result['trajectory']
        style = styles[agent_name]
        
        label = f"{agent_name} ({result['steps']} steps, {result['result']})"
        ax.plot(trajectory[:, 0], trajectory[:, 1], 
                color=style['color'], 
                linestyle=style['linestyle'], 
                linewidth=style['linewidth'],
                label=label, 
                zorder=5)
    
    # Configure plot
    ax.set_xlim([-2.5, env.width + 2.5])
    ax.set_ylim([-2.5, env.height + 2.5])
    ax.set_aspect('equal')
    ax.set_title(f'All Algorithms - Dynamic Obstacles (Seed {seed})', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Save plot
    output_path = os.path.join(output_folder, f"combined_dynamic_seed_{seed}.png")
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nSaved combined plot: {output_path}")
    
    return all_results

def main():
    """Create combined plots for all seeds"""
    seeds = [42, 101, 2024]
    output_folder = "dynamic_results"
    
    for seed in seeds:
        try:
            create_combined_plot(seed, output_folder)
        except Exception as e:
            print(f"Error processing seed {seed}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("All combined plots created successfully!")
    print(f"Results saved to: {os.path.abspath(output_folder)}")
    print("="*60)

if __name__ == "__main__":
    main()
