import sys
import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.patches import Circle
import scipy.spatial

# Suppress warnings
warnings.filterwarnings("ignore")

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "thirdparty"))

import marinenav_env.envs.marinenav_env as marinenav_env
from thirdparty import IQNAgent
from stable_baselines3 import DQN
import APF
import BA

def create_animation_and_static_plot(seed, agent_name, agent, output_folder):
    """
    Run simulation with dynamic obstacles and create both GIF animation and static plot
    """
    print(f"\n=== Testing {agent_name} on Seed {seed} ===")
    
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
    env.dynamic_obstacles = True  # Enable dynamic obstacles
    
    # Reset environment
    env.reset()
    obs = env.get_observation()
    
    # Storage for history
    robot_positions = [np.array([env.robot.x, env.robot.y])]
    obstacle_positions_history = []
    
    # Store initial obstacle positions
    initial_obs_pos = [(obs.x, obs.y, obs.r) for obs in env.obstacles]
    obstacle_positions_history.append([(obs.x, obs.y) for obs in env.obstacles])
    
    # Run simulation
    done = False
    steps = 0
    max_steps = 1000
    total_reward = 0
    
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
        obstacle_positions_history.append([(obs.x, obs.y) for obs in env.obstacles])
    
    print(f"  Simulation finished: {steps} steps, Result: {info.get('state')}, Reward: {total_reward:.2f}")
    
    # Convert to arrays
    robot_positions = np.array(robot_positions)
    
    # ===== Create Static Plot =====
    print(f"  Creating static plot...")
    fig_static, ax_static = plt.subplots(figsize=(10, 10))
    
    # Plot background current field
    plot_current_field(ax_static, env)
    
    # Plot boundary
    boundary = np.array([[0, 0], [env.width, 0], [env.width, env.height], [0, env.height], [0, 0]])
    ax_static.plot(boundary[:, 0], boundary[:, 1], 'r-.', linewidth=3)
    
    # Plot initial obstacle positions (lighter)
    for x, y, r in initial_obs_pos:
        circle = Circle((x, y), r, color='magenta', alpha=0.2, label='Initial Obstacles' if x == initial_obs_pos[0][0] else "")
        ax_static.add_patch(circle)
    
    # Plot final obstacle positions (darker)
    for obs in env.obstacles:
        circle = Circle((obs.x, obs.y), obs.r, color='magenta', alpha=0.8)
        ax_static.add_patch(circle)
    
    # Plot start and goal
    ax_static.plot(env.start[0], env.start[1], 'go', markersize=15, label='Start', zorder=10)
    ax_static.plot(env.goal[0], env.goal[1], 'y*', markersize=20, label='Goal', zorder=10)
    
    # Plot robot trajectory
    ax_static.plot(robot_positions[:, 0], robot_positions[:, 1], 'k-', linewidth=2, label=f'{agent_name} Path', zorder=5)
    
    ax_static.set_xlim([-2.5, env.width + 2.5])
    ax_static.set_ylim([-2.5, env.height + 2.5])
    ax_static.set_aspect('equal')
    ax_static.set_title(f'{agent_name} - Dynamic Obstacles (Seed {seed})\nSteps: {steps}, Result: {info.get("state")}', fontsize=14)
    ax_static.legend(loc='upper left', fontsize=10)
    ax_static.set_xticks([])
    ax_static.set_yticks([])
    
    # Save static plot
    static_path = os.path.join(output_folder, f"dynamic_{agent_name.replace(' ', '_')}_seed_{seed}_static.png")
    fig_static.savefig(static_path, dpi=100, bbox_inches='tight')
    plt.close(fig_static)
    print(f"  Saved static plot: {static_path}")
    
    # ===== Create Animation =====
    print(f"  Creating GIF animation (this may take a while)...")
    fig_anim, ax_anim = plt.subplots(figsize=(10, 10))
    
    # Plot background (once)
    plot_current_field(ax_anim, env)
    boundary = np.array([[0, 0], [env.width, 0], [env.width, env.height], [0, env.height], [0, 0]])
    ax_anim.plot(boundary[:, 0], boundary[:, 1], 'r-.', linewidth=3)
    
    # Plot start and goal (once)
    ax_anim.plot(env.start[0], env.start[1], 'go', markersize=15, label='Start', zorder=10)
    ax_anim.plot(env.goal[0], env.goal[1], 'y*', markersize=20, label='Goal', zorder=10)
    
    ax_anim.set_xlim([-2.5, env.width + 2.5])
    ax_anim.set_ylim([-2.5, env.height + 2.5])
    ax_anim.set_aspect('equal')
    ax_anim.set_xticks([])
    ax_anim.set_yticks([])
    
    # Initialize animated elements
    robot_plot, = ax_anim.plot([], [], 'ro', markersize=10, zorder=15)
    trajectory_plot, = ax_anim.plot([], [], 'k-', linewidth=2, alpha=0.7, zorder=5)
    obstacle_patches = []
    title_text = ax_anim.text(0.5, 1.05, '', transform=ax_anim.transAxes, 
                              ha='center', va='top', fontsize=12)
    
    # Initialization function
    def init():
        robot_plot.set_data([], [])
        trajectory_plot.set_data([], [])
        for patch in obstacle_patches:
            patch.remove()
        obstacle_patches.clear()
        return [robot_plot, trajectory_plot, title_text] + obstacle_patches
    
    # Animation function
    def animate(frame):
        # Update robot position
        robot_plot.set_data([robot_positions[frame, 0]], [robot_positions[frame, 1]])
        
        # Update trajectory
        trajectory_plot.set_data(robot_positions[:frame+1, 0], robot_positions[:frame+1, 1])
        
        # Update obstacles
        for patch in obstacle_patches:
            patch.remove()
        obstacle_patches.clear()
        
        for i, (x, y, r) in enumerate(initial_obs_pos):
            # Get current position
            curr_x, curr_y = obstacle_positions_history[frame][i]
            circle = Circle((curr_x, curr_y), r, color='magenta', alpha=0.7, zorder=8)
            ax_anim.add_patch(circle)
            obstacle_patches.append(circle)
        
        # Update title
        title_text.set_text(f'{agent_name} - Step {frame}/{len(robot_positions)-1}')
        
        return [robot_plot, trajectory_plot, title_text] + obstacle_patches
    
    # Create animation (sample every N frames to reduce file size)
    frame_skip = max(1, len(robot_positions) // 100)  # Limit to ~100 frames
    frames = range(0, len(robot_positions), frame_skip)
    
    anim = animation.FuncAnimation(fig_anim, animate, init_func=init, 
                                   frames=frames, interval=50, blit=True, repeat=True)
    
    # Save animation
    gif_path = os.path.join(output_folder, f"dynamic_{agent_name.replace(' ', '_')}_seed_{seed}.gif")
    anim.save(gif_path, writer='pillow', fps=20, dpi=80)
    plt.close(fig_anim)
    print(f"  Saved GIF animation: {gif_path}")
    
    return steps, info.get('state'), total_reward

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

def run_dynamic_tests():
    """Run tests with dynamic obstacles for multiple agents and seeds"""
    
    seeds = [42, 101, 2024]
    output_folder = "dynamic_results"
    
    # Load agents once
    save_dir_iqn = "pretrained_models/IQN/seed_3"
    save_dir_dqn = "pretrained_models/DQN/seed_3/latest_model.zip"
    device = "cpu"
    
    print("Loading agents...")
    
    # Create temporary env to get dimensions
    temp_env = marinenav_env.MarineNavEnv(0)
    
    # IQN
    iqn_agent = IQNAgent(temp_env.get_state_space_dimension(),
                         temp_env.get_action_space_dimension(),
                         device=device, seed=0)
    iqn_agent.load_model(save_dir_iqn, device)
    
    # DQN
    dqn_agent = DQN.load(save_dir_dqn, device=device)
    
    agents = {
        "Adaptive IQN": iqn_agent,
        "DQN": dqn_agent,
        "APF": None,  # Will be created per-env
        "BA": None    # Will be created per-env
    }
    
    results = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Processing Seed {seed}")
        print('='*60)
        
        # Create environment-specific agents for APF and BA
        env_temp = marinenav_env.MarineNavEnv(seed)
        agents["APF"] = APF.APF_agent(env_temp.robot.a, env_temp.robot.w)
        agents["BA"] = BA.BA_agent(env_temp.robot.a, env_temp.robot.w)
        
        for agent_name, agent in agents.items():
            try:
                steps, result, reward = create_animation_and_static_plot(
                    seed, agent_name, agent, output_folder
                )
                results.append({
                    'seed': seed,
                    'agent': agent_name,
                    'steps': steps,
                    'result': result,
                    'reward': reward
                })
            except Exception as e:
                print(f"  ERROR with {agent_name} on seed {seed}: {e}")
                import traceback
                traceback.print_exc()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        print(f"Seed {r['seed']:4d} | {r['agent']:15s} | Steps: {r['steps']:3d} | Result: {r['result']:15s} | Reward: {r['reward']:7.2f}")
    
    print(f"\nAll results saved to: {os.path.abspath(output_folder)}")

if __name__ == "__main__":
    run_dynamic_tests()
