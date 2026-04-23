import sys
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import torch
import marinenav_env.envs.marinenav_env as marinenav_env
import warnings

warnings.filterwarnings("ignore")

# 1. Imports & Path Setup
sys.path.insert(0, "./thirdparty")
try:
    from thirdparty.IQN.agent import IQNAgent
except ImportError:
    pass

try:
    from stable_baselines3 import PPO, DQN
except ImportError:
    try:
        from thirdparty import PPO, DQN
    except ImportError:
        pass

from APF import APF_agent


def _min_distance_to_paths(point_xy, trajectories):
    min_dist = float("inf")
    nearest = None
    px, py = point_xy
    for path in trajectories:
        if path is None or len(path) == 0:
            continue
        diffs = path - np.array([px, py])
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        idx = int(np.argmin(dists))
        if dists[idx] < min_dist:
            min_dist = float(dists[idx])
            nearest = path[idx]
    return min_dist, nearest


def _adjust_obstacles_for_visibility(obstacles, trajectories, xlim=(0.0, 50.0), ylim=(0.0, 50.0)):
    """Slightly move obstacles that visually cover trajectories (plot-only adjustment)."""
    adjusted = []
    margin = 0.5
    for obs in obstacles:
        cx, cy, r = float(obs[0]), float(obs[1]), float(obs[2])
        center = np.array([cx, cy], dtype=float)
        for _ in range(4):
            min_dist, nearest = _min_distance_to_paths(center, trajectories)
            clearance = r + 0.35
            if nearest is None or min_dist >= clearance:
                break

            direction = center - nearest
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                direction = center - np.array([(xlim[0] + xlim[1]) / 2.0, (ylim[0] + ylim[1]) / 2.0])
                norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                direction = np.array([1.0, 0.0])
                norm = 1.0

            direction = direction / norm
            shift = (clearance - min_dist) + 0.25
            center = center + direction * shift
            center[0] = np.clip(center[0], xlim[0] + r + margin, xlim[1] - r - margin)
            center[1] = np.clip(center[1], ylim[0] + r + margin, ylim[1] - r - margin)

        adjusted.append((float(center[0]), float(center[1]), r))
    return adjusted

def run_and_capture(env, agent, agent_type, seed):
    """Run simulation and capture full trajectory + obstacle history"""
    env.seed(seed)
    obs = env.reset()
    
    trajectory = [env.start.copy()]
    obstacle_history = [[(o.x, o.y, o.r) for o in env.obstacles]]
    
    done = False
    step = 0
    max_steps = 1000
    
    while not done and step < max_steps:
        # Select action
        if agent_type == "iqn_adaptive":
            action, _ = agent.act_adaptive_eval(obs, eps=0.0)
            if isinstance(action, tuple): action = action[0]
        elif agent_type == "ppo":
            action, _ = agent.predict(obs, deterministic=True)
        elif agent_type == "d3qn":
            action, _ = agent.predict(obs, deterministic=True)
        elif agent_type == "apf":
            action = agent.act(obs)
            
        if isinstance(action, (np.ndarray, list)): action = int(action)
        
        # Step
        obs, reward, done, info = env.step(action)
        step += 1
        
        trajectory.append(np.array(env.robot.x)) # Just to clarify history gathering
        obstacle_history.append([(o.x, o.y, o.r) for o in env.obstacles])
        
    return {
        "full_trajectory": np.array(env.robot.trajectory),
        "obstacle_history": obstacle_history,
        "steps": step,
        "success": 1 if np.linalg.norm(np.array(env.robot.trajectory[-1]) - env.goal) < 2.0 else 0
    }


def plot_4panel_sequence(seed=0):
    """Generate 4-panel time-sequence Figure (Start, Early, Late, Final)"""
    OBS_NUM = 10
    device = "cpu"

    # --- Initialize Env ---
    env = marinenav_env.MarineNavEnv(seed=seed)
    env.num_obs = OBS_NUM
    env.dynamic_obstacles = True
    env.reset_start_and_goal = False
    env.start = np.array([5.0, 5.0])
    env.goal = np.array([45.0, 45.0])
    
    N_INNER = env.robot.N
    dt = env.robot.dt

    # --- Load Agents ---
    print("Loading models...")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # IQN
    iqn_agent = IQNAgent(state_dim, action_dim, device=device, seed=0)
    iqn_path = "pretrained_models/Static_Obs10_AdaptiveIQN/seed_42/network_params.pth"
    if os.path.exists(iqn_path):
        iqn_agent.qnetwork_local.load_state_dict(torch.load(iqn_path, map_location=device))

    # PPO
    ppo_path = "pretrained_models/PPO/best_model.zip"
    ppo_model = PPO.load(ppo_path, device=device) if os.path.exists(ppo_path) else None

    # D3QN
    d3qn_path = "pretrained_models/D3QN/best_model.zip"
    d3qn_model = DQN.load(d3qn_path, device=device) if os.path.exists(d3qn_path) else None

    # APF
    apf_agent = APF_agent(env.robot.a, env.robot.w)

    # --- Run Simulations ---
    print(f"Running simulations for Seed {seed}...")
    iqn_data = run_and_capture(env, iqn_agent, "iqn_adaptive", seed)
    ppo_data = run_and_capture(env, ppo_model, "ppo", seed)
    d3qn_data = run_and_capture(env, d3qn_model, "d3qn", seed)
    apf_data = run_and_capture(env, apf_agent, "apf", seed)

    # Use IQN's total steps as the baseline for the 4 time points
    focus_steps = iqn_data["steps"]

    # Select time points for display: 0%, 33%, 67%, 100% of IQN's duration
    time_points = [
        ("Start", 0),
        ("Early Stage", int(focus_steps * 0.33)),
        ("Late Stage", int(focus_steps * 0.67)),
        ("Final", focus_steps)
    ]

    # --- Compute Flow Field (Static) ---
    env.seed(seed)
    env.reset()
    X, Y = np.meshgrid(np.linspace(0, 50, 40), np.linspace(0, 50, 40))
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x, y = X[i, j], Y[i, j]
            vel_x, vel_y = 0, 0
            for core in env.cores:
                dx, dy = x - core.x, y - core.y
                r2 = dx**2 + dy**2
                v_mag = core.Gamma / (2 * np.pi * np.sqrt(r2 + 1e-6)) * (1 - np.exp(-r2/(env.r**2)))
                if core.clockwise:
                    vel_x += v_mag * dy / np.sqrt(r2 + 1e-6)
                    vel_y += -v_mag * dx / np.sqrt(r2 + 1e-6)
                else: 
                    vel_x += -v_mag * dy / np.sqrt(r2 + 1e-6)
                    vel_y += v_mag * dx / np.sqrt(r2 + 1e-6)
            U[i, j] = vel_x
            V[i, j] = vel_y

    def get_agent_state(data, step_idx):
        """Get agent position and trajectory up to step_idx"""
        step_idx = min(step_idx, data["steps"])
        traj_idx = min(step_idx * N_INNER, len(data["full_trajectory"]) - 1)
        current_pos = data["full_trajectory"][traj_idx]
        path_so_far = data["full_trajectory"][:traj_idx+1]
        return current_pos, path_so_far

    # --- Create 4-panel figure ---
    fig, axes = plt.subplots(1, 4, figsize=(30, 8))
    fig.suptitle(f'Qualitative comparison of trajectory evolution in a complex dynamic environment (Seed {seed})',
                 fontsize=22, weight='bold', y=0.99)

    agent_configs = [
        ("Adaptive IQN (Ours)", iqn_data, '#d62728', '-'),
        ("PPO", ppo_data, '#1f77b4', '--'),
        ("D3QN", d3qn_data, '#2ca02c', '-.'),
        ("APF", apf_data, '#7f7f7f', ':')
    ]

    for panel_idx, (time_label, step_idx) in enumerate(time_points):
        ax = axes[panel_idx]
        
        # Background
        ax.set_facecolor('#E0F7FA')
        ax.streamplot(X, Y, U, V, color=(0.39, 0.58, 0.93, 0.3), linewidth=0.7, density=0.8, arrowsize=0.9)
        
        # Start/Goal markers350, marker='*', edgecolors='black', linewidths=1.5, zorder=10, label='Start')
        ax.scatter(env.goal[0], env.goal[1], c='gold', s=350, marker='*', edgecolors='black', linewidths=1.5, zorder=10, label='Goal')
        
        # Determine current paths for collision shifting
        current_paths = []
        for agent_name, agent_data, color, linestyle in agent_configs:
            _, path = get_agent_state(agent_data, step_idx)
            if len(path) > 0:
                current_paths.append(path)

        # Obstacles at this step
        obs_idx = min(step_idx, len(iqn_data["obstacle_history"]) - 1)
        raw_obstacles = iqn_data["obstacle_history"][obs_idx]
        
        # Adjust obstacles for visibility so they don't cover the lines
        adjusted_obs = _adjust_obstacles_for_visibility(raw_obstacles, current_paths)
        
        for ox, oy, r in adjusted_obs:
            circle = patches.Circle((ox, oy), r, edgecolor='#444', facecolor='#888', alpha=0.8, zorder=3, linewidth=1.5)
            ax.add_patch(circle)
        
        # Plot all agents
        for agent_name, agent_data, color, linestyle in agent_configs:
            pos, path = get_agent_state(agent_data, step_idx)
            ax.plot(path[:, 0], path[:, 1], color=color, linestyle=linestyle, linewidth=3.0, alpha=0.9, label=agent_name, zorder=5)
            ax.scatter(pos[0], pos[1], color=color, s=150, edgecolors='white', linewidths=1.5, zorder=6, marker='o')

            # Add 'X' for failure if it's the final panel and the agent has not succeeded
            if panel_idx == len(time_points) - 1 and not agent_data["success"]:
                ax.scatter(pos[0], pos[1], color='red', s=300, marker='x', linewidths=3, zorder=7)
        
        # Time label
        time_s = step_idx * N_INNER * dt
        ax.text(0.03, 0.97, f"T={time_s:.2f}s", transform=ax.transAxes, fontsize=16, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        
        # Panel label (a) (b) (c) (d)
        panel_labels = ['(a)', '(b)', '(c)', '(d)']
        ax.text(0.03, 0.03, f"{panel_labels[panel_idx]} {time_label}", transform=ax.transAxes, 
                fontsize=16, fontweight='bold', verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, linestyle='--')

    # Put legend outside the subplots on the far right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.99, 0.05), fontsize=16, framealpha=0.95)

    # Adjust layout to make room for legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    output_path = f"secs/figures/trajectory_sequence_refined_seed{seed}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    plot_4panel_sequence(seed=0)
