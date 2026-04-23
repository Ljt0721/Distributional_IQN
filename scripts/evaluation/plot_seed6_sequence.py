import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import torch
import marinenav_env.envs.marinenav_env as marinenav_env
import warnings
import copy

warnings.filterwarnings("ignore")

# 1. Imports & Path Setup
sys.path.insert(0, "./thirdparty")
try:
    from thirdparty.IQN.agent import IQNAgent
except ImportError:
    pass

try:
    from stable_baselines3 import PPO
except ImportError:
    try:
        from thirdparty import PPO
    except ImportError:
        pass

def run_and_capture(env, agent, agent_type, seed):
    # Reset
    env.seed(seed)
    obs = env.reset()
    
    trajectory = [] # List of (x,y) from env.robot.trajectory
    obstacle_history = [] # List of list of (x,y) per step
    
    done = False
    step = 0
    max_steps = 1000
    
    # Capture initial state
    trajectory.append(env.start.copy())
    obstacle_history.append([(o.x, o.y) for o in env.obstacles])
    
    while not done and step < max_steps:
        # Select action
        if agent_type == "iqn":
            res, _ = agent.act_adaptive_eval(obs, eps=0.0)
            action = res[0]
        elif agent_type == "ppo":
            action, _ = agent.predict(obs, deterministic=True)
            
        if isinstance(action, (np.ndarray, list)): action = int(action)
        
        # Step
        obs, reward, done, info = env.step(action)
        step += 1
        
        # Capture state (env.robot.trajectory contains high-res path)
        # We only need to store the obstacle positions at this 'macro' step
        # The robot path is fully stored in env.robot.trajectory
        # But for obstacle history, we only have current positions.
        obstacle_history.append([(o.x, o.y) for o in env.obstacles])
        
    return {
        "full_trajectory": np.array(env.robot.trajectory),
        "obstacle_history": obstacle_history,
        "steps": step,
        "success": 1 if np.linalg.norm(np.array(env.robot.trajectory[-1]) - env.goal) < 2.0 else 0
    }

def plot_sequence(seed=6):
    OBS_NUM = 10
    device = "cpu"
    N_INNER = 10 # Default inner steps per action in marinenav_env usually 10? Need to check.
                 # Actually capturing it from env.robot.N is safer.

    # --- Initialize Env ---
    env = marinenav_env.MarineNavEnv(seed=seed)
    env.num_obs = OBS_NUM
    env.dynamic_obstacles = True
    env.reset_start_and_goal = False
    env.start = np.array([5.0, 5.0])
    env.goal = np.array([45.0, 45.0])
    
    N_INNER = env.robot.N

    # --- Load Agents ---
    print("Loading models...")
    
    # IQN
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    iqn_agent = IQNAgent(state_dim, action_dim, device=device, seed=0)
    iqn_path = "pretrained_models/Static_Obs10_AdaptiveIQN/seed_42/network_params.pth"
    if os.path.exists(iqn_path):
        iqn_agent.qnetwork_local.load_state_dict(torch.load(iqn_path, map_location=device))
        
    # PPO
    ppo_path = "pretrained_models/PPO/best_model.zip"
    ppo_model = PPO.load(ppo_path, device=device) if os.path.exists(ppo_path) else None

    # --- Run Simulations ---
    print("Running Adaptive IQN...")
    iqn_data = run_and_capture(env, iqn_agent, "iqn", seed)
    
    print("Running PPO...")
    ppo_data = run_and_capture(env, ppo_model, "ppo", seed)

    # --- Prepare Plot Grid ---
    # We want 4 snapshots.
    # We'll use the IQN duration to define the checkpoints, 
    # but extend if PPO is longer to show it lagging? 
    # Or just show 0%, 33%, 66%, 100% of the MAX duration.
    
    max_steps = max(iqn_data["steps"], ppo_data["steps"])
    # We have obstacle history length = steps + 1
    
    # Select indices for snapshots (macro steps)
    indices = [
        int(max_steps * 0.25),
        int(max_steps * 0.50),
        int(max_steps * 0.75),
        max_steps - 1 # End
    ]
    # Ensure indices are within bounds for each agent
    # If agent finished earlier, we use its last state.
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(f'Trajectory Evolution (Seed {seed}): Adaptive IQN vs PPO', fontsize=20, weight='bold', y=1.02)
    
    # Pre-calculate Streamlines (Background) - they change over space but static in time? 
    # Yes, cores are static in this setup unless dynamic cores. 
    # Code says: self.cores are initialized in reset. No update_position for cores in step. 
    # So Flow is static.
    
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

    # Helper to get state at step k
    def get_agent_state(data, k):
        # k is macro step index
        # obstacle index is k
        # trajectory index? 
        # trajectory has length (steps * N_INNER) + 1 (approx)
        # Step k corresponds to trajectory index k * N_INNER
        
        # Clamp k to max steps of agent
        k_clamped = min(k, data["steps"])
        
        traj_idx = min(k_clamped * N_INNER, len(data["full_trajectory"]) - 1)
        current_pos = data["full_trajectory"][traj_idx]
        path_so_far = data["full_trajectory"][:traj_idx+1]
        
        return current_pos, path_so_far

    for idx, ax in zip(indices, axes):
        # Setup background
        ax.set_facecolor('#E0F7FA')
        ax.streamplot(X, Y, U, V, color=(0.39, 0.58, 0.93, 0.4), linewidth=0.6, density=0.8, arrowsize=0.8)
        
        # Start/Goal
        ax.scatter(env.start[0], env.start[1], c='lime', s=150, marker='*', edgecolors='black', zorder=10)
        ax.scatter(env.goal[0], env.goal[1], c='gold', s=150, marker='*', edgecolors='black', zorder=10)
        
        # Obstacles at step idx
        # Use IQN's obstacle history (should be same as PPO's for same seed)
        # But we must ensure idx is within bounds.
        # Ideally, simulation continues forever in plots? No, environment stops or persists.
        # We will use the recorded history. If idx > length, use last.
        obs_idx = min(idx, len(iqn_data["obstacle_history"]) - 1)
        obstacles = iqn_data["obstacle_history"][obs_idx]
        
        for (ox, oy) in obstacles:
            # We assume radius is roughly constant or we should have saved it. 
            # Env generator makes random radii. We can grab them from env (they don't change size).
            # We need to map obs index to radius. 
            # Since obstacles list order is preserved, we can take radii from env.obstacles now (since reset preserves order/params for same seed)
            pass
            
        # Re-getting radii
        radii = [o.r for o in env.obstacles]
        for i, (ox, oy) in enumerate(obstacles):
            circle = patches.Circle((ox, oy), radii[i], edgecolor='#444', facecolor='#888', alpha=0.6, zorder=2)
            ax.add_patch(circle)
            
        # Plot Agents
        # PPO
        ppo_pos, ppo_path = get_agent_state(ppo_data, idx)
        ax.plot(ppo_path[:, 0], ppo_path[:, 1], label='PPO', color='#1f77b4', linestyle='--', linewidth=2, alpha=0.8)
        ax.scatter(ppo_pos[0], ppo_pos[1], color='#1f77b4', s=80, edgecolors='white', zorder=6)
        
        # IQN
        iqn_pos, iqn_path = get_agent_state(iqn_data, idx)
        ax.plot(iqn_path[:, 0], iqn_path[:, 1], label='IQN (Ours)', color='#d62728', linestyle='-', linewidth=2.5, alpha=0.9)
        ax.scatter(iqn_pos[0], iqn_pos[1], color='#d62728', s=100, edgecolors='white', zorder=7)

        # Labels
        time_s = idx * N_INNER * env.robot.dt
        ax.set_title(f"Time: {time_s:.1f}s (Step {idx})", fontsize=14, fontweight='bold')
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)
        
        if idx == indices[0]:
            ax.legend(loc='lower left', facecolor='white', framealpha=0.9)

    plt.tight_layout()
    output_path = f"secs/figures/trajectory_sequence_seed{seed}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved sequence plot to {output_path}")

if __name__ == "__main__":
    print("Script started!", flush=True)
    try:
        print("Calling plot_sequence...", flush=True)
        plot_sequence()
    except Exception as e:
        import traceback
        traceback.print_exc()

