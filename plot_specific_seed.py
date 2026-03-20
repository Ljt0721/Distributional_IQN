import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
    try:
        from sb3_contrib import QRDQN
    except ImportError:
        QRDQN = None
except ImportError:
    try:
        from thirdparty import PPO, DQN, QRDQN
    except ImportError:
        QRDQN = None

from APF import APF_agent

def calculate_energy(env, action_history, trajectory):
    total_energy = 0.0
    current_pos = np.array(env.start)
    N = env.robot.N
    
    for k in range(len(trajectory)):
        next_pos = np.array(trajectory[k])
        dist = np.linalg.norm(next_pos - current_pos)
        step_idx = k // N
        if step_idx < len(action_history):
            action_idx = action_history[step_idx]
            try:
                a_cmd, w_cmd = env.robot.actions[int(action_idx)]
                total_energy += np.abs(a_cmd) * dist
            except: pass
        current_pos = next_pos
    return total_energy

def run_single_episode(env, agent, type_):
    obs = env.reset()
    done = False
    step = 0
    max_steps = 1000
    
    while not done and step < max_steps:
        if type_ == "iqn_adaptive":
            res, _ = agent.act_adaptive_eval(obs, eps=0.0)
            action = res[0]
        elif type_ == "ppo":
            action, _ = agent.predict(obs, deterministic=True)
        elif type_ == "d3qn" or type_ == "rainbow":
            action, _ = agent.predict(obs, deterministic=True)
        elif type_ == "apf":
            action = agent.act(obs)

        if isinstance(action, (np.ndarray, list)): action = int(action)
        obs, reward, done, info = env.step(action)
        step += 1
    
    traj = np.array(env.robot.trajectory)
    if len(traj) == 0: traj = np.array([[env.start[0], env.start[1]]])
    
    path_len = np.sum(np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1)))
    energy = calculate_energy(env, env.robot.action_history, env.robot.trajectory)
    time_taken = step * env.robot.N * env.robot.dt
    success = 1 if np.linalg.norm(traj[-1] - env.goal) < 2.0 else 0
    
    return {
        "Success": success,
        "Time": time_taken,
        "Energy": energy,
        "Length": path_len,
        "Trajectory": traj
    }

def generate_plot(env, seed, agents_config, output_filename, title_suffix=""):
    print(f"Generating Plot for Seed {seed} -> {output_filename}")
    
    # Run all agents
    final_results = []
    
    for name, agt, typ, col, styl in agents_config:
        if agt is None: continue
        env.seed(seed)
        res = run_single_episode(env, agt, typ)
        final_results.append({**res, "Method": name, "Color": col, "Style": styl})

    # Plot
    env.seed(seed)
    env.reset()
    cores = env.cores
    obstacles = env.obstacles
    
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor('#E0F7FA')
    
    X, Y = np.meshgrid(np.linspace(0, 50, 40), np.linspace(0, 50, 40))
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x, y = X[i, j], Y[i, j]
            vel_x, vel_y = 0, 0
            for core in cores:
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
    ax.streamplot(X, Y, U, V, color=(0.39, 0.58, 0.93, 0.4), linewidth=0.6, density=1.0, arrowsize=0.8)

    for obs in obstacles:
        circle = patches.Circle((obs.x, obs.y), obs.r, edgecolor='#444', facecolor='#888', alpha=0.6, zorder=2)
        ax.add_patch(circle)

    ax.scatter(env.start[0], env.start[1], c='lime', s=250, marker='*', edgecolors='black', zorder=10, label='Start')
    ax.scatter(env.goal[0], env.goal[1], c='gold', s=250, marker='*', edgecolors='black', zorder=10, label='Goal')

    text_str = "Performance Metrics:\n"
    text_str += "-" * 55 + "\n"
    text_str += f"{'Method':<20} | {'Time(s)':<8} | {'Ener(J)':<8} | {'Dist(m)':<8}\n"
    text_str += "-" * 55 + "\n"
    
    order_map = {name: i for i, (name, *_) in enumerate(agents_config)}
    final_results.sort(key=lambda x: order_map.get(x["Method"], 99))

    for res in final_results:
        path = res["Trajectory"]
        ax.plot(path[:, 0], path[:, 1], label=res["Method"], color=res["Color"], linestyle=res["Style"], linewidth=2.5, alpha=0.9, zorder=5)
        
        name = res["Method"]
        if "Adaptive IQN" in name: name = "Adaptive IQN"
        elif "PPO" in name: name = "PPO"
        elif "D3QN" in name: name = "D3QN"
        elif "Rainbow" in name: name = "Rainbow"
        elif "APF" in name: name = "APF"
        
        start_txt = f"{name:<20} | {res['Time']:<8.2f} | {res['Energy']:<8.2f} | {res['Length']:<8.2f}"
        if res["Success"] == 0: start_txt += "(Fail)" 
        text_str += start_txt + "\n"

    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    ax.set_title(f"Navigation Trajectories (Seed {seed})", fontsize=18, fontweight='bold', pad=15)
    ax.legend(loc='lower right', facecolor='white', framealpha=0.9, fontsize=12)
    ax.set_xlabel("X (m)", fontsize=14)
    ax.set_ylabel("Y (m)", fontsize=14)

    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Done. Saved to {output_filename}")

def plot_specific_seed(seed):
    OBS_NUM = 10
    device = "cpu"
    
    # --- Initialize Env ---
    env = marinenav_env.MarineNavEnv(seed=seed)
    env.num_obs = OBS_NUM
    env.dynamic_obstacles = True 
    env.reset_start_and_goal = False 
    env.start = np.array([5.0, 5.0])
    env.goal = np.array([45.0, 45.0])
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # --- Load Agents ---
    print("Loading models...")
    iqn_agent = IQNAgent(state_dim, action_dim, device=device, seed=0)
    iqn_path = "pretrained_models/Static_Obs10_AdaptiveIQN/seed_42/network_params.pth"
    if os.path.exists(iqn_path):
        state_dict = torch.load(iqn_path, map_location=device)
        iqn_agent.qnetwork_local.load_state_dict(state_dict)

    ppo_path = "pretrained_models/PPO/best_model.zip"
    ppo_model = PPO.load(ppo_path, device=device) if os.path.exists(ppo_path) else None
    
    d3qn_path = "pretrained_models/D3QN/best_model.zip"
    d3qn_model = DQN.load(d3qn_path, device=device) if os.path.exists(d3qn_path) else None

    rainbow_path = "pretrained_models/Rainbow/seed_42/final_model.zip"
    rainbow_model = None
    if os.path.exists(rainbow_path):
        try:
            rainbow_model = QRDQN.load(rainbow_path, device=device) if QRDQN else DQN.load(rainbow_path, device=device)
        except: pass

    apf_agent = APF_agent(env.robot.a, env.robot.w)

    agents_config = [
        ("Adaptive IQN (Ours)", iqn_agent, "iqn_adaptive", '#d62728', '-'),
        ("PPO", ppo_model, "ppo", '#1f77b4', '--'),
        ("D3QN", d3qn_model, "d3qn", '#2ca02c', '-.'),
        ("Rainbow", rainbow_model, "rainbow", '#9467bd', (0, (3, 1, 1, 1))),
        ("APF", apf_agent, "apf", '#7f7f7f', ':')
    ]

    output_file = f"secs/figures/trajectory_strict_win_seed{seed}.png"
    generate_plot(env, seed, agents_config, output_file)

if __name__ == "__main__":
    plot_specific_seed(6)
