import sys
import os
import json
import numpy as np
import torch
import warnings
import pandas as pd
import gym
import time

# Add thirdparty to path for imports
sys.path.insert(0, os.path.join(os.getcwd(), "thirdparty"))

# Import Stable Baselines 3
try:
    from stable_baselines3 import PPO, DQN
    from sb3_contrib import QRDQN
except ImportError:
    print("Warning: Could not import stable_baselines3 or sb3_contrib directly. Attempting local import.")
    try:
        from thirdparty.stable_baselines3 import PPO, DQN
        from thirdparty.sb3_contrib import QRDQN
    except ImportError:
        print("Error: Failed to import RL algorithms.")
        sys.exit(1)

# Import IQN Agent
try:
    from IQN.agent import IQNAgent
except ImportError:
    try:
        from thirdparty.IQN.agent import IQNAgent
    except ImportError:
        print("Error: Failed to import IQNAgent.")
        sys.exit(1)

import marinenav_env.envs.marinenav_env as marinenav_env

warnings.filterwarnings("ignore")

# --- Configuration ---
BENCHMARK_FILE = "benchmark_final_results.json"
MODELS_DIR = "pretrained_models"
EPISODES = 100
OBSTACLE_COUNTS = [6, 8, 10]
SEED = 42

# --- Model Paths ---
# Define the models to test. 
# Format: "Name": {"type": "SB3"|"IQN", "class": Class, "path": "path/to/model", "args": {}}
MODELS = {
    "PPO": {
        "type": "SB3",
        "class": PPO,
        "path": os.path.join(MODELS_DIR, "PPO", "seed_42", "final_model.zip")
    },
    "D3QN": {
        "type": "SB3",
        "class": DQN,
        "path": os.path.join(MODELS_DIR, "D3QN", "seed_42", "final_model.zip")
    },
    "Rainbow": {
        "type": "SB3",
        "class": QRDQN,
        "path": os.path.join(MODELS_DIR, "Rainbow", "seed_42", "final_model.zip")
    },
    "AdaptiveIQN": {
        "type": "IQN", 
        "path": os.path.join(MODELS_DIR, "Static_Obs10_AdaptiveIQN", "seed_42", "network_params.pth"),
        "seed_dir": os.path.join(MODELS_DIR, "Static_Obs10_AdaptiveIQN", "seed_42"),
        "adaptive": True
    }
}

# Add standard IQN if available (using seed_3 as found in exploration)
iqn_path = os.path.join(MODELS_DIR, "IQN", "seed_3", "network_params.pth")
if os.path.exists(iqn_path):
    MODELS["IQN"] = {
        "type": "IQN",
        "path": iqn_path,
        "seed_dir": os.path.join(MODELS_DIR, "IQN", "seed_3"),
        "adaptive": False
    }

# --- Helper Functions ---

def load_iqn_model(config, env):
    """Loads an IQN agent with parameters from constructor_params.json if available."""
    path = config["path"]
    seed_dir = config["seed_dir"]
    param_path = os.path.join(seed_dir, "constructor_params.json")
    
    # Default params
    params = {
        "state_size": env.get_state_space_dimension(),
        "action_size": env.get_action_space_dimension(),
        "layer_size": 64,
        "n_step": 1,
        "BATCH_SIZE": 32,
        "BUFFER_SIZE": 1000000,
        "LR": 5e-4,
        "TAU": 1e-3,
        "GAMMA": 0.99,
        "device": "cpu"
    }

    if os.path.exists(param_path):
        try:
            with open(param_path, 'r') as f:
                loaded_params = json.load(f)
            # Update params with loaded values, filtering for valid keys if necessary
            # For simplicity, we trust the param names match broadly or we rely on defaults for missing ones
            # Map specific json keys to __init__ args if names differ
            if "layer_size" in loaded_params: params["layer_size"] = loaded_params["layer_size"]
            if "n_step" in loaded_params: params["n_step"] = loaded_params["n_step"]
            if "gamma" in loaded_params: params["GAMMA"] = loaded_params["gamma"]
            if "tau" in loaded_params: params["TAU"] = loaded_params["tau"]
            if "lr" in loaded_params: params["LR"] = loaded_params["lr"]
        except Exception as e:
            print(f"Warning: Failed to load IQN params from {param_path}: {e}")

    # Initialize Agent
    agent = IQNAgent(
        state_size=params["state_size"],
        action_size=params["action_size"],
        layer_size=params["layer_size"],
        n_step=params["n_step"],
        BATCH_SIZE=params["BATCH_SIZE"],
        BUFFER_SIZE=params["BUFFER_SIZE"],
        LR=params["LR"],
        TAU=params["TAU"],
        GAMMA=params["GAMMA"],
        device="cpu"
    )

    # Load Weights
    if os.path.exists(path):
        try:
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            agent.qnetwork_local.load_state_dict(state_dict)
            agent.qnetwork_local.eval()
        except Exception as e:
            print(f"Error loading IQN weights from {path}: {e}")
            return None
    else:
        print(f"Error: IQN model path not found: {path}")
        return None
        
    return agent

def calculate_energy(env, action_history, trajectory):
    """
    Calculates Energy consumption defined as Work = Force * Distance.
    Energy = Sum( |a_cmd| * distance_segment ) for all segments.
    """
    total_energy = 0.0
    N = env.robot.N
    
    if not trajectory or len(trajectory) == 0:
        return 0.0

    # trajectory contains positions after every sub-step (0.1s)
    # action_history contains action index for every step (1.0s = 10 * 0.1s)
    
    # We iterate through the trajectory points.
    # The trajectory length should be roughly len(action_history) * N
    
    traj_idx = 0
    current_pos = np.array(env.start) # Robot starts here
    # However, env.robot.trajectory[0] is the position AFTER the first sub-step.
    
    # We can iterate through the full trajectory list directly
    # But we need to know which action was active for each segment.
    # Action changes every N sub-steps.
    
    points = [current_pos] + trajectory
    
    # points[0] is start. points[1] is after substep 1.
    # Distance for substep 1 is dist(points[0], points[1]).
    # Action for substep 1 is action_history[0].
    
    # Substep k (1-based) uses action_history[(k-1)//N]
    
    for k in range(1, len(points)):
        p_prev = np.array(points[k-1])
        p_curr = np.array(points[k])
        dist = np.linalg.norm(p_curr - p_prev)
        
        # Determine which action was active
        # k=1 (first substep) -> index 0. action_index = 0 // N = 0.
        action_seq_idx = (k-1) // N
        
        if action_seq_idx < len(action_history):
            action_idx = action_history[action_seq_idx]
            # Retrieve acceleration command (a, w)
            # action_idx might be numpy int, convert to int
            a_cmd, w_cmd = env.robot.actions[int(action_idx)]
            
            # Energy = Force * Distance = |a| * dist (assuming unit mass)
            # User Definition: Energy = F * s
            total_energy += abs(a_cmd) * dist
            
    return total_energy

def evaluate_model(model_name, config, env, n_episodes):
    print(f"  > Loading {model_name}...")
    
    # Load Model
    agent = None
    if config["type"] == "SB3":
        if not os.path.exists(config["path"]):
            print(f"    Error: Model path {config['path']} does not exist.")
            return None
        try:
            agent = config["class"].load(config["path"], device="cpu")
        except Exception as e:
            print(f"    Error loading SB3 model: {e}")
            return None
            
    elif config["type"] == "IQN":
        agent = load_iqn_model(config, env)
        if agent is None:
            return None

    # Run Evaluation
    results = {
        "success_count": 0,
        "time_sum": 0.0,
        "energy_sum": 0.0,
        "episodes": []
    }
    
    print(f"  > Running {n_episodes} episodes...")
    for i in range(n_episodes):
        obs = env.reset()
        done = False
        steps = 0
        
        # Inner Loop
        while not done:
            action = None
            if config["type"] == "SB3":
                action, _ = agent.predict(obs, deterministic=True)
            elif config["type"] == "IQN":
                if config["adaptive"]:
                    (action, _, _), _ = agent.act_adaptive_eval(obs)
                else:
                    action, _, _ = agent.act_eval(obs)
            
            obs, reward, done, info = env.step(int(action))
            steps += 1
        
        # Collect Metrics
        is_success = False
        if info.get('state') == 'reach goal' or info.get('is_success') or info.get('reach_goal'):
            is_success = True
            
        time_taken = steps * env.robot.N * env.robot.dt
        energy = calculate_energy(env, env.robot.action_history, env.robot.trajectory)
        
        results["success_count"] += 1 if is_success else 0
        results["time_sum"] += time_taken
        results["energy_sum"] += energy
        results["episodes"].append({
            "episode": i,
            "success": is_success,
            "time": time_taken,
            "energy": energy
        })
        
        if (i+1) % 20 == 0:
            print(f"    Episode {i+1}/{n_episodes} complete.")

    # Aggregate
    avg_results = {
        "success_rate": results["success_count"] / n_episodes,
        "avg_time": results["time_sum"] / n_episodes,
        "avg_energy": results["energy_sum"] / n_episodes
    }
    
    print(f"  > Result: Success={avg_results['success_rate']:.2f}, Time={avg_results['avg_time']:.2f}, Energy={avg_results['avg_energy']:.2f}")
    return avg_results

def main():
    print("Starting Benchmark...")
    
    # Initialize Environment
    # We use the standard environment but override the generation parameters
    env = gym.make("MarineNavEnv-v0")
    env.seed(SEED)
    env.schedule = None # Disable curriculum to enforce static obstacles
    
    final_data = {}
    
    for obs_count in OBSTACLE_COUNTS:
        print(f"\nExample Group: {obs_count} Obstacles")
        print("="*40)
        
        # Configure Environment
        env.num_obs = obs_count
        # We assume standard num_cores = 8 (default in env)
        env.num_cores = 8 
        
        group_results = {}
        
        for name, config in MODELS.items():
            print(f"\nEvaluating Model: {name}")
            try:
                res = evaluate_model(name, config, env, EPISODES)
                if res:
                    group_results[name] = res
            except Exception as e:
                print(f"FAILED to evaluate {name}: {e}")
                import traceback
                traceback.print_exc()
        
        final_data[f"Obstacles_{obs_count}"] = group_results

    # Save Results
    with open(BENCHMARK_FILE, "w") as f:
        json.dump(final_data, f, indent=4)
        
    # Print Table
    print("\n\n" + "="*60)
    print("FINAL BENCHMARK RESULTS")
    print("="*60)
    
    for group, models in final_data.items():
        print(f"\n{group}:")
        print(f"{'Model':<15} | {'Success':<10} | {'Time':<10} | {'Energy':<10}")
        print("-" * 55)
        for name, stats in models.items():
            print(f"{name:<15} | {stats['success_rate']:<10.2f} | {stats['avg_time']:<10.2f} | {stats['avg_energy']:<10.2f}")
            
    print("="*60)
    print(f"Detailed results saved to {BENCHMARK_FILE}")

if __name__ == "__main__":
    main()
