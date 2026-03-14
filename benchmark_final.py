import sys
import os
import json
import numpy as np

# Aggressive NumPy 2.0 Compatibility Patch
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
if not hasattr(np, 'int_'):
    np.int_ = np.int64
if not hasattr(np, 'float_'):
    np.float_ = np.float64

import torch
import warnings
import pandas as pd
import gym
import time
import shutil

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
        # sys.exit(1) # Don't exit yet, might be IQN only

BENCHMARK_FILE = "benchmark_data_final.json"

# Import IQN Agent
try:
    from IQN.agent import IQNAgent
except ImportError:
    try:
        from thirdparty.IQN.agent import IQNAgent
    except ImportError:
        print("Error: Failed to import IQNAgent.")
        # sys.exit(1)

# Import Improved IQN
try:
    from improved_agent import ImprovedIQNAgent
except ImportError:
    print("Warning: Could not import ImprovedIQNAgent.")

# Import APF and BA Agents
try:
    from APF import APF_agent
    from BA import BA_agent
except ImportError:
    print("Warning: Could not import APF or BA agents.")

MODELS_DIR = "pretrained_models"

EPISODES = 100
OBSTACLE_COUNTS = [6, 8, 10]
SEED = 42

# --- Model Paths ---
# Define the models to test. 
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
    },
    "ImprovedAdaptiveIQN": {
        "type": "ImprovedIQN",
        "path": "improved_adaptive_iqn/improved_model.pth", 
        "adaptive": True
    },
    "DQN": {
        "type": "SB3",
        "class": DQN,
        "path": os.path.join(MODELS_DIR, "DQN", "seed_3", "best_model.zip")
    },
    "DDQN": {
        "type": "SB3",
        "class": DQN, # DDQN is usually DQN with double_q=True, here we just load the weights
        "path": os.path.join(MODELS_DIR, "DDQN", "best_model.zip")
    },
    "APF": {
        "type": "APF",
        "path": None
    },
    "BA": {
        "type": "BA",
        "path": None
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
        "device": "cpu" # Force CPU for evaluation stability
    }

    if os.path.exists(param_path):
        try:
            with open(param_path, 'r') as f:
                loaded_params = json.load(f)
            # Map known keys. Be defensive.
            if "layer_size" in loaded_params: params["layer_size"] = int(loaded_params["layer_size"])
            if "n_step" in loaded_params: params["n_step"] = int(loaded_params["n_step"])
            if "gamma" in loaded_params: params["GAMMA"] = float(loaded_params["gamma"])
            if "tau" in loaded_params: params["TAU"] = float(loaded_params["tau"])
            if "lr" in loaded_params: params["LR"] = float(loaded_params["lr"])
        except Exception as e:
            print(f"Warning: Failed to load IQN params from {param_path}: {e}")

    # Initialize Agent
    try:
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
    except Exception as e:
        print(f"Error initializing IQNAgent: {e}")
        return None

    # Load Weights
    if os.path.exists(path):
        try:
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            
            # Use strict=False to be more robust
            if hasattr(agent, "qnetwork_local"):
               agent.qnetwork_local.load_state_dict(state_dict, strict=False)
               agent.qnetwork_local.eval()
            else:
               print("Agent has no qnetwork_local")
        except:
             # Try deeper keys if direct load fails
            try:
                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    agent.qnetwork_local.load_state_dict(state_dict['model_state_dict'], strict=False)
                elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    agent.qnetwork_local.load_state_dict(state_dict['state_dict'], strict=False)
                else:
                    print("Could not identify state dict structure")
                    return None
                agent.qnetwork_local.eval()
            except Exception as e:
                print(f"Error loading IQN weights: {e}")
                return None
    else:
        print(f"Error: IQN model path not found: {path}")
        return None
        
    return agent

def load_improved_iqn_model(config, env):
    """Loads the new custom ImprovedIQN agent."""
    path = config["path"]
    if not os.path.exists(path):
        print(f"Error: Improved IQN path not found: {path}")
        return None
        
    state_size = env.get_state_space_dimension()
    action_size = env.get_action_space_dimension()
    
    # Initialize with same params used in training
    try:
        agent = ImprovedIQNAgent(state_size, action_size, 
                                layer_size=256, 
                                device="cpu")
        
        agent.qnetwork_local.load_state_dict(torch.load(path, map_location="cpu"))
        agent.qnetwork_local.eval()
        return agent
    except Exception as e:
        print(f"Error loading ImprovedIQN: {e}")
        return None

def calculate_energy_v2(env, action_history, trajectory):
    """
    Calculates Energy consumption defined as Work = Force * Distance.
    Energy = Sum( |a_cmd| * distance_segment ) for all segments.
    """
    total_energy = 0.0
    N = env.robot.N
    
    if not trajectory or len(trajectory) == 0:
        return 0.0

    # Start position from env
    # But for calculation consistency, we use the trajectory points directly.
    # trajectory contains N points per step.
    
    # We need to map each point in trajectory to the action that produced the movement to it.
    # trajectory[k] is the position after the k-th sub-step (0-indexed).
    # The action for sub-step k (0-indexed) is action_history[k // N].
    # The distance for sub-step k is dist(trajectory[k-1], trajectory[k]).
    # For k=0, dist(start, trajectory[0]).
    
    # NOTE: trajectory length matches action_history * N exactly.
    
    current_pos = np.array(env.start)
    
    # Iterate through all points
    for k in range(len(trajectory)):
        next_pos = np.array(trajectory[k])
        dist = np.linalg.norm(next_pos - current_pos)
        
        # Action Index
        step_idx = k // N
        if step_idx < len(action_history):
            action_idx = action_history[step_idx]
            # Retrieve (a, w)
            a_cmd, w_cmd = env.robot.actions[int(action_idx)]
            
            # Energy = |Force| * Distance. Force ~ |a|.
            total_energy += np.abs(a_cmd) * dist
            
        current_pos = next_pos
            
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
            
    elif config["type"] == "ImprovedIQN":
        agent = load_improved_iqn_model(config, env)
        if agent is None:
            return None
            
    elif config["type"] == "APF":
        # Initialize APF
        a = np.array([-0.4, 0.0, 0.4])
        w = np.array([-np.pi/6, 0.0, np.pi/6])
        agent = APF_agent(a, w)
        
    elif config["type"] == "BA":
        # Initialize BA
        a = np.array([-0.4, 0.0, 0.4])
        w = np.array([-np.pi/6, 0.0, np.pi/6])
        agent = BA_agent(a, w)

    # Run Evaluation
    results = {

        "success_count": 0,
        "time_sum": 0.0,
        "energy_sum": 0.0,
        "episodes": []
    }
    
    print(f"  > Running {n_episodes} episodes...")
    start_bench_time = time.time()
    
    for i in range(n_episodes):
        obs = env.reset()
        done = False
        steps = 0
        info = {}
        
        while not done:
            action = None
            if config["type"] == "SB3":
                action, _ = agent.predict(obs, deterministic=True)
            elif config["type"] == "IQN":
                if config["adaptive"]:
                    (action, _, _), _ = agent.act_adaptive_eval(obs)
                else:
                    action, _, _ = agent.act_eval(obs)
            
            elif config["type"] == "ImprovedIQN":
                # Our new agent.act() signature
                # Supports adaptive=True
                action = agent.act(obs, eval_mode=True, adaptive=config["adaptive"])
            
            elif config["type"] == "APF":
                action = agent.act(obs)
                
            elif config["type"] == "BA":
                action = agent.act(obs)

            obs, reward, done, info = env.step(int(action))
            steps += 1
            
            if steps > 1000: # Max steps safeguard
                done = True
        
        # Collect Metrics
        is_success = False
        if info.get('state') == 'reach goal' or info.get('is_success') or info.get('reach_goal'):
            is_success = True
            
        time_taken = steps * env.robot.N * env.robot.dt
        energy = calculate_energy_v2(env, env.robot.action_history, env.robot.trajectory)
        
        results["success_count"] += 1 if is_success else 0
        results["time_sum"] += time_taken
        results["energy_sum"] += energy
        results["episodes"].append({
            "episode": i,
            "success": is_success,
            "time": time_taken,
            "energy": energy
        })
        
        if (i+1) % 25 == 0:
             print(f"    Episode {i+1}/{n_episodes}...")

    end_bench_time = time.time()
    
    # Aggregate
    if n_episodes > 0:
        avg_results = {
            "success_rate": results["success_count"] / n_episodes,
            "avg_time": results["time_sum"] / n_episodes,
            "avg_energy": results["energy_sum"] / n_episodes
        }
    else:
        avg_results = {"success_rate": 0, "avg_time": 0, "avg_energy": 0}
    
    print(f"  > Done in {end_bench_time - start_bench_time:.1f}s. Result: Success={avg_results['success_rate']:.2f}, Time={avg_results['avg_time']:.2f}, Energy={avg_results['avg_energy']:.2f}")
    return avg_results

def main():
    print("Starting Benchmark...")
    
    # Initialize Environment - CORRECTED ID
    try:
        env = gym.make("marinenav_env:marinenav_env-v0")
        env.seed(SEED)
        env.schedule = None 
    except Exception as e:
        print(f"Failed to create environment: {e}")
        return
    
    final_data = {}
    
    for obs_count in OBSTACLE_COUNTS:
        print(f"\n==========================================")
        print(f" Experiment Group: {obs_count} Obstacles")
        print(f"==========================================")
        
        env.num_obs = obs_count
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

    # Save
    with open(BENCHMARK_FILE, "w") as f:
        json.dump(final_data, f, indent=4)
        
    # Print Table
    print("\n\n" + "="*80)
    print("FINAL BENCHMARK RESULTS")
    print("="*80)
    
    print(f"{'Group':<12} | {'Model':<12} | {'Success':<8} | {'Time':<8} | {'Energy':<8}")
    print("-" * 65)
    
    for group, models in final_data.items():
        g_name = group.replace("Obstacles_", "Obs_")
        for name, stats in models.items():
            print(f"{g_name:<12} | {name:<12} | {stats['success_rate']:<8.2f} | {stats['avg_time']:<8.2f} | {stats['avg_energy']:<8.2f}")
            
    print("="*80)
    print(f"Results saved to {BENCHMARK_FILE}")

if __name__ == "__main__":
    main()
