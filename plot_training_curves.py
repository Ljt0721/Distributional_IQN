import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set style
sns.set_theme(style="darkgrid")
plt.rcParams['font.family'] = 'DejaVu Sans' # Robust font

LOG_DIR = "logs"
files = {
    "PPO": os.path.join(LOG_DIR, "PPO_log.monitor.csv"),
    "D3QN": os.path.join(LOG_DIR, "D3QN_log.monitor.csv")
}

def load_data(file_path, algorithm_name):
    try:
        # Skip the metadata line (starts with #)
        df = pd.read_csv(file_path, skiprows=1)
        
        # Verify columns
        if 'r' not in df.columns or 'l' not in df.columns:
            print(f"Error: Columns 'r' and 'l' missing in {file_path}")
            return None
            
        # Cumulative timesteps
        df['timesteps'] = df['l'].cumsum()
        df['timesteps_10e5'] = df['timesteps'] / 100000.0
        
        # is_success handling
        # It might be missing if wrapper failed or key name mismatch
        if 'is_success' in df.columns:
            # Convert boolean/string to int (1/0)
            df['is_success'] = df['is_success'].replace({True: 1, False: 0, 'True': 1, 'False': 0}).astype(float)
        else:
            print(f"Warning: 'is_success' column missing in {file_path}. Using reward heuristic > 50.")
            # Heuristic: if reward > 50 (goal reward=100, collision=-50), likely success
            df['is_success'] = (df['r'] > 50).astype(float)
            
        df['Algorithm'] = algorithm_name
        return df
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_and_plot():
    files = {
        "PPO": os.path.join("logs", "PPO_log.monitor.csv"),
        "D3QN": os.path.join("logs", "D3QN_log.monitor.csv")
    }

    dfs = []
    for algo, path in files.items():
        if os.path.exists(path):
            df = load_data(path, algo)
            if df is not None:
                # Smoothing
                # Use expanding window or rolling
                # Rolling is better for trend. expanding is cumulative average.
                # User asked for "curve changing with timestep". Rolling is standard.
                w = 50
                df['Reward'] = df['r'].rolling(w, min_periods=1).mean()
                df['Success Rate'] = df['is_success'].rolling(w, min_periods=1).mean()
                df['Travel Time'] = df['l'].rolling(w, min_periods=1).mean()
                dfs.append(df)
        else:
            print(f"Log file not found: {path} (Skipping {algo})")
            
    if not dfs:
        return

    plot_df = pd.concat(dfs, ignore_index=True)
    
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # 1. Reward
    sns.lineplot(data=plot_df, x="timesteps_10e5", y="Reward", hue="Algorithm", ax=axes[0])
    axes[0].set_ylabel("Average Reward")
    axes[0].set_title("Training Curve: Cumulative Reward")
    
    # 2. Success Rate
    sns.lineplot(data=plot_df, x="timesteps_10e5", y="Success Rate", hue="Algorithm", ax=axes[1])
    axes[1].set_ylabel("Success Rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Training Curve: Success Rate")
    
    # 3. Travel Time
    sns.lineplot(data=plot_df, x="timesteps_10e5", y="Travel Time", hue="Algorithm", ax=axes[2])
    axes[2].set_ylabel("Avg Travel Time (Steps)")
    axes[2].set_xlabel("Timesteps (x 10^5)")
    axes[2].set_title("Training Curve: Average Travel Time")
    
    plt.tight_layout()
    plt.savefig("training_process_curves.png", dpi=300)
    print("Saved training_process_curves.png")

if __name__ == "__main__":
    process_and_plot()
