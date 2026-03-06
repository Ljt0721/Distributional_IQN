import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for scientific plotting
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.family'] = 'DejaVu Sans' 

def load_eval_data(file_path, algorithm_label):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
        
    try:
        # Load with allow_pickle=True
        data = np.load(file_path, allow_pickle=True)
        
        # Extract arrays
        timesteps = data['timesteps'] # Shape (N,)
        rewards = data['rewards']     # Shape (N, M) -> M episodes per checkpoint
        successes = data['successes'] # Shape (N, M)
        times = data['times']         # Shape (N, M)
        
        N, M = rewards.shape
        
        # Flatten and create DataFrame
        # Repeat timesteps M times
        # Flatten metrics
        
        # Create long-form data
        # Steps: [t1, t1, ..., t2, t2, ...]
        steps_repeated = np.repeat(timesteps, M)
        
        df = pd.DataFrame({
            'Timestep': steps_repeated,
            'Reward': rewards.flatten(),
            'Success': successes.flatten().astype(float),
            'Time': times.flatten(),
            'Algorithm': algorithm_label
        })
        
        # Scale Timestep to x 10^5
        df['Timestep (x10^5)'] = df['Timestep'] / 100000.0
        
        print(f"Loaded {algorithm_label}: {N} checkpoints x {M} episodes. Max step: {timesteps.max()}")
        return df
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def plot_paper_curves():
    # Define files to load
    files = [
        ("pretrained_models/IQN/seed_3/adaptive_evaluations.npz", "Adaptive IQN"),
        ("pretrained_models/IQN/seed_3/greedy_evaluations.npz", "Greedy IQN"),
        ("pretrained_models/DQN/seed_3/evaluations.npz", "DQN (DDQN)")
    ]
    
    dfs = []
    for path, label in files:
        df = load_eval_data(path, label)
        if df is not None:
            dfs.append(df)
            
    if not dfs:
        print("No data loaded. Exiting.")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    
    # Create the plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # Plot 1: Cumulative Reward
    sns.lineplot(data=full_df, x="Timestep (x10^5)", y="Reward", hue="Algorithm", ax=axes[0], ci=95)
    axes[0].set_ylabel("Cumulative Reward")
    axes[0].set_title("Training Performance: Cumulative Reward")
    
    # Plot 2: Success Rate
    sns.lineplot(data=full_df, x="Timestep (x10^5)", y="Success", hue="Algorithm", ax=axes[1], ci=95)
    axes[1].set_ylabel("Success Rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Training Performance: Success Rate")
    
    # Plot 3: Average Travel Time
    sns.lineplot(data=full_df, x="Timestep (x10^5)", y="Time", hue="Algorithm", ax=axes[2], ci=95)
    axes[2].set_ylabel("Average Travel Time (s)") # Assuming seconds based on dt=1
    axes[2].set_xlabel("Timestep (x $10^5$)")
    axes[2].set_title("Training Performance: Travel Time")
    
    plt.tight_layout()
    output_filename = "paper_training_curves.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    plot_paper_curves()
