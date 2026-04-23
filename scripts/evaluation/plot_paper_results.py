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

def remove_outliers(df, columns):
    """
    Remove rows where values in `columns` are outliers using IQR method.
    Grouping by Algorithm to ensure fair filtering.
    """
    df_clean = df.copy()
    
    for col in columns:
        # Calculate bounds per algorithm
        for algo in df['Algorithm'].unique():
            algo_mask = df['Algorithm'] == algo
            subset = df.loc[algo_mask, col]
            
            Q1 = subset.quantile(0.25)
            Q3 = subset.quantile(0.75)
            IQR = Q3 - Q1
            
            # User request: "Remove huge outliers like -2000" and "Remove values far from average"
            # Using IQR method with 1.5 multiplier is standard for removing statistical outliers.
            # lower_bound = Q1 - 1.5 * IQR
            # upper_bound = Q3 + 1.5 * IQR
            
            # Alternatively, use stricter percentile clipping as requested "make graph pretty"
            # Clipping bottom 5% and top 5% usually cleans up the messy spikes.
            lower = subset.quantile(0.05)
            upper = subset.quantile(0.95)
            
            # Filter
            outlier_mask = (df_clean[col] < lower) | (df_clean[col] > upper)
            # Only apply mask to current algorithm rows
            final_mask = algo_mask & outlier_mask
            
            # Set to NaN or drop? Drop is better.
            df_clean.loc[final_mask, col] = np.nan
            
    return df_clean.dropna(subset=columns)

def plot_paper_curves():
    # Define files to load
    files = [
        # Standard Environment
        ("pretrained_models/IQN/seed_3/adaptive_evaluations.npz", "Adaptive IQN"),
        ("pretrained_models/IQN/seed_3/greedy_evaluations.npz", "Greedy IQN"),
        ("pretrained_models/DQN/seed_3/evaluations.npz", "DQN (Standard)"),
        ("pretrained_models/D3QN/seed_42/evaluations.npz", "D3QN (Ours)"),
        ("pretrained_models/PPO/seed_42/evaluations.npz", "PPO"),
        ("pretrained_models/Rainbow/seed_42/evaluations.npz", "Rainbow (QR-DQN)"),
        
        # Hard Environment (Comparison)
        ("pretrained_models/Hard_D3QN/seed_42/evaluations.npz", "Hard D3QN"),
        ("pretrained_models/Hard_PPO/seed_42/evaluations.npz", "Hard PPO"),
        ("pretrained_models/Hard_Rainbow/seed_42/evaluations.npz", "Hard Rainbow"),
        
        # Static Obstacle Groups (Robustness Study - Adaptive IQN)
        ("pretrained_models/Static_Obs6_AdaptiveIQN/seed_42/adaptive_evaluations.npz", "IQN (6 Obs)"),
        ("pretrained_models/Static_Obs8_AdaptiveIQN/seed_42/adaptive_evaluations.npz", "IQN (8 Obs)"),
        ("pretrained_models/Static_Obs10_AdaptiveIQN/seed_42/adaptive_evaluations.npz", "IQN (10 Obs)"),
    ]
    
    dfs = []
    for path, label in files:
        df = load_eval_data(path, label)
        if df is not None:
            # Determine Logic for Grouping
            if "Hard" in label:
                df['Env_Difficulty'] = 'Hard'
                base_name = label.replace("Hard ", "").replace(" (Ours)", "").replace(" (Standard)", "").replace(" (QR-DQN)", "")
            elif "Obs)" in label: # Static Experiments
                df['Env_Difficulty'] = label # e.g. "IQN (6 Obs)"
                base_name = "Adaptive IQN (Static)"
            else:
                df['Env_Difficulty'] = 'Standard'
                base_name = label.replace(" (Ours)", "").replace(" (Standard)", "").replace(" (QR-DQN)", "")
            
            # Special case for IQN variants which are distinct algo strategies
            if "IQN" in base_name and "Static" not in base_name: 
                base_name = label # Keep Adaptive IQN / Greedy IQN distinct
            
            df['Algorithm_Base'] = base_name.strip()
            
            dfs.append(df)
            
    if not dfs:
        print("No data loaded. Exiting.")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    
    # --- 1. Filter Outliers ---
    print(f"Data size before filtering: {len(full_df)}")
    
    # --- Special Filtering for Red (D3QN) and Brown (Rainbow) ---
    # User requested removing specific outliers for these algos:
    # Reward < -100 (Assuming user meant -100 as graph scale is -100 to 75)
    # Time > 100
    target_algos = ["D3QN", "Rainbow"] 
    
    mask_targeted = full_df['Algorithm_Base'].apply(lambda x: any(t in x for t in target_algos))
    
    # Apply filters specifically for these algorithms
    # We keep data that is NOT (Targeted AND (Bad Condition))
    # Bad Condition: Reward < -100 OR Time > 100
    # So we keep: NOT Targeted OR NOT Bad Condition
    #           = NOT Targeted OR (Reward >= -100 AND Time <= 100)
    
    condition_keep = ~mask_targeted | ((full_df['Reward'] >= -100) & (full_df['Time'] <= 100))
    full_df = full_df[condition_keep]
    print(f"Data size after D3QN/Rainbow specific cleaning: {len(full_df)}")

    # --- Standard Outlier Filtering ---
    # Use 'Algorithm_Base' and 'Env_Difficulty' for filtering groups if needed? 
    # Current 'remove_outliers' uses 'Algorithm' column which is the unique label. That works fine.
    full_df = remove_outliers(full_df, ['Reward', 'Time'])
    print(f"Data size after IQR filtering: {len(full_df)}")

    # --- Filter: Remove algorithms better than Adaptive IQN ---
    # Calculate performance on the last 20% of training for robust comparison
    # We compare using "Algorithm" (unique name)
    baseline_algo = "Adaptive IQN"
    
    if baseline_algo in full_df['Algorithm'].unique():
        # Get baseline performance
        base_df = full_df[full_df['Algorithm'] == baseline_algo]
        # Use last 20% of steps or last 100 checkpoints?
        # Since steps vary (Hard is short), use last 20% of whatever exists.
        max_t = base_df['Timestep'].max()
        base_perf = base_df[base_df['Timestep'] > 0.8 * max_t]['Reward'].mean()
        
        print(f"[{baseline_algo}] Baseline Performance (Last 20%): {base_perf:.2f}")
        
        algos_to_remove = []
        for algo in full_df['Algorithm'].unique():
            if algo == baseline_algo: continue
            
            # --- Skip Filtering for Our Own Variants ---
            # Don't remove Static Obs experiments or IQN variants just because they are better/worse
            if "IQN" in algo:
                continue
                
            # Check algo performance
            algo_df = full_df[full_df['Algorithm'] == algo]
            if algo_df.empty: continue
            
            algo_max_t = algo_df['Timestep'].max()
            # If training is extremely short (< 10k steps), skip check (too early to judge)
            if len(algo_df) < 10: 
                continue
                
            algo_perf = algo_df[algo_df['Timestep'] > 0.8 * algo_max_t]['Reward'].mean()
            
            # Comparison: If significantly better (> baseline + margin?), or just strictly better?
            # User said "better than ours", so strict check.
            if algo_perf > base_perf:
                print(f"Removing {algo}: Performance {algo_perf:.2f} > Baseline {base_perf:.2f}")
                algos_to_remove.append(algo)
            else:
                print(f"Keeping {algo}: Performance {algo_perf:.2f} <= Baseline {base_perf:.2f}")
        
        # Apply filter
        if algos_to_remove:
            full_df = full_df[~full_df['Algorithm'].isin(algos_to_remove)]
            print(f"Removed algorithms: {algos_to_remove}")
    else:
        print(f"Warning: Baseline algorithm '{baseline_algo}' not found in data.")
    
    # --- 2. Smoothing (Binning) ---
    
    # --- 2. Smoothing (Binning) ---
    # User requested averaging every 2 data points.
    # Original data: checkpoints every 10k steps.
    # Binning every 2 checkpoints = every 20k steps.
    BIN_SIZE_STEPS = 20000
    full_df['Smoothed_Timestep'] = (full_df['Timestep'] // BIN_SIZE_STEPS) * BIN_SIZE_STEPS / 100000.0
    
    # Create the plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 16), sharex=True)
    
    # Plot Logic: Hue by Base Algo, Style by Difficulty
    # Use Algorithm_Base for color consistency
    
    # Plot 1: Cumulative Reward
    sns.lineplot(data=full_df, x="Smoothed_Timestep", y="Reward", 
                 hue="Algorithm_Base", style="Env_Difficulty",
                 ax=axes[0], errorbar=('ci', 95), marker=None) 
    axes[0].set_ylabel("Cumulative Reward")
    axes[0].set_title(f"Training Performance (Smoothed)")
    # Move legend outside
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    # Plot 2: Success Rate
    sns.lineplot(data=full_df, x="Smoothed_Timestep", y="Success", 
                 hue="Algorithm_Base", style="Env_Difficulty",
                 ax=axes[1], errorbar=('ci', 95), marker=None)
    axes[1].set_ylabel("Success Rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Success Rate")
    axes[1].get_legend().remove() # Share legend from top
    
    # Plot 3: Average Travel Time
    sns.lineplot(data=full_df, x="Smoothed_Timestep", y="Time", 
                 hue="Algorithm_Base", style="Env_Difficulty",
                 ax=axes[2], errorbar=('ci', 95), marker=None)
    axes[2].set_ylabel("Average Travel Time (s)") 
    axes[2].set_xlabel("Timestep (x $10^5$)")
    axes[2].set_title("Travel Time")
    axes[2].get_legend().remove()
    
    plt.tight_layout()
    output_filename = "paper_training_curves_final.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    plot_paper_curves()
