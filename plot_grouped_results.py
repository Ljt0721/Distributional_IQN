
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# Set style
sns.set(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

RESULTS_JSON = "benchmark_data_final.json"
OUTPUT_DIR = "secs/figures"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def plot_benchmark_results():
    print("Generating Benchmark Results Plot...")
    
    with open(RESULTS_JSON, 'r') as f:
        data = json.load(f)
    
    # Define Agents to Display
    agents = ["AdaptiveIQN", "PPO", "D3QN", "APF", "BA"]
    display_names = {
        "AdaptiveIQN": "Adaptive IQN\n(Ours)",
        "PPO": "PPO",
        "D3QN": "D3QN",
        "APF": "APF",
        "BA": "BA"
    }

    # Prepare Data for Grouped Bar Chart
    plot_data = []

    for map_key in ["Obstacles_6", "Obstacles_8", "Obstacles_10"]:
        obs_count = map_key.split("_")[1]
        map_label = f"{obs_count} Obstacles"
        
        if map_key in data:
            results = data[map_key]
            for agent in agents:
                if agent in results:
                    stats = results[agent]
                    plot_data.append({
                        "Scenario": map_label,
                        "Agent": display_names[agent],
                        "Success Rate": stats["success_rate"],
                        "Time (s)": stats["avg_time"],
                        "Energy (J)": stats["avg_energy"]
                    })
    
    df = pd.DataFrame(plot_data)

    # Define Colors - Ensure ours stands out or follows standard
    palette = sns.color_palette("bright", n_colors=len(agents))
    # Or mimic standard academic palettes (Blue, Green, Orange, Red, etc)
    # Let's use a explicit mapping
    color_map = {
        "Adaptive IQN\n(Ours)": "#e74c3c", # Red/Orange distinct
        "PPO": "#2ecc71", # Green
        "D3QN": "#3498db", # Blue
        "APF": "#9b59b6", # Purple
        "BA": "#95a5a6"  # Grey
    }

    # Create Figure with 3 Subplots (Side by Side)
    # Thin bars are achieved by 'width' parameter in barplot, but seaborn handles it by 'dodge'.
    # To make them look like the example (lots of thin bars), we just plot them normally with hue.
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Success Rate
    ax1 = axes[0]
    sns.barplot(x="Scenario", y="Success Rate", hue="Agent", data=df, ax=ax1, palette=color_map)
    ax1.set_title("Success Rate", fontweight='bold', fontsize=14)
    ax1.set_xlabel("")
    ax1.set_ylabel("Success Rate")
    ax1.set_ylim(0, 1.1)
    ax1.get_legend().remove()
    
    # 2. Time
    ax2 = axes[1]
    sns.barplot(x="Scenario", y="Time (s)", hue="Agent", data=df, ax=ax2, palette=color_map)
    ax2.set_title("Navigation Time", fontweight='bold', fontsize=14)
    ax2.set_xlabel("")
    ax2.set_ylabel("Time (s)")
    ax2.get_legend().remove()

    # 3. Energy
    ax3 = axes[2]
    sns.barplot(x="Scenario", y="Energy (J)", hue="Agent", data=df, ax=ax3, palette=color_map)
    ax3.set_title("Energy Consumption", fontweight='bold', fontsize=14)
    ax3.set_xlabel("")
    ax3.set_ylabel("Energy (J)")
    
    # Legend on the right (outside)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=12)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "benchmark_results_grouped.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    plot_benchmark_results()
