
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def plot_results():
    # Load Data
    with open("final_benchmark_results.json", "r") as f:
        data = json.load(f)

    # Prepare DataFrames for Seaborn
    # Success Rate Data
    success_rates = {}
    for method, res in data.items():
        # Calculate percentage
        success_rates[method] = np.mean(res["success"]) * 100

    # Metric Data (Time & Energy) - Only for successful episodes
    df_metrics = []
    
    for method, res in data.items():
        success_indices = [i for i, s in enumerate(res["success"]) if s == 1]
        
        for idx in success_indices:
            df_metrics.append({
                "Method": method,
                "Time (s)": res["time"][idx],
                "Energy (J)": res["energy"][idx]
            })
            
    df = pd.DataFrame(df_metrics)

    # Setup Plot Style
    sns.set(style="whitegrid", context="talk")
    fig = plt.figure(figsize=(18, 6))
    
    # Define Palette (Highlight Ours)
    palette = {
        "Adaptive IQN (Ours)": "#2ecc71", # Green
        "DQN": "#3498db",                 # Blue
        "APF": "#95a5a6",                 # Grey
        "BA": "#7f8c8d"                   # Dark Grey
    }

    # --- Plot 1: Success Rate (Bar) ---
    ax1 = plt.subplot(1, 3, 1)
    
    methods = list(success_rates.keys())
    rates = list(success_rates.values())
    colors = [palette.get(m, "#95a5a6") for m in methods]
    
    bars = ax1.bar(methods, rates, color=colors, edgecolor="black", linewidth=1)
    
    ax1.set_title("Navigation Success Rate", fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel("Success Rate (%)", fontsize=14)
    ax1.set_ylim(0, 100)
    
    # Add labels on top
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45)

    # --- Plot 2: Navigation Time (Box) ---
    ax2 = plt.subplot(1, 3, 2)
    sns.boxplot(x="Method", y="Time (s)", data=df, ax=ax2, palette=palette, flierprops={"marker": "x"})
    ax2.set_title("Navigation Time (Successful)", fontsize=16, fontweight='bold', pad=15)
    ax2.set_ylabel("Time (seconds)", fontsize=14)
    ax2.set_xlabel("")
    plt.xticks(rotation=45)

    # --- Plot 3: Energy Consumption (Box) ---
    ax3 = plt.subplot(1, 3, 3)
    sns.boxplot(x="Method", y="Energy (J)", data=df, ax=ax3, palette=palette, flierprops={"marker": "x"})
    ax3.set_title("Energy Consumption", fontsize=16, fontweight='bold', pad=15)
    ax3.set_ylabel("Energy Cost", fontsize=14)
    ax3.set_xlabel("")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("final_performance_comparison.png", dpi=300)
    print("Plot saved to final_performance_comparison.png")

if __name__ == "__main__":
    plot_results()
