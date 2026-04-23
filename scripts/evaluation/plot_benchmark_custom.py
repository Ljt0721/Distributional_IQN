
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Load Data
try:
    with open("benchmark_data_final.json", "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: benchmark_data_final.json not found.")
    exit()

# Transform Data for Plotting
records = []

for group_name, models_data in data.items():
    # Extract obstacle count from group name (e.g., "Obstacles_6" -> 6)
    obs_count = int(group_name.split("_")[1])
    
    for model_name, metrics in models_data.items():
        records.append({
            "Obstacles": obs_count,
            "Model": model_name,
            "Success Rate": metrics["success_rate"],
            "Average Time (s)": metrics["avg_time"],
            "Average Energy (J)": metrics["avg_energy"]
        })

df = pd.DataFrame(records)

# Use Seaborn for nice plotting
sns.set_style("whitegrid")
sns.set_context("talk")

# Plot 1: Success Rate
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=df, x="Obstacles", y="Success Rate", hue="Model", palette="viridis")
plt.title("Success Rate vs Obstacle Density")
plt.ylim(0, 1.1)

# Annotate with values
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{p.get_height():.2f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=9) 

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("benchmark_success.png")
plt.close()

# Plot 2: Average Time
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="Obstacles", y="Average Time (s)", hue="Model", palette="viridis")
plt.title("Average Navigation Time vs Obstacle Density")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("benchmark_time.png")
plt.close()

# Plot 3: Average Energy
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="Obstacles", y="Average Energy (J)", hue="Model", palette="viridis")
plt.title("Average Energy Consumption vs Obstacle Density")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("benchmark_energy.png")
plt.close()

# Create a Combined Summary Table Image (Optional but nice)
# Just print the summary to console
pivot_success = df.pivot(index="Model", columns="Obstacles", values="Success Rate")
pivot_time = df.pivot(index="Model", columns="Obstacles", values="Average Time (s)")
pivot_energy = df.pivot(index="Model", columns="Obstacles", values="Average Energy (J)")

print("Success Rate Summary:")
print(pivot_success)
print("\nTime Summary:")
print(pivot_time)
print("\nEnergy Summary:")
print(pivot_energy)
