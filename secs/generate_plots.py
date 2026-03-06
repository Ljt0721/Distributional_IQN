import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Load data
csv_path = 'comprehensive_benchmark_results.csv'
df = pd.read_csv(csv_path)

# Rename Agents for better display
agent_map = {
    'New Adaptive IQN': 'Adaptive IQN (Ours)',
    'Old Greedy IQN': 'Greedy IQN',
    'Old DQN': 'DQN',
    'APF': 'APF',
    'BA': 'BA'
}
df['Agent'] = df['Agent'].replace(agent_map)

# Filter for relevant agents (if any unexpected ones appear)
relevant_agents = list(agent_map.values())
df = df[df['Agent'].isin(relevant_agents)]

# Ensure Success column is boolean
df['Success'] = df['Success'].astype(str).str.lower() == 'true'

# Calculate Success Rates
success_rates = df.groupby('Agent')['Success'].mean() * 100
success_rates = success_rates.reindex(relevant_agents) # Ensure consistent order

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# 1. Success Rate Bar Chart
ax1 = axes[0, 0]
colors = sns.color_palette("muted", len(relevant_agents))
# Highlight ours
bar_colors = [colors[0] if agent != 'Adaptive IQN (Ours)' else '#e74c3c' for agent in relevant_agents] # Red highlight for ours? Or maybe distinguish better. Let's use specific color palette.
# Let's use a standard palette but make ours stand out slightly or just be consistent.
sns.barplot(x=success_rates.index, y=success_rates.values, ax=ax1, palette="viridis")
ax1.set_title('(a) Success Rate (%)', fontweight='bold')
ax1.set_ylabel('Success Rate (%)')
ax1.set_xlabel('')
ax1.set_ylim(0, 105)
for i, v in enumerate(success_rates.values):
    ax1.text(i, v + 2, f"{v:.1f}%", ha='center', va='bottom', fontweight='bold')

# Filter for Successful Episodes Only for the next plots
df_success = df[df['Success'] == True]

# 2. Time to Goal Boxplot
ax2 = axes[0, 1]
sns.boxplot(x='Agent', y='Time (s)', data=df_success, ax=ax2, order=relevant_agents, palette="viridis", showfliers=False)
ax2.set_title('(b) Time to Goal (s)', fontweight='bold')
ax2.set_ylabel('Time (s)')
ax2.set_xlabel('')

# 3. Energy Consumption Boxplot
ax3 = axes[1, 0]
sns.boxplot(x='Agent', y='Energy (J)', data=df_success, ax=ax3, order=relevant_agents, palette="viridis", showfliers=False)
ax3.set_title('(c) Energy Consumption (J)', fontweight='bold')
ax3.set_ylabel('Energy (J)')
ax3.set_xlabel('')

# 4. Path Length Boxplot
ax4 = axes[1, 1]
sns.boxplot(x='Agent', y='Path Length (m)', data=df_success, ax=ax4, order=relevant_agents, palette="viridis", showfliers=False)
ax4.set_title('(d) Path Length (m)', fontweight='bold')
ax4.set_ylabel('Path Length (m)')
ax4.set_xlabel('')

# Rotate x-axis labels for better readability
for ax in axes.flat:
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

# Save figure
output_dir = 'figures'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, 'benchmark_results_plot.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to {output_path}")
