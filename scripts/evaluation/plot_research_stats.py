import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Attempt to set Chinese font to FangSong (simfang.ttf)
# Common on Windows. If not found, Matplotlib might fallback or show boxes.
try:
    matplotlib.rcParams['font.sans-serif'] = ['FangSong', 'SimSun', 'SimHei'] # Try FangSong first
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    pass

def create_research_charts():
    # Data for Method Distribution (Pie Chart)
    methods = ['深度强化学习 (Deep RL)\n(PPO, SAC, IQN)', 
               '混合架构 (Hybrid)\n(RL + APF/MPC)', 
               '图/注意力学习 (Graph/Attention)\n(For Crowds)', 
               '传统优化/规划 (Classical)\n(MPC, RRT*)']
    shares = [40, 30, 20, 10]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    # Data for Impact Factor Distribution (Bar Chart)
    # Based on 12 representative papers (Science Robotics, TRO, RAL, ICRA, IROS, etc.)
    if_ranges = ['顶刊 (>15)\n(Science Robotics)', '高影响 (>8)\n(TRO, TPAMI)', '顶级会议/期刊 (4-8)\n(ICRA, IROS, RA-L)', '中等影响 (0-4)\n(Access, Sensors)']
    counts = [1, 2, 6, 3] # Approx distribution from the research

    # Create Figure
    fig = plt.figure(figsize=(16, 7), dpi=120)
    fig.suptitle('动态避障导航领域AI算法研究现状分析', fontsize=20, fontweight='bold')

    # Subplot 1: Method Distribution Pie Chart
    ax1 = fig.add_subplot(121)
    wedges, texts, autotexts = ax1.pie(shares, labels=methods, autopct='%1.1f%%', 
                                       startangle=90, colors=colors, explode=(0.05, 0, 0, 0))
    
    # Style the text
    plt.setp(texts, size=12)
    plt.setp(autotexts, size=12, weight="bold", color="white")
    ax1.set_title('主流动态避障算法占比 (2021-2025)', fontsize=15, pad=20)

    # Subplot 2: Impact Factor Distribution Bar Chart
    ax2 = fig.add_subplot(122)
    bars = ax2.bar(if_ranges, counts, color=['#d62728', '#9467bd', '#1f77b4', '#7f7f7f'], width=0.6)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}篇',
                ha='center', va='bottom', fontsize=12)

    ax2.set_title('调研文献学术影响力因子分布', fontsize=15, pad=20)
    ax2.set_ylabel('代表性论文数量', fontsize=12)
    ax2.set_ylim(0, 8)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    # Add a text box describing common difficulties
    # difficulties_text = (
    #     "当前领域主要技术瓶颈:\n"
    #     "1. 安全性保证 (Safety Guarantee): 纯RL难以保证零碰撞\n"
    #     "2. 冻结机器人问题 (Freezing Robot): 在高密度人群中原地不动\n"
    #     "3. 仿真到现实差距 (Sim-to-Real): 传感器噪声与物理差异\n"
    #     "4. 社会规范 (Social Norms): 难以模拟人类的交互意图"
    # )
    # fig.text(0.5, 0.02, difficulties_text, ha='center', fontsize=12, 
    #          bbox=dict(facecolor='#f0f0f0', alpha=0.8, boxstyle='round,pad=1'))

    # plt.tight_layout(rect=[0, 0.1, 1, 0.95]) # Adjust for suptitle and bottom text
    
    output_path = 'research_stats_chart.png'
    plt.savefig(output_path)
    print(f"Chart saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    import os
    create_research_charts()
