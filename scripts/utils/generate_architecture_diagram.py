
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_neural_network_diagram(output_path="network_architecture_diagram.png"):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Utility to draw centered text in box
    def draw_box(x, y, width, height, text, color='#E0E0E0', fontsize=10):
        rect = patches.Rectangle((x, y), width, height, linewidth=1.5, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=fontsize)
        return (x, y, width, height)

    def draw_arrow(start, end):
        ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1], 
                 head_width=1.5, head_length=2, fc='black', ec='black', length_includes_head=True)

    # --- System Constants ---
    col1_x = 5
    col2_x = 25
    col3_x = 50
    col4_x = 75
    col5_x = 92
    
    # --- 1. Inputs ---
    # Velocity
    v_box = draw_box(col1_x, 80, 10, 6, "Velocity\n(2)", color='#d4ebf2')
    # Goal
    g_box = draw_box(col1_x, 70, 10, 6, "Goal\n(2)", color='#d4ebf2')
    # Lidar
    s_box = draw_box(col1_x, 50, 10, 15, "Lidar/Sensor\n(22)", color='#d4ebf2')
    
    # Tau Input
    tau_box = draw_box(col1_x, 15, 10, 6, "Tau (τ)\n~ U[0,1]", color='#f2d4d4')

    # --- 2. Encoders ---
    # Velocity Enc
    ve_box = draw_box(col2_x, 80, 12, 6, "Linear\n(2→16)", color='#ffffff')
    # Goal Enc
    ge_box = draw_box(col2_x, 70, 12, 6, "Linear\n(2→16)", color='#ffffff')
    # Sensor Enc
    se_box = draw_box(col2_x, 50, 12, 15, "Linear\n(22→176)", color='#ffffff')
    
    # Quantile Embedding
    qe1_box = draw_box(col2_x - 3, 15, 8, 6, "Cos Emb\n(64)", color='#fff0f0')
    qe2_box = draw_box(col2_x + 8, 15, 10, 6, "Linear\n(64→208)\n+ ReLU", color='#fff0f0')

    # Connections to Encoders
    draw_arrow((v_box[0]+v_box[2], v_box[1]+v_box[3]/2), (ve_box[0], ve_box[1]+ve_box[3]/2))
    draw_arrow((g_box[0]+g_box[2], g_box[1]+g_box[3]/2), (ge_box[0], ge_box[1]+ge_box[3]/2))
    draw_arrow((s_box[0]+s_box[2], s_box[1]+s_box[3]/2), (se_box[0], se_box[1]+se_box[3]/2))
    
    draw_arrow((tau_box[0]+tau_box[2], tau_box[1]+tau_box[3]/2), (qe1_box[0], qe1_box[1]+qe1_box[3]/2))
    draw_arrow((qe1_box[0]+qe1_box[2], qe1_box[1]+qe1_box[3]/2), (qe2_box[0], qe2_box[1]+qe2_box[3]/2))

    # --- Feature Concatenation ---
    # Visualizing feature concatenation
    concat_box = draw_box(col3_x, 50, 4, 40, "Concat\n(208)", color='#e0e0e0')
    
    # Arrows to concat
    draw_arrow((ve_box[0]+ve_box[2], ve_box[1]+ve_box[3]/2), (concat_box[0], 83))
    draw_arrow((ge_box[0]+ge_box[2], ge_box[1]+ge_box[3]/2), (concat_box[0], 73))
    draw_arrow((se_box[0]+se_box[2], se_box[1]+se_box[3]/2), (concat_box[0], 57))

    # --- Element-wise Multiplication ---
    mult_circle = patches.Circle((col3_x + 10, 45), 2.5, fc='yellow', ec='black')
    ax.add_patch(mult_circle)
    ax.text(col3_x + 10, 45, "⊙", ha='center', va='center', fontsize=20, fontweight='bold')
    
    # Arrow from state features to multiply
    draw_arrow((concat_box[0]+concat_box[2], 70), (col3_x + 10, 48)) # Approx center

    # Arrow from quantile features to multiply
    draw_arrow((qe2_box[0]+qe2_box[2], qe2_box[1]+qe2_box[3]/2), (col3_x + 10, 42))
    
    # Label for Hadamard
    ax.text(col3_x + 10, 38, "Element-wise\nProduct", ha='center', fontsize=9)

    # --- 3. Hidden Layers ---
    h1_box = draw_box(col4_x, 60, 10, 8, "Linear\n(208→64)\n+ ReLU", color='#ffffff')
    h2_box = draw_box(col4_x, 40, 10, 8, "Linear\n(64→64)\n+ ReLU", color='#ffffff')
    
    # Connections
    draw_arrow((col3_x + 12.5, 45), (h1_box[0], h1_box[1]+h1_box[3]/2))
    draw_arrow((h1_box[0]+h1_box[2]/2, h1_box[1]), (h2_box[0]+h2_box[2]/2, h2_box[1]+h2_box[3]))

    # --- 4. Output ---
    out_box = draw_box(col4_x, 20, 10, 8, "Output Head\n(64→N_Act)", color='#d4f2d4')
    draw_arrow((h2_box[0]+h2_box[2]/2, h2_box[1]), (out_box[0]+out_box[2]/2, out_box[1]+out_box[3]))
    
    # Final Output
    dist_box = draw_box(col5_x, 20, 8, 8, "Quantiles\nZ(τ)", color='#d4f2d4', fontsize=10)
    draw_arrow((out_box[0]+out_box[2], out_box[1]+out_box[3]/2), (dist_box[0], dist_box[1]+dist_box[3]/2))

    # --- Grouping Labels ---
    # Input Group
    # ax.text(col1_x + 5, 90, "Inputs", ha='center', fontsize=12, fontweight='bold')
    # Encoder Group
    # ax.text(col2_x + 6, 90, "Feature Encoders", ha='center', fontsize=12, fontweight='bold')

    plt.title("IQN Architecture for Navigation (ObsEncoder)", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Diagram saved to {output_path}")

if __name__ == "__main__":
    draw_neural_network_diagram()
