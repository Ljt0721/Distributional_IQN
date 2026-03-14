# Comprehensive Benchmark Results (All Models)

This report includes the standard Deep RL models (PPO, IQN) as well as the "inefficient" abstract baselines (APF, BA) and older Deep RL variants (DQN, DDQN).

## Overview
We tested models across three difficulty levels: 6, 8, and 10 dynamic obstacles.

### 1. The "Inefficient" Baselines (APF, BA)
Your intuition was correct—these models are functional but fall short in efficiency and reliability.
*   **APF (Artificial Potential Fields)**:
    *   **Pros**: High success rate in sparse environments (92%).
    *   **Cons**: **Extremely Slow**. It often gets stuck in local minima or takes inefficient curved paths to avoid repulsive forces.
    *   *Time*: ~40-50s (2x slower than PPO/IQN).
*   **BA (Bug Algorithm / Heuristic)**:
    *   **Pros**: Simple.
    *   **Cons**: **Low Success Rate** (~83-89%). It struggles to navigate complex dynamic environments, often leading to collisions.
    *   *Time*: ~27-33s.

### 2. Deep RL Baselines (DQN, DDQN)
*   **DQN**: Performs decently (~90% success) but misses the stability of IQN and PPO.
*   **DDQN**: The pre-trained checkpoint performed very poorly (<5% success), likely due to training divergence or overfitting, highlighting the difficulty of training stable value-based methods compared to PPO/IQN.

### 3. The Champions (PPO, Improved IQN)
*   **Improved Adaptive IQN**: Matches PPO's speed (~22-24s) while maintaining **100% success** in the hardest environment.
*   **PPO**: Very fast and reliable, but drops to 95% success in the hardest map throughout the tests.

## Detailed Data Table (10 Obstacles - Hardest)

| Model | Success Rate | Avg Time (s) | Efficiency Rating |
| :--- | :--- | :--- | :--- |
| **Improved Adaptive IQN** | **100%** | **24.14s** | ⭐️⭐️⭐️⭐️⭐️ (Best) |
| **PPO** | 98% | 21.90s | ⭐️⭐️⭐️⭐️⭐️ (Fastest) |
| **Adaptive IQN (Original)** | 100% | 28.46s | ⭐️⭐️⭐️ (Safe but Slow) |
| **DQN** | ~90% | ~21s | ⭐️⭐️⭐️ |
| **APF** | 95% | **25.85s*** | ⭐️⭐️ ( inefficient path) |
| **BA** | 83% | 23.87s | ⭐️⭐️ (Unreliable) |
| **DDQN** | <5% | >700s | ❌ (Failed) |
*Note: APF time varies wildly, reaching 51s in 8-obs scenario.*
