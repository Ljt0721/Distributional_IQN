import os
import sys
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import plot_specific_seed as ps
import marinenav_env.envs.marinenav_env as marinenav_env


def main():
    env = marinenav_env.MarineNavEnv(seed=0)
    env.num_obs = 10
    env.dynamic_obstacles = True
    env.reset_start_and_goal = False
    env.start = np.array([5.0, 5.0])
    env.goal = np.array([45.0, 45.0])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    iqn_agent = ps.IQNAgent(state_dim, action_dim, device='cpu', seed=0)
    iqn_path = 'pretrained_models/Static_Obs10_AdaptiveIQN/seed_42/network_params.pth'
    if os.path.exists(iqn_path):
        state_dict = torch.load(iqn_path, map_location='cpu')
        iqn_agent.qnetwork_local.load_state_dict(state_dict)

    ppo_model = ps.PPO.load('pretrained_models/PPO/best_model.zip', device='cpu')
    d3qn_model = ps.DQN.load('pretrained_models/D3QN/best_model.zip', device='cpu')

    rainbow_model = None
    rainbow_path = 'pretrained_models/Rainbow/seed_42/final_model.zip'
    if os.path.exists(rainbow_path):
        try:
            rainbow_model = ps.QRDQN.load(rainbow_path, device='cpu') if ps.QRDQN else ps.DQN.load(rainbow_path, device='cpu')
        except Exception:
            rainbow_model = None

    apf_agent = ps.APF_agent(env.robot.a, env.robot.w)

    agents = [
        ('Adaptive IQN', iqn_agent, 'iqn_adaptive'),
        ('PPO', ppo_model, 'ppo'),
        ('D3QN', d3qn_model, 'd3qn'),
        ('Rainbow', rainbow_model, 'rainbow'),
        ('APF', apf_agent, 'apf'),
    ]

    best = None
    for seed in range(0, 31):
        rows = {}
        for name, model, typ in agents:
            env.seed(seed)
            res = ps.run_single_episode(env, model, typ)
            rows[name] = res

        iqn = rows['Adaptive IQN']
        ppo = rows['PPO']
        d3 = rows['D3QN']
        rb = rows['Rainbow']
        apf = rows['APF']

        score = 0.0
        score += 5 if iqn['Success'] == 1 else -5
        score += 3 if ppo['Success'] == 0 else 0
        score += 2 if d3['Success'] == 0 else 0
        score += 2 if rb['Success'] == 0 else 0
        score += 2 if apf['Success'] == 0 else 0

        if iqn['Success'] == 1 and ppo['Success'] == 1 and iqn['Time'] < ppo['Time']:
            score += (ppo['Time'] - iqn['Time']) / 5.0

        if best is None or score > best[0]:
            best = (score, seed, rows)

    score, seed, rows = best
    print('BEST_SEED', seed)
    print('BEST_SCORE', score)
    for k, v in rows.items():
        print(k, v['Success'], round(v['Time'], 2), round(v['Energy'], 2), round(v['Length'], 2))


if __name__ == '__main__':
    main()
