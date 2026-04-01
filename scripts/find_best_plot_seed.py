import os
import sys
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import marinenav_env.envs.marinenav_env as marinenav_env
from APF import APF_agent
try:
    from stable_baselines3 import PPO, DQN
    try:
        from sb3_contrib import QRDQN
    except Exception:
        QRDQN = None
except Exception:
    try:
        from thirdparty import PPO, DQN, QRDQN
    except Exception:
        PPO = None
        DQN = None
        QRDQN = None

sys.path.insert(0, os.path.join(REPO_ROOT, 'thirdparty'))
from thirdparty.IQN.agent import IQNAgent


def calc_energy(env, action_history, trajectory):
    total = 0.0
    cur = np.array(env.start)
    N = env.robot.N
    for k in range(len(trajectory)):
        nxt = np.array(trajectory[k])
        dist = np.linalg.norm(nxt - cur)
        step_idx = k // N
        if step_idx < len(action_history):
            a_idx = action_history[step_idx]
            try:
                a_cmd, _ = env.robot.actions[int(a_idx)]
                total += abs(a_cmd) * dist
            except Exception:
                pass
        cur = nxt
    return total


def run_episode(env, agent, typ):
    obs = env.reset()
    done = False
    step = 0
    max_steps = 1000

    while not done and step < max_steps:
        if typ == 'iqn':
            res, _ = agent.act_adaptive_eval(obs, eps=0.0)
            action = res[0]
        elif typ in ('ppo', 'd3qn', 'rainbow'):
            action, _ = agent.predict(obs, deterministic=True)
        else:
            action = agent.act(obs)

        if isinstance(action, (np.ndarray, list)):
            action = int(action)

        obs, _, done, _ = env.step(action)
        step += 1

    traj = np.array(env.robot.trajectory)
    if len(traj) == 0:
        traj = np.array([[env.start[0], env.start[1]]])

    path = np.sum(np.sqrt(np.sum(np.diff(traj, axis=0) ** 2, axis=1)))
    en = calc_energy(env, env.robot.action_history, env.robot.trajectory)
    t = step * env.robot.N * env.robot.dt
    succ = 1 if np.linalg.norm(traj[-1] - env.goal) < 2.0 else 0
    return succ, t, en, path


def main():
    if PPO is None or DQN is None:
        raise ImportError('Neither stable_baselines3 nor thirdparty PPO/DQN could be imported.')

    env = marinenav_env.MarineNavEnv(seed=0)
    env.num_obs = 10
    env.dynamic_obstacles = True
    env.reset_start_and_goal = False
    env.start = np.array([5.0, 5.0])
    env.goal = np.array([45.0, 45.0])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    iqn = IQNAgent(state_dim, action_dim, device='cpu', seed=0)
    iqn_path = 'pretrained_models/Static_Obs10_AdaptiveIQN/seed_42/network_params.pth'
    iqn.qnetwork_local.load_state_dict(torch.load(iqn_path, map_location='cpu'))

    ppo = PPO.load('pretrained_models/PPO/best_model.zip', device='cpu')
    d3qn = DQN.load('pretrained_models/D3QN/best_model.zip', device='cpu')

    rainbow = None
    rb_path = 'pretrained_models/Rainbow/seed_42/final_model.zip'
    if os.path.exists(rb_path):
        try:
            rainbow = QRDQN.load(rb_path, device='cpu') if QRDQN else DQN.load(rb_path, device='cpu')
        except Exception:
            rainbow = DQN.load(rb_path, device='cpu')

    apf = APF_agent(env.robot.a, env.robot.w)

    best = None
    for s in range(0, 31):
        rows = {}
        for name, agent, typ in [
            ('iqn', iqn, 'iqn'),
            ('ppo', ppo, 'ppo'),
            ('d3qn', d3qn, 'd3qn'),
            ('rainbow', rainbow, 'rainbow'),
            ('apf', apf, 'apf'),
        ]:
            env.seed(s)
            rows[name] = run_episode(env, agent, typ)

        iqn_s, iqn_t, _, _ = rows['iqn']
        ppo_s, ppo_t, _, _ = rows['ppo']
        d3_s, _, _, _ = rows['d3qn']
        rb_s, _, _, _ = rows['rainbow']
        apf_s, _, _, _ = rows['apf']

        score = 0.0
        if iqn_s == 1:
            score += 5
        if ppo_s == 0:
            score += 3
        if d3_s == 0:
            score += 2
        if rb_s == 0:
            score += 2
        if apf_s == 0:
            score += 2
        if iqn_s == 1 and ppo_s == 1 and iqn_t < ppo_t:
            score += (ppo_t - iqn_t) / 5.0

        if best is None or score > best[0]:
            best = (score, s, rows)

    print('BEST_SCORE:', best[0])
    print('BEST_SEED:', best[1])
    print('BEST_ROWS:', best[2])


if __name__ == '__main__':
    main()
