import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
from collections import deque
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
if not hasattr(np, 'int_'):
    np.int_ = np.int64
if not hasattr(np, 'float_'):
    np.float_ = np.float64

import gym
import copy

# Add thirdparty to path for imports
sys.path.insert(0, os.path.join(os.getcwd(), "thirdparty"))

# Import Replay Buffer manually (to avoid circular dep)
try:
    from thirdparty.IQN.replay_buffer import ReplayBuffer
except ImportError:
    # Build it locally if import fails
    from collections import deque, namedtuple
    class ReplayBuffer:
        def __init__(self, buffer_size, batch_size, device, seed, gamma, n_step=1):
            self.device = device
            self.memory = deque(maxlen=buffer_size)  
            self.batch_size = batch_size
            self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
            self.seed = random.seed(seed)
            self.gamma = gamma
            self.n_step = n_step
            self.n_step_buffer = deque(maxlen=self.n_step)
        
        def add(self, state, action, reward, next_state, done):
            self.n_step_buffer.append((state, action, reward, next_state, done))
            if len(self.n_step_buffer) == self.n_step:
                state, action, reward, next_state, done = self.n_step_buffer[0]
                for i in range(1, self.n_step):
                    r = self.n_step_buffer[i][2]
                    d = self.n_step_buffer[i][4]
                    reward += (self.gamma ** i) * r
                    if d:
                        next_state = self.n_step_buffer[i][3]
                        done = d
                        break
                    else:
                        next_state = self.n_step_buffer[i][3]
                e = self.experience(state, action, reward, next_state, done)
                self.memory.append(e)

        def sample(self):
            experiences = random.sample(self.memory, k=self.batch_size)
            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
            return (states, actions, rewards, next_states, dones)

        def __len__(self):
            return len(self.memory)

import marinenav_env.envs.marinenav_env as marinenav_env


# --- Improved IQN Agent Definition ---
# We define the classes here to allow customizations (like layer size, quantile count)
# without modifying the thirdparty library directly.

class ImprovedIQNModel(nn.Module):
    def __init__(self, state_size, action_size, layer_size, seed, device):
        super(ImprovedIQNModel, self).__init__()
        self.device = torch.device(device)
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        
        # INCREASED QUANTILE SAMPLES
        self.N = 64  # Number of quantiles for target (was 8 or 32)
        self.K = 32  # Number of quantiles for action selection
        self.n_cos = 64 # Embedding dimension
        self.layer_size = layer_size

        self.pis = torch.FloatTensor([np.pi * i for i in range(self.n_cos)]).view(1, 1, self.n_cos).to(self.device)
        
        # Network Architecture
        self.head = nn.Linear(self.state_size, layer_size) # Feature extractor
        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        
        # Deeper Network
        self.fc1 = nn.Linear(layer_size, layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.output_layer = nn.Linear(layer_size, action_size)

    def calc_cos(self, batch_size, n_tau=8, cvar=1.0):
        # CVaR-aware sampling: Sample taus from [0, cvar] instead of [0, 1]
        taus = torch.rand(batch_size, n_tau).to(self.device)
        taus = taus * cvar 
        
        cos = torch.cos(taus.unsqueeze(2) * self.pis)
        cos = cos.view(batch_size * n_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, n_tau, self.layer_size)
        return cos_x, taus

    def forward(self, inputs, num_tau=8, cvar=1.0):
        batch_size = inputs.shape[0]
        x = torch.relu(self.head(inputs))
        
        cos_x, taus = self.calc_cos(batch_size, num_tau, cvar)
        
        # Interaction between features and quantiles
        x = (x.unsqueeze(1) * cos_x).view(batch_size * num_tau, self.layer_size)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = self.output_layer(x)
        
        return out.view(batch_size, num_tau, self.action_size), taus

    def get_action(self, inputs, cvar):
        quantiles, _ = self.forward(inputs, self.K, cvar)
        # Mean of quantiles is the Q-value
        actions = quantiles.mean(dim=2) 
        return actions

class ImprovedIQNAgent():
    def __init__(self, state_size, action_size, 
                 layer_size=256,  # Increased from 64
                 BATCH_SIZE=32, 
                 BUFFER_SIZE=100000, 
                 LR=5e-4, 
                 GAMMA=0.99, 
                 TAU=1e-3, 
                 device="cpu", 
                 seed=42):
        
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.LR = LR
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.BATCH_SIZE = BATCH_SIZE
        
        # Create Networks
        self.qnetwork_local = ImprovedIQNModel(state_size, action_size, layer_size, seed, device).to(device)
        self.qnetwork_target = ImprovedIQNModel(state_size, action_size, layer_size, seed, device).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, device, seed, GAMMA, 1) # n_step=1
        
        # Epsilon Decay
        self.eps = 1.0
        self.eps_end = 0.05
        self.eps_decay = 0.995 # Slower decay

        self.t_step = 0
        self.UPDATE_EVERY = 4
        
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.GAMMA)

    def act(self, state, eval_mode=False, adaptive=False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        
        cvar = 1.0
        if adaptive:
            cvar = self.adjust_cvar_improved(state)

        with torch.no_grad():
            quantiles, _ = self.qnetwork_local.forward(state, self.qnetwork_local.K, cvar)
            action_values = quantiles.mean(dim=1)

        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > self.eps or eval_mode:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def adjust_cvar_improved(self, state_tensor):
        # Extract sonar data from state (assuming standard format: pos(2) + velocity(2) + sonar(22))
        # state is tensor (1, 26)
        state_np = state_tensor.cpu().numpy()[0]
        sonar_points = state_np[4:]
        
        closest_d = 10.0 # Default max reasonable distance
        
        # Safely compute minimum distance
        # Reshape sonar into (N, 2)
        if len(sonar_points) % 2 == 0:
            points = sonar_points.reshape(-1, 2)
            distances = np.linalg.norm(points, axis=1)
            # Filter non-zero (valid) points? Assume 0,0 is "no data" or "robot center"?
            # Actually standard is relative coordinates. 
            # If (0,0), it's a collision or sensor failure.
            # Using simple min.
            valid_dists = distances[distances > 0.01]
            if len(valid_dists) > 0:
                closest_d = np.min(valid_dists)
        
        # Improved CVaR Logic:
        # Don't go below 0.3 to prevent freezing.
        # Scale linearly between 0 and 10 meters.
        
        cvar = max(0.3, min(1.0, closest_d / 8.0))
        return cvar

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values (for next states) from target model
        # IQN Loss Calculation
        
        # 1. Get Target Quantiles
        # Select action using Greedy Policy on Target Network (Double DQN style or simple?)
        # Simple IQN: Use target network for selection and valuation
        
        with torch.no_grad():
             # Target Quantiles: (batch, N, action_size)
            target_quantiles, _ = self.qnetwork_target.forward(next_states, self.qnetwork_target.N, cvar=1.0)
            # Max over actions: (batch, N)
            target_q_values = target_quantiles.mean(dim=1)
            best_actions = target_q_values.argmax(dim=1, keepdim=True) # (batch, 1)
            
            # Gather best action quantiles
            # target_quantiles: (batch, N, actions) -> Gather -> (batch, N, 1)
            max_target_quantiles = target_quantiles.gather(2, best_actions.unsqueeze(1).expand(-1, self.qnetwork_target.N, -1)).squeeze(2)
            
            # Compute Targets
            # rewards: (batch, 1) -> (batch, N) by expansion
            # dones: (batch, 1)
            rewards = rewards.expand(-1, self.qnetwork_target.N)
            dones = dones.expand(-1, self.qnetwork_target.N)
            
            Q_targets = rewards + (gamma * max_target_quantiles * (1 - dones))

        # 2. Get Expected Quantiles
        # Current States -> (batch, K, actions)
        current_quantiles, taus = self.qnetwork_local.forward(states, self.qnetwork_target.N, cvar=1.0)
        # Gather actions taken: (batch, K, 1)
        Q_expected = current_quantiles.gather(2, actions.unsqueeze(1).expand(-1, self.qnetwork_target.N, -1)).squeeze(2)
        
        # 3. Quantile Huber Loss
        # Q_targets: (batch, N) -> (batch, 1, N)
        # Q_expected: (batch, N) -> (batch, N, 1)
        # Diff: (batch, N, N)
        
        td_error = Q_targets.unsqueeze(1) - Q_expected.unsqueeze(2)
        huber_loss = torch.where(torch.abs(td_error) < 1.0, 0.5 * td_error**2, torch.abs(td_error) - 0.5)
        
        # Taus: (batch, N, 1) for broadcasting
        taus = taus.unsqueeze(2).expand(-1, -1, self.qnetwork_target.N)
        
        element_wise_loss = torch.abs(taus - (td_error.detach() < 0).float()) * huber_loss
        loss = element_wise_loss.sum(dim=2).mean(dim=1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()

        # Update Target Network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def update_epsilon(self):
        self.eps = max(self.eps_end, self.eps_decay * self.eps)

def train_agent():
    # Setup Environment
    # Training on HARD environment (10 obstacles) to ensure robustness
    env = gym.make("marinenav_env:marinenav_env-v0")
    env.seed(42)
    env.num_obs = 10
    env.num_cores = 8
    
    # We want static obstacles for consistent learning, OR dynamic? 
    # User said "Adaptive IQN should be stronger in dynamic". 
    # Let's train on Dynamic Obstacles enabled? 
    # Actually, let's train on STATIC first to fix the baseline "weakness". 
    # The benchmark was static. If it fails static, it fails dynamic.
    env.schedule = None 
    
    state_size = env.get_state_space_dimension()
    action_size = env.get_action_space_dimension()
    
    print(f"State Size: {state_size}, Action Size: {action_size}")
    
    # Initialize Agent
    agent = ImprovedIQNAgent(state_size, action_size, 
                             layer_size=256, 
                             LR=1e-4, # Lower LR for stability
                             BATCH_SIZE=64,
                             device="cpu") # Use CPU for stability
    
    # Check for existing model to resume
    model_path = os.path.join("improved_adaptive_iqn", "improved_model.pth")
    start_episode = 1
    if os.path.exists(model_path):
        print(f"Resuming training from {model_path}...")
        try:
            agent.qnetwork_local.load_state_dict(torch.load(model_path))
            agent.qnetwork_target.load_state_dict(torch.load(model_path))
            # Lower epsilon if resuming
            agent.eps = 0.1 
        except Exception as e:
            print(f"Could not load model: {e}")

    n_episodes = 5000 # Continuous training
    max_t = 1000
    
    scores = []
    scores_window = deque(maxlen=50)
    best_score = -np.inf
    
    save_dir = "improved_adaptive_iqn"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting Continuous Training Loop ({n_episodes} episodes)...")
    
    for i_episode in range(start_episode, n_episodes + 1):
        state = env.reset()
        score = 0
        
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            # Custom Reward Shaping for Robustness
            # Penalty for being too close to obstacles to encourage safety
            # But not too high to discourage movement
            
            dist_to_obs = agent.adjust_cvar_improved(torch.from_numpy(next_state).float().unsqueeze(0).to(agent.device)) * 8.0
            if dist_to_obs < 1.0:
                 reward -= 0.1 # Small proximity penalty
            
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
                
        agent.update_epsilon()
        scores_window.append(score)
        scores.append(score)
        
        avg_score = np.mean(scores_window)
        
        print(f'\rEpisode {i_episode}\tAvg Score: {avg_score:.2f}\tEps: {agent.eps:.2f}', end="")
        
        if i_episode % 25 == 0:
            print(f'\rEpisode {i_episode}\tAvg Score: {avg_score:.2f}\tBest: {best_score:.2f}')
            # Save latest
            torch.save(agent.qnetwork_local.state_dict(), os.path.join(save_dir, "improved_model.pth"))
            
        if avg_score > best_score and i_episode > 50:
            best_score = avg_score
            torch.save(agent.qnetwork_local.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"\nNew Best Model Saved! Score: {best_score:.2f}")

    # Final Save
    torch.save(agent.qnetwork_local.state_dict(), os.path.join(save_dir, "final_model.pth"))
    
    # Save Metadata with Constructor Params (Important for loading later)
    metadata = {
        "state_size": state_size,
        "action_size": action_size,
        "layer_size": 256,
        "seed": 42
    }
    with open(os.path.join(save_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f)
        
    print(f"\nTraining Complete. Model saved to {save_dir}")

def evaluate_new_model():
    # Quick evaluation
    save_dir = "improved_adaptive_iqn"
    path = os.path.join(save_dir, "improved_model.pth")
    
    env = gym.make("marinenav_env:marinenav_env-v0")
    env.seed(42)
    env.num_obs = 10 
    
    state_size = env.get_state_space_dimension()
    action_size = env.get_action_space_dimension()
    
    agent = ImprovedIQNAgent(state_size, action_size, layer_size=256, device="cpu")
    agent.qnetwork_local.load_state_dict(torch.load(path))
    agent.qnetwork_local.eval()
    
    print("\nEvaluations (Adaptive Mode):")
    successes = 0
    total_time = 0
    episodes = 20
    
    for i in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            action = agent.act(state, eval_mode=True, adaptive=True)
            state, reward, done, key = env.step(action)
            steps += 1
            if steps > 500: done = True
            
        success = True if key.get("state") == "reach goal" else False
        if success: successes += 1
        total_time += steps * 0.5 # approx
        
    print(f"Success Rate: {successes/episodes:.2f}")

if __name__ == "__main__":
    train_agent()
    # evaluate_new_model()
