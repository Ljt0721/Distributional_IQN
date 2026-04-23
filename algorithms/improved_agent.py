import torch
import torch.nn as nn
import numpy as np
import random
import os
from collections import deque, namedtuple

# Dummy Buffer to avoid import errors if needed, though we only use the Agent for eval
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device, seed, gamma, n_step=1):
        self.device = device
        self.memory = deque(maxlen=buffer_size)
    def add(self, *args): pass
    def sample(self): return None
    def __len__(self): return 0

class ImprovedIQNModel(nn.Module):
    def __init__(self, state_size, action_size, layer_size, seed, device):
        super(ImprovedIQNModel, self).__init__()
        self.device = torch.device(device)
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        
        # INCREASED QUANTILE SAMPLES
        self.N = 64  
        self.K = 32  
        self.n_cos = 64 
        self.layer_size = layer_size

        self.pis = torch.FloatTensor([np.pi * i for i in range(self.n_cos)]).view(1, 1, self.n_cos).to(self.device)
        
        # Network Architecture
        self.head = nn.Linear(self.state_size, layer_size) 
        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        
        # Deeper Network
        self.fc1 = nn.Linear(layer_size, layer_size)
        self.fc2 = nn.Linear(layer_size, layer_size)
        self.output_layer = nn.Linear(layer_size, action_size)

    def calc_cos(self, batch_size, n_tau=8, cvar=1.0):
        # CVaR-aware sampling
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
        
        x = (x.unsqueeze(1) * cos_x).view(batch_size * num_tau, self.layer_size)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        out = self.output_layer(x)
        
        return out.view(batch_size, num_tau, self.action_size), taus

class ImprovedIQNAgent():
    def __init__(self, state_size, action_size, 
                 layer_size=256,  
                 device="cpu", 
                 seed=42):
        
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Initialize Replay Buffer to satisfy logical consistency, though not used in eval
        self.memory = ReplayBuffer(1, 1, device, seed, 0.99)
        
        # Create Networks
        self.qnetwork_local = ImprovedIQNModel(state_size, action_size, layer_size, seed, device).to(device)
        # Target not needed for eval

    def act(self, state, eval_mode=True, adaptive=False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        
        cvar = 1.0
        if adaptive:
            cvar = self.adjust_cvar_improved(state)

        with torch.no_grad():
            quantiles, _ = self.qnetwork_local.forward(state, self.qnetwork_local.K, cvar)
            action_values = quantiles.mean(dim=1)

        return np.argmax(action_values.cpu().data.numpy())

    def adjust_cvar_improved(self, state_tensor):
        state_np = state_tensor.cpu().numpy()[0]
        # Assuming state format has sonar starting at index 4
        sonar_points = state_np[4:]
        
        closest_d = 10.0 
        
        if len(sonar_points) % 2 == 0:
            points = sonar_points.reshape(-1, 2)
            distances = np.linalg.norm(points, axis=1)
            valid_dists = distances[distances > 0.01]
            if len(valid_dists) > 0:
                closest_d = np.min(valid_dists)
        
        # Improved CVaR Logic:
        # Lower bound at 0.3 to prevent freezing behavior
        cvar = max(0.3, min(1.0, closest_d / 8.0))
        return cvar
