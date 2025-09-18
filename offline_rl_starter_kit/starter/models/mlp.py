
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 256), act=nn.ReLU):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), act()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256,256)):
        super().__init__()
        self.mlp = MLP(obs_dim, act_dim, hidden)
    def forward(self, x):
        return torch.tanh(self.mlp(x))

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256,256)):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, hidden)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden)
    def forward(self, x, a):
        xa = torch.cat([x, a], dim=-1)
        return self.q1(xa), self.q2(xa)
