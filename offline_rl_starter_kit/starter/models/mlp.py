import torch
import torch.nn as nn



class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=obs_dim, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, act_dim)

    def forward(self, x, h=None):
        # x: [B, T, obs_dim]
        y, h = self.gru(x, h)
        a = torch.tanh(self.head(y))  # [B, T, act_dim]
        return a, h

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128, num_layers=1):
        super().__init__()
        self.gru1 = nn.GRU(input_size=obs_dim+act_dim, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.head1 = nn.Linear(hidden, 1)
        self.gru2 = nn.GRU(input_size=obs_dim+act_dim, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.head2 = nn.Linear(hidden, 1)

    def forward(self, x, a, h1=None, h2=None):
        # x: [B, T, obs_dim], a: [B, T, act_dim]
        xa = torch.cat([x, a], dim=-1)
        y1, h1 = self.gru1(xa, h1)
        y2, h2 = self.gru2(xa, h2)
        q1 = self.head1(y1)  # [B, T, 1]
        q2 = self.head2(y2)
        return q1, q2

# 用法示例
# actor = SeqActor(obs_dim, act_dim)
# critic = SeqCritic(obs_dim, act_dim)
# obs_seq = torch.randn(B, T, obs_dim)
# act_seq = torch.randn(B, T, act_dim)
# a, _ = actor(obs_seq)
# q1, q2 = critic(obs_seq, act_seq)