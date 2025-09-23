"""
Offline evaluation utilities for mixed-quality datasets.

Features:
- Load dataset with segment boundaries (index-based, serial-ordered)
- Load trained Actor (GRU) and scaler from submission/agent
- FQE (Fitted Q Evaluation): learn Q(s,a) from logged transitions and evaluate V^pi via Q(s, pi(s)) at initial states
- Auxiliary metrics: behavior return (dataset), action deviation (MSE between pi(s) and logged a)

Usage:
  python offline_rl_starter_kit/starter/evaluate_offline.py \
    --data IndustrialControl-main/data/data.csv \
    --agent_dir offline_rl_starter_kit/submission/agent \
    --epochs 5 --batch 1024 --hidden 256

Notes:
- We treat the end of each index segment as terminal regardless of 'done' column; if 'done' exists in CSV we OR it with segment terminal.
- Observations are standardized using saved scaler.npz in agent_dir (same as training).
- Actor is loaded from agent_dir/model.pth['gru'] and applied on T=1 sequences to produce deterministic actions in [-1,1].
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse dataset loader logic from training script
from train_td3 import load_dataset
# Reuse Actor definition to load the policy weights directly
from models.mlp import Actor


def set_seed(seed: int = 42):
    import random
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=-1)
        return self.net(x).squeeze(-1)  # [B]


def build_policy(agent_dir: Path, device: torch.device, obs_dim: int, act_dim: int):
    # Load config and scaler
    cfg_path = agent_dir / "config.json"
    scaler_path = agent_dir / "scaler.npz"
    model_path = agent_dir / "model.pth"

    mean = None
    std = None
    if scaler_path.exists():
        dat = np.load(scaler_path)
        mean = dat.get("mean")
        std = dat.get("std")
    if mean is None or std is None:
        raise RuntimeError(f"Scaler not found or invalid at {scaler_path}")

    # Build Actor and load weights
    actor = Actor(obs_dim, act_dim).to(device).eval()
    state = torch.load(model_path, map_location=device)
    if not isinstance(state, dict) or ("gru" not in state):
        raise RuntimeError(f"Unexpected model state in {model_path}; expected key 'gru'")
    actor.load_state_dict(state["gru"])

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    def standardize(obs_np: np.ndarray) -> np.ndarray:
        return (obs_np - mean) / np.clip(std, 1e-6, None)

    @torch.no_grad()
    def pi_action(obs_np: np.ndarray) -> np.ndarray:
        # obs_np: [N, obs_dim] or [obs_dim]
        single = obs_np.ndim == 1
        if single:
            obs_np = obs_np[None, :]
        x = torch.from_numpy(obs_np.astype(np.float32)).to(device)
        a, _ = actor(x[:, None, :])  # [N, 1, act_dim]
        a = a[:, -1, :].clamp(-1.0, 1.0)
        a_np = a.cpu().numpy()
        return a_np[0] if single else a_np

    return standardize, pi_action


def build_transitions(obs: np.ndarray, acts: np.ndarray, rews: np.ndarray, dones: np.ndarray, segments: list[tuple[int,int]]):
    """Construct one-step transitions and terminal mask using segments.
    Return indices of valid transitions i such that next index i+1 is in the same segment; for the last element in a segment, we still create a transition to a dummy next (ignored via done=1).
    """
    N = len(obs)
    # terminal vector by segments
    term_seg = np.zeros((N,), dtype=np.float32)
    for s, e in segments:
        if e - 1 >= s:
            term_seg[e - 1] = 1.0
    dones_vec = dones.reshape(-1).astype(np.float32)
    dones_combined = np.clip(dones_vec + term_seg - dones_vec * term_seg, 0.0, 1.0)  # OR

    # valid indices where next exists (even terminals will have next index, but masked)
    idxs = []
    for s, e in segments:
        if e - s <= 0:
            continue
        # i ranges from s to e-1; but if i==e-1 it's terminal; we can include it for completeness
        for i in range(s, e):
            idxs.append(i)
    idxs = np.array(idxs, dtype=np.int64)

    # Precompute next indices (clip last of segment to itself; done will mask it)
    next_idx = idxs + 1
    # For last index of a segment, set next to itself to keep shapes
    seg_ends = set(e - 1 for s, e in segments if e - 1 >= s)
    for k, i in enumerate(idxs):
        if i in seg_ends:
            next_idx[k] = i

    return idxs, next_idx, dones_combined


def evaluate_offline(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # Load dataset and segments
    obs, actions, rewards, dones, segments = load_dataset(args.data)
    obs_dim, act_dim = obs.shape[1], actions.shape[1]

    # Build policy and scaler
    agent_dir = Path(args.agent_dir)
    standardize, pi_action = build_policy(agent_dir, device, obs_dim, act_dim)

    # Standardize observations
    obs_n = standardize(obs)
    obs_next_n = obs_n.copy()
    idxs, idxs_next, dones_combined = build_transitions(obs_n, actions, rewards, dones, segments)
    obs_next_n = obs_n[idxs_next]

    # Behavior baseline return by segments (undiscounted and discounted)
    ret_undiscounted = []
    ret_discounted = []
    for s, e in segments:
        r = rewards[s:e, 0].astype(np.float32)
        ret_undiscounted.append(float(r.sum()))
        if len(r) == 0:
            ret_discounted.append(0.0)
        else:
            gammas = (args.gamma ** np.arange(len(r), dtype=np.float32))
            ret_discounted.append(float((r * gammas).sum()))
    behavior_ret_mean = float(np.mean(ret_undiscounted)) if ret_undiscounted else 0.0
    behavior_ret_d_mean = float(np.mean(ret_discounted)) if ret_discounted else 0.0

    # Action deviation (MSE between policy and logged actions)
    with torch.no_grad():
        pi_a = []
        B = 4096
        for i in range(0, len(obs_n), B):
            pi_a.append(pi_action(obs_n[i:i+B]))
        pi_a = np.concatenate(pi_a, axis=0).astype(np.float32)
    act_mse = float(np.mean((pi_a - actions)**2))

    # FQE setup
    qnet = QNet(obs_dim, act_dim, hidden=args.hidden).to(device)
    qtarget = QNet(obs_dim, act_dim, hidden=args.hidden).to(device)
    qtarget.load_state_dict(qnet.state_dict())
    optim = torch.optim.Adam(qnet.parameters(), lr=args.lr)

    # Training data tensors (we index per batch on-the-fly)
    obs_t = torch.from_numpy(obs_n).to(device)
    act_t = torch.from_numpy(actions.astype(np.float32)).to(device)
    rew_t = torch.from_numpy(rewards.reshape(-1).astype(np.float32)).to(device)
    done_t = torch.from_numpy(dones_combined.astype(np.float32)).to(device)

    idxs_all = torch.from_numpy(idxs)
    idxs_next_t = torch.from_numpy(idxs_next)

    # simple training loop
    for epoch in range(1, args.epochs + 1):
        # shuffle
        perm = torch.randperm(len(idxs_all))
        for k in range(0, len(idxs_all), args.batch):
            mb_idx = perm[k:k+args.batch]
            if len(mb_idx) == 0:
                continue
            ii = idxs_all[mb_idx]
            jj = idxs_next_t[mb_idx]

            s = obs_t[ii]
            a = act_t[ii]
            r = rew_t[ii]
            d = done_t[ii]
            s2 = obs_t[jj]

            with torch.no_grad():
                a2_np = pi_action(s2.cpu().numpy())
                a2 = torch.from_numpy(a2_np.astype(np.float32)).to(device)
                y = r + args.gamma * (1.0 - d) * qtarget(s2, a2)

            q = qnet(s, a)
            loss = F.mse_loss(q, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qnet.parameters(), max_norm=5.0)
            optim.step()

        # target update (soft)
        with torch.no_grad():
            for tp, sp in zip(qtarget.parameters(), qnet.parameters()):
                tp.data.mul_(1.0 - args.tau).add_(args.tau * sp.data)
        if epoch % max(1, args.epochs // 5) == 0:
            print(f"[FQE] epoch {epoch}/{args.epochs}")

    # Estimate policy value from initial states of each segment
    init_states = []
    for s, e in segments:
        if s < e:
            init_states.append(s)
    v_list = []
    with torch.no_grad():
        if len(init_states) > 0:
            s0 = obs_t[init_states]
            a0_np = pi_action(s0.cpu().numpy())
            a0 = torch.from_numpy(a0_np.astype(np.float32)).to(device)
            v = qnet(s0, a0).cpu().numpy()
            v_list = v.tolist()
    v_mean = float(np.mean(v_list)) if len(v_list) else 0.0

    print("\n===== Offline Evaluation (FQE) =====")
    print(f"Behavior return (undiscounted, mean over segments): {behavior_ret_mean:.4f}")
    print(f"Behavior return (discounted, gamma={args.gamma}, mean): {behavior_ret_d_mean:.4f}")
    print(f"Action deviation MSE (pi vs logged): {act_mse:.6f}")
    print(f"FQE estimated V^pi (mean over initial states): {v_mean:.4f}")
    print("===================================\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="CSV dataset path")
    ap.add_argument("--agent_dir", type=str, default="offline_rl_starter_kit/submission/agent", help="Directory with model.pth, scaler.npz, config.json")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.01, help="soft target update rate")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    evaluate_offline(args)
