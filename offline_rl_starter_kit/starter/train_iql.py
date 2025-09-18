
"""
IQL (Implicit Q-Learning) minimal trainer.
- Expects offline dataset CSV/NPZ (same loader as TD3+BC).
- Implements: value expectile regression, Q Bellman with V target, advantage-weighted policy regression.
- Saves actor to submission/agent/model.pth compatible with agent/agent.py

Usage:
  python starter/train_iql.py --data path/to/dataset.csv --out /mnt/data/offline_rl_starter_kit
"""
from __future__ import annotations
import os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp import MLP, Actor, Critic
from utils.normalization import Scaler

def set_seed(seed=42):
    import random
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_dataset(path: str):
    p = Path(path)
    if p.suffix.lower() == ".npz":
        data = np.load(p)
        obs = data["observations"]
        actions = data["actions"]
        next_obs = data["next_observations"]
        rewards = data["rewards"].reshape(-1, 1)
        dones = data["terminals"].reshape(-1, 1)
    else:
        df = pd.read_csv(p)
        obs_cols = [c for c in df.columns if c.startswith("obs")]
        act_cols = [c for c in df.columns if c.startswith("action")]
        next_cols = [c for c in df.columns if c.startswith("next_obs")]
        rew_col = "reward" if "reward" in df.columns else [c for c in df.columns if "reward" in c][0]
        done_col = "done" if "done" in df.columns else ("terminal" if "terminal" in df.columns else None)

        obs = df[obs_cols].values.astype(np.float32)
        actions = df[act_cols].values.astype(np.float32)
        next_obs = df[next_cols].values.astype(np.float32) if next_cols else obs.copy()
        rewards = df[[rew_col]].values.astype(np.float32)
        dones = df[[done_col]].values.astype(np.float32) if done_col else np.zeros_like(rewards)

    return obs, actions, next_obs, rewards, dones

def expectile_loss(u, tau):
    # u: residuals (target - pred)
    w = torch.where(u < 0, 1.0 - tau, tau)
    return (w * (u ** 2)).mean()

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    obs, acts, next_obs, rews, dones = load_dataset(args.data)
    obs_dim = obs.shape[1]; act_dim = acts.shape[1]
    print(f"Loaded dataset: obs_dim={obs_dim}, act_dim={act_dim}, N={len(obs)}")

    scaler = Scaler()
    scaler.fit(obs)
    S = torch.from_numpy(scaler.transform(obs)).to(device)
    S2 = torch.from_numpy(scaler.transform(next_obs)).to(device)
    A = torch.from_numpy(acts).to(device)
    R = torch.from_numpy(rews).to(device)
    D = torch.from_numpy(dones).to(device)

    # Networks
    actor = Actor(obs_dim, act_dim, hidden=(args.hid, args.hid)).to(device)
    qnet = Critic(obs_dim, act_dim, hidden=(args.hid, args.hid)).to(device)
    vnet = MLP(obs_dim, 1, hidden=(args.hid, args.hid)).to(device)

    q_target = Critic(obs_dim, act_dim, hidden=(args.hid, args.hid)).to(device)
    v_target = MLP(obs_dim, 1, hidden=(args.hid, args.hid)).to(device)
    q_target.load_state_dict(qnet.state_dict())
    v_target.load_state_dict(vnet.state_dict())

    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    opt_q = torch.optim.Adam(qnet.parameters(), lr=args.lr_q)
    opt_v = torch.optim.Adam(vnet.parameters(), lr=args.lr_v)

    gamma = args.gamma
    tau = args.tau
    tau_v = args.expectile
    beta = args.beta
    n = len(S)
    idxs = np.arange(n)

    for step in range(1, args.updates + 1):
        batch_idx = np.random.choice(idxs, size=args.batch, replace=True)
        s = S[batch_idx]; a = A[batch_idx]; r = R[batch_idx]; s2 = S2[batch_idx]; d = D[batch_idx]

        with torch.no_grad():
            v2 = v_target(s2)
            y_q = r + gamma * (1.0 - d) * v2

        # Q update (to V target)
        q1, q2 = qnet(s, a)
        q_loss = F.mse_loss(q1, y_q) + F.mse_loss(q2, y_q)
        opt_q.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(qnet.parameters(), max_norm=1.0)
        opt_q.step()

        # V update (expectile regression to min(Q1,Q2))
        with torch.no_grad():
            q_min = torch.min(*qnet(s, a))
        v = vnet(s)
        v_loss = expectile_loss(q_min - v, tau_v)
        opt_v.zero_grad(set_to_none=True)
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(vnet.parameters(), max_norm=1.0)
        opt_v.step()

        # Policy update (advantage-weighted regression)
        with torch.no_grad():
            q_pi1, _ = qnet(s, actor(s))
            adv = (q_pi1 - v).clamp(max=100.0)
            weights = torch.exp(beta * adv).clamp(max=100.0)  # stabilize
        a_pi = actor(s)
        pi_loss = (weights * ((a_pi - a) ** 2).sum(dim=-1)).mean()
        opt_actor.zero_grad(set_to_none=True)
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        opt_actor.step()

        # Targets
        if step % 2 == 0:
            # soft updates
            with torch.no_grad():
                for tp, sp in zip(q_target.parameters(), qnet.parameters()):
                    tp.data.mul_(1 - tau).add_(sp.data, alpha=tau)
                for tp, sp in zip(v_target.parameters(), vnet.parameters()):
                    tp.data.mul_(1 - tau).add_(sp.data, alpha=tau)

        if step % 1000 == 0:
            print(f"[{step}/{args.updates}] q={q_loss.item():.4f} v={v_loss.item():.4f} pi={pi_loss.item():.4f}")

    # Save to submission
    sub_agent = Path(args.out) / "submission" / "agent"
    sub_agent.mkdir(parents=True, exist_ok=True)
    torch.save({"mlp": actor.state_dict()}, sub_agent / "model.pth")
    np.savez(sub_agent / "scaler.npz", mean=scaler.mean.astype(np.float32), std=scaler.std.astype(np.float32))

    cfg = {
        "obs_dim": int(obs_dim),
        "act_dim": int(act_dim),
        "hidden": [args.hid, args.hid],
        "use_gru": False,
        "stack": 1,
        "include_prev_actions": False,
        "maxlen": 10
    }
    (sub_agent / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved model to {(sub_agent / 'model.pth')}")
    print("Done. You can zip the 'submission' dir for uploading.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="/mnt/data/offline_rl_starter_kit")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--updates", type=int, default=50_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--lr_actor", type=float, default=1e-4)
    ap.add_argument("--lr_q", type=float, default=1e-4)
    ap.add_argument("--lr_v", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--expectile", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=3.0)
    args = ap.parse_args()
    train(args)
