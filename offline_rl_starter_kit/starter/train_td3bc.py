
"""
TD3+BC minimal trainer (from scratch, PyTorch).
- Reads dataset from CSV or NPZ.
- Assumes actions/observations already in [-1, 1].
- Saves actor to submission/agent/model.pth and scaler to submission/agent/scaler.npz
- Also writes submission/agent/config.json

Usage:
  python starter/train_td3bc.py --data path/to/dataset.csv --out /mnt/data/offline_rl_starter_kit
"""
#python3 offline_rl_starter_kit/starter/train_td3bc.py --data /home/xuan/cats_world-sAIC_game/offline_rl_starter_kit/data.csv --out /home/xuan/cats_world-sAIC_game/offline_rl_starter_kit/

#python3 offline_rl_starter_kit/starter/train_td3bc.py --data /home/xuan/cats_world-sAIC_game/offline_rl_starter_kit/data.csv --out /home/xuan/cats_world-sAIC_game/offline_rl_starter_kit/ --updates 20000
from __future__ import annotations
import os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mlp import Actor, Critic
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
        idx = data["episode_index"] if "episode_index" in data else np.zeros((len(obs),), dtype=np.int32)
    else:
        df = pd.read_csv(p)
        # Try flexible column names
        obs_cols = [c for c in df.columns if c.startswith("obs")]
        act_cols = [c for c in df.columns if c.startswith("action")]
        next_cols = [c for c in df.columns if c.startswith("next_obs")]
        rew_col = "reward" if "reward" in df.columns else [c for c in df.columns if "reward" in c][0]
        done_col = "done" if "done" in df.columns else ("terminal" if "terminal" in df.columns else None)
        idx_col = "index" if "index" in df.columns else ("episode" if "episode" in df.columns else None)

        obs = df[obs_cols].values.astype(np.float32)
        actions = df[act_cols].values.astype(np.float32)
        next_obs = df[next_cols].values.astype(np.float32) if next_cols else None
        rewards = df[[rew_col]].values.astype(np.float32)
        dones = df[[done_col]].values.astype(np.float32) if done_col else np.zeros_like(rewards)
        idx = df[idx_col].values.astype(np.int32) if idx_col else np.zeros((len(df),), dtype=np.int32)

    return obs, actions, next_obs, rewards, dones, idx

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1 - tau).add_(sp.data, alpha=tau)

def init_weights(m):
    """Better weight initialization for stability"""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=0.1)  # Small gain for stability
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    obs, acts, next_obs, rews, dones, idx = load_dataset(args.data)
    obs_dim = obs.shape[1]; act_dim = acts.shape[1]
    print(f"Loaded dataset: obs_dim={obs_dim}, act_dim={act_dim}, N={len(obs)}")

    scaler = Scaler()
    scaler.fit(obs)
    obs_n = scaler.transform(obs)
    next_obs_n = scaler.transform(next_obs) if next_obs is not None else None

    # Networks
    hidden = tuple(args.hidden) if hasattr(args, 'hidden') else (args.hid, args.hid)
    actor = Actor(obs_dim, act_dim, hidden=hidden).to(device)
    actor_t = Actor(obs_dim, act_dim, hidden=hidden).to(device)
    critic = Critic(obs_dim, act_dim, hidden=hidden).to(device)
    critic_t = Critic(obs_dim, act_dim, hidden=hidden).to(device)
    
    # Better weight initialization for critics
    critic.apply(init_weights)
    critic_t.apply(init_weights)
    
    actor_t.load_state_dict(actor.state_dict())
    critic_t.load_state_dict(critic.state_dict())

    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

    # Buffers as tensors
    S = torch.from_numpy(obs_n).to(device)
    A = torch.from_numpy(acts).to(device)
    R = torch.from_numpy(rews).to(device)
    if next_obs_n is None:
        # approximate next obs by repeating S (safe fallback)
        next_obs_n = obs_n.copy()
    S2 = torch.from_numpy(next_obs_n).to(device)
    D = torch.from_numpy(dones).to(device)

    # Normalize rewards for stability
    R = (R - R.mean()) / (R.std() + 1e-8)

    # TD3+BC hyperparams
    gamma = args.gamma
    tau = args.tau
    policy_noise = args.policy_noise
    noise_clip = args.noise_clip
    policy_delay = args.policy_delay
    bc_coef = args.bc_coef

    n = len(S)
    idxs = np.arange(n)

    for step in range(1, args.updates + 1):
        batch_idx = np.random.choice(idxs, size=args.batch, replace=True)
        s = S[batch_idx]; a = A[batch_idx]; r = R[batch_idx]; s2 = S2[batch_idx]; d = D[batch_idx]

        with torch.no_grad():
            # target policy smoothing
            a2 = actor_t(s2)
            noise = (torch.randn_like(a2) * policy_noise).clamp(-noise_clip, noise_clip)
            a2 = (a2 + noise).clamp(-1.0, 1.0)
            q1_t, q2_t = critic_t(s2, a2)
            q_t = torch.min(q1_t, q2_t)
            y = r + gamma * (1.0 - d) * q_t
            
            # Clamp target values to prevent explosion
            y = torch.clamp(y, -100, 100)

        # critic update
        q1, q2 = critic(s, a)
        
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        
        # Early stopping if loss is exploding
        if critic_loss.item() > 1000:
            print(f"Critic loss exploding at step {step}: {critic_loss.item():.2f}")
            print("Reducing learning rate and continuing...")
            for param_group in opt_critic.param_groups:
                param_group['lr'] *= 0.5
            for param_group in opt_actor.param_groups:
                param_group['lr'] *= 0.5
        
        opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        # More aggressive gradient clipping
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
        opt_critic.step()

        # delayed policy update
        if step % policy_delay == 0:
            a_pi = actor(s)
            q1_pi, _ = critic(s, a_pi)
            # 只用强化学习目标，不加行为克隆项
            policy_loss = -q1_pi.mean()

            opt_actor.zero_grad(set_to_none=True)
            policy_loss.backward()
            # More aggressive gradient clipping for actor too
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
            opt_actor.step()

            # target nets
            soft_update(actor_t, actor, tau)
            soft_update(critic_t, critic, tau)

        if step % 1000 == 0:
            q_mean = q1.mean().item()
            q_std = q1.std().item()
            print(f"[{step}/{args.updates}] critic={critic_loss.item():.4f}, Q_mean={q_mean:.2f}, Q_std={q_std:.2f}")

    # Save artifacts to submission package
    sub_agent = Path(args.out) / "submission" / "agent"
    sub_agent.mkdir(parents=True, exist_ok=True)
    torch.save({"mlp": actor.state_dict()}, sub_agent / "model.pth")
    np.savez(sub_agent / "scaler.npz", mean=scaler.mean.astype(np.float32), std=scaler.std.astype(np.float32))

    cfg = {
        "obs_dim": int(obs_dim),
        "act_dim": int(act_dim),
        "hidden": list(hidden),
        "use_gru": False,
        "stack": 3,
        "include_prev_actions": False,
        "maxlen": 10
    }
    (sub_agent / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved model to {(sub_agent / 'model.pth')}")
    print("Done. You can zip the 'submission' dir for uploading.")
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="path to dataset (.csv or .npz)")
    ap.add_argument("--out", type=str, default="/mnt/data/offline_rl_starter_kit", help="root output dir")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--updates", type=int, default=50_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--hidden", type=int, nargs='+', help="hidden layer sizes, e.g. --hidden 256 256 128")
    ap.add_argument("--lr_actor", type=float, default=5e-5)
    ap.add_argument("--lr_critic", type=float, default=5e-5)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--policy_noise", type=float, default=0.2)
    ap.add_argument("--noise_clip", type=float, default=0.5)
    ap.add_argument("--policy_delay", type=int, default=2)
    ap.add_argument("--bc_coef", type=float, default=2.5)
    args = ap.parse_args()
    train(args)
