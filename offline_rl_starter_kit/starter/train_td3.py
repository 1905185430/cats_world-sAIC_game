"""
TD3 trainer (sequence-aware, GRU-based)

概述
- 读取CSV离线数据（列名包含 obs_*, action_*, reward[, done]；若有 index 表示分段；首列若为空名/Unnamed 作为序号 serial）
- 将观测标准化（均值/方差），按固定长度T滑窗采样序列（严格不跨 index 段），训练基于GRU的Actor/Critic（TD3）
- 仅用每个序列的最后一帧参与TD目标与损失，保持时间依赖
- 训练完成后导出 submission/agent/{model.pth, scaler.npz, config.json}

用法示例
1) 默认参数运行
   python offline_rl_starter_kit/starter/train_td3.py \
       --data offline_rl_starter_kit/data.csv \
       --out offline_rl_starter_kit/

2) 指定序列长度、训练步数、batch、学习率
   python offline_rl_starter_kit/starter/train_td3.py \
       --data IndustrialControl-main/data/data.csv \
       --out offline_rl_starter_kit/ \
       --seq_len 12 --updates 80000 --batch 128 \
       --lr_actor 1e-4 --lr_critic 1e-4

3) 启用CPU（禁用CUDA）并调整噪声
   python offline_rl_starter_kit/starter/train_td3.py \
       --data offline_rl_starter_kit/data.csv --cpu \
       --policy_noise 0.1 --noise_clip 0.3

输出产物
- submission/agent/model.pth: GRU Actor 权重（键为"gru"）
- submission/agent/scaler.npz: 观测标准化参数 mean / std
- submission/agent/config.json: 推理所需配置（obs_dim, act_dim, use_gru, seq_len等）

注意
- 数据应当已将obs与action映射到[-1,1]（若非如此，请在训练前额外做缩放）
- 若CSV中不含done列，则默认全为0（无终止信号）
- 若包含 index 列，则按 index 分段，并在每段内依据首列 serial（若存在）排序；滑窗不会跨段
"""

import os, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from models.mlp import Actor, Critic
from utils.normalization import Scaler

def set_seed(seed: int = 42):
    """设置随机种子，保证可复现"""
    import random
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_dataset(path: str):
    """从CSV中加载数据，并返回numpy数组以及分段信息。

    期望列名：
      - obs_*: 观测分量
      - action_*: 动作分量
      - reward: 每步奖励（标量）
      - done: episode终止标记（可选）
      - index: 分段标识（可选；若存在将按段滑窗，不跨段）
      - 首列若为空名或 Unnamed:*，视为全局序号 serial，用于段内排序

    返回：
      obs, actions, rewards, dones, segments
      其中 segments 为 List[(start, end)]，半开区间，表示每个 index 段在扁平数组中的范围。
    """
    df = pd.read_csv(path)

    # 识别并重命名首列为空名/Unnamed 情况为 serial
    if len(df.columns) > 0:
        first_col = df.columns[0]
        if (first_col is None) or (str(first_col).strip() == "") or str(first_col).lower().startswith("unnamed"):
            df = df.rename(columns={first_col: "serial"})

    # 若包含 index，则在每个 index 内依据 serial（若存在）稳定排序
    if "index" in df.columns:
        if "serial" in df.columns:
            df = df.sort_values(["index", "serial"], kind="stable")
        else:
            # 保持原始顺序的分组拼接
            df = pd.concat([g for _, g in df.groupby("index", sort=False)], axis=0)

    obs_cols = [c for c in df.columns if c.startswith("obs")]
    act_cols = [c for c in df.columns if c.startswith("action")]

    # 基础数组
    obs = df[obs_cols].values.astype(np.float32)
    actions = df[act_cols].values.astype(np.float32)
    rewards = df[["reward"]].values.astype(np.float32)
    dones = df[["done"]].values.astype(np.float32) if "done" in df.columns else np.zeros_like(rewards)

    # 计算分段范围：半开区间 [start, end)
    segments = []
    if "index" in df.columns:
        start = 0
        last_idx = None
        # 遍历已排序后的行，按 index 聚合
        for idx_val, g in df.groupby("index", sort=False):
            n = len(g)
            end = start + n
            segments.append((start, end))
            start = end
            last_idx = idx_val
    else:
        # 单一段（全部）
        segments.append((0, len(df)))

    return obs, actions, rewards, dones, segments

def sample_sequences(obs, acts, rews, dones, seq_len: int, batch_size: int, segments=None):
        """随机滑窗采样批量时序片段（严格不跨段）。

        参数：
            - segments: List[(start, end)] 半开区间，表示可采样的段（通常来自 index 分段）；
                                    若为 None，则视为单一段 [0, N)。

        返回：
            S, A, R, D, S2，形状分别为 [B, T, *]。
            其中 S2 是 S 右移一位的 next_obs 序列。
        """
        N = len(obs)
        assert N > seq_len, f"序列长度T({seq_len})必须小于数据长度N({N})"

        # 构建合法的窗口起点集合，确保 i..i+seq_len 与 i+1..i+seq_len+1 均在同一段内
        if segments is None:
                segments = [(0, N)]
        valid_starts = []
        for (s, e) in segments:
                # 需要 i+seq_len < e，因此 i <= e - seq_len - 1
                max_start = e - seq_len - 1
                if max_start >= s:
                        valid_starts.extend(range(s, max_start + 1))

        if not valid_starts:
                raise ValueError(f"没有足够长的段可供采样：需要至少 T+1={seq_len+1} 长度的段")

        idxs = np.random.choice(valid_starts, size=batch_size)
        S = np.stack([obs[i:i+seq_len] for i in idxs])
        A = np.stack([acts[i:i+seq_len] for i in idxs])
        R = np.stack([rews[i:i+seq_len] for i in idxs])
        D = np.stack([dones[i:i+seq_len] for i in idxs])
        S2 = np.stack([obs[i+1:i+seq_len+1] for i in idxs])
        return S, A, R, D, S2

def soft_update(target_net: torch.nn.Module, source_net: torch.nn.Module, tau: float):
    """软更新 target ← (1-τ)*target + τ*source"""
    with torch.no_grad():
        for t_param, s_param in zip(target_net.parameters(), source_net.parameters()):
            t_param.data.mul_(1.0 - tau).add_(tau * s_param.data)

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    # 1) 加载与标准化数据（含分段信息）
    obs, acts, rews, dones, segments = load_dataset(args.data)
    scaler = Scaler()
    scaler.fit(obs)
    obs_n = scaler.transform(obs)
    acts_n = acts  # 如需可对动作也做缩放

    # 2) 超参数与训练配置
    gamma = args.gamma
    tau = args.tau
    policy_noise = args.policy_noise
    noise_clip = args.noise_clip
    policy_delay = args.policy_delay
    seq_len = args.seq_len
    batch = args.batch

    obs_dim, act_dim = obs_n.shape[1], acts_n.shape[1]

    # 3) 构建网络与优化器（GRU Actor/Critic，见 models/mlp.py）
    actor = Actor(obs_dim, act_dim).to(device)
    actor_t = Actor(obs_dim, act_dim).to(device)
    critic = Critic(obs_dim, act_dim).to(device)
    critic_t = Critic(obs_dim, act_dim).to(device)
    actor_t.load_state_dict(actor.state_dict())
    critic_t.load_state_dict(critic.state_dict())
    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

    for step in range(1, args.updates + 1):
        # 4) 采样批量时序片段 [B, T, *]
        S, A, R, D, S2 = sample_sequences(obs_n, acts_n, rews, dones, seq_len, batch, segments=segments)
        S = torch.from_numpy(S).to(device)  # [B, T, obs_dim]
        A = torch.from_numpy(A).to(device)
        R = torch.from_numpy(R).to(device)
        D = torch.from_numpy(D).to(device)
        S2 = torch.from_numpy(S2).to(device)

        with torch.no_grad():
            # 5) 目标动作与TD目标（使用最后一帧）
            a2, _ = actor_t(S2)  # 目标策略在下一时刻序列上的动作
            a2 = a2 + (torch.randn_like(a2) * policy_noise).clamp(-noise_clip, noise_clip)
            a2 = a2.clamp(-1.0, 1.0)
            q1_t, q2_t = critic_t(S2, a2)
            q_t = torch.min(q1_t, q2_t)
            y = R + gamma * (1.0 - D) * q_t
            y = torch.clamp(y, -100, 100)
            y_last = y[:, -1]  # 仅最后一帧参与loss

        # 6) Critic更新（双Q网络，最后一帧）
        q1, q2 = critic(S, A)
        q1_last = q1[:, -1]
        q2_last = q2[:, -1]
        critic_loss = F.mse_loss(q1_last, y_last) + F.mse_loss(q2_last, y_last)
        opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
        opt_critic.step()

        # 7) Actor延迟更新
        if step % policy_delay == 0:
            a_pi, _ = actor(S)
            q1_pi, _ = critic(S, a_pi)
            q1_pi_last = q1_pi[:, -1]
            policy_loss = -q1_pi_last.mean()
            opt_actor.zero_grad(set_to_none=True)
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
            opt_actor.step()

            # 8) 软更新目标网络
            soft_update(actor_t, actor, tau)
            soft_update(critic_t, critic, tau)

        # 9) 训练日志
        if step % args.log_interval == 0:
            msg = f"[{step}/{args.updates}] critic={critic_loss.item():.4f}"
            if 'policy_loss' in locals():
                msg += f", policy={policy_loss.item():.4f}"
            print(msg)

    # 保存模型与配置
    sub_agent = Path(args.out) / "submission" / "agent"
    sub_agent.mkdir(parents=True, exist_ok=True)
    torch.save({"gru": actor.state_dict()}, sub_agent / "model.pth")
    # 稳健保存scaler（在fit后应存在mean/std；若意外缺失则回退为数据统计）
    mean_arr = getattr(scaler, 'mean', None)
    std_arr = getattr(scaler, 'std', None)
    if mean_arr is None or std_arr is None:
        mean_arr = obs.mean(axis=0).astype(np.float32)
        std_arr = (obs.std(axis=0) + 1e-6).astype(np.float32)
    else:
        mean_arr = np.asarray(mean_arr, dtype=np.float32)
        std_arr = np.asarray(std_arr, dtype=np.float32)
    np.savez(sub_agent / "scaler.npz", mean=mean_arr, std=std_arr)
    cfg = {
        "obs_dim": int(obs_dim),
        "act_dim": int(act_dim),
        "hidden": [128],  # 可在 models.mlp 中自定义
        "use_gru": True,
        "seq_len": int(seq_len)
    }
    (sub_agent / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved model to {(sub_agent / 'model.pth')}")
    print("Done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="TD3 (GRU版) 离线序列训练脚本")
    ap.add_argument("--data", type=str, required=True, help="CSV数据路径，需包含obs_*, action_*, reward[, done]")
    ap.add_argument("--out", type=str, default="/mnt/data/offline_rl_starter_kit", help="输出根目录，将写入submission/agent")
    ap.add_argument("--cpu", action="store_true", help="强制使用CPU（禁用CUDA）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--updates", type=int, default=50000, help="训练总步数（迭代次数）")
    ap.add_argument("--batch", type=int, default=64, help="batch size（序列条数）")
    ap.add_argument("--seq_len", type=int, default=10, help="序列长度T（GRU时间步）")
    ap.add_argument("--lr_actor", type=float, default=5e-5, help="Actor学习率")
    ap.add_argument("--lr_critic", type=float, default=5e-5, help="Critic学习率")
    ap.add_argument("--gamma", type=float, default=0.99, help="折扣因子γ")
    ap.add_argument("--tau", type=float, default=0.005, help="软更新系数τ")
    ap.add_argument("--policy_noise", type=float, default=0.2, help="目标策略噪声强度")
    ap.add_argument("--noise_clip", type=float, default=0.5, help="目标策略噪声裁剪幅度")
    ap.add_argument("--policy_delay", type=int, default=2, help="Actor延迟更新步数")
    ap.add_argument("--log_interval", type=int, default=1000, help="训练日志打印间隔")
    args = ap.parse_args()
    train(args)