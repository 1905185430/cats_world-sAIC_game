
"""
Minimal submission agent for "离线强化学习工业应用" 赛题
- Exposes `PolicyAgent` with methods: load(ckpt_dir) and get_action(obs: np.ndarray) -> np.ndarray
- Small MLP policy with tanh output in [-1, 1].
- Supports optional window stacking and (lightweight) GRU variant for POMDP.
- Robust to missing files: if no weights are found, uses a random but stable policy seed for smoke tests.
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Optional, Tuple, Deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

def _to_tensor(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device=device, dtype=torch.float32)

class MLPPolicy(nn.Module):
    def __init__(self, in_dim: int, act_dim: int, hidden=(128, 128)):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, act_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # tanh to keep actions in [-1, 1]
        return torch.tanh(self.net(x))

class GRUPolicy(nn.Module):
    def __init__(self, in_dim: int, act_dim: int, hidden=128, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, act_dim)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None):
        # x: [B, T, in_dim]
        y, h = self.gru(x, h)
        a = torch.tanh(self.head(y[:, -1]))
        return a, h

class PolicyAgent:
    """
    Submission entry. Evaluator will do:
        agent = PolicyAgent()
        agent.load('agent')
        action = agent.get_action(obs)
    """
    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cfg = {
            "obs_dim": None,
            "act_dim": None,
            "hidden": [128, 128],
            "use_gru": False,
            "stack": 1,               # how many past obs to stack
            "include_prev_actions": False,
            "maxlen": 10,             # internal buffer length (>= stack)
        }
        self.scaler_mean = None
        self.scaler_std = None
        self.model_mlp: Optional[MLPPolicy] = None
        self.model_gru: Optional[GRUPolicy] = None
        self.h: Optional[torch.Tensor] = None
        self.obs_buffer: Deque[np.ndarray] = deque(maxlen=self.cfg["maxlen"])
        self.act_buffer: Deque[np.ndarray] = deque(maxlen=self.cfg["maxlen"])
        self._loaded = False
        # For deterministic fallback policy
        torch.manual_seed(0)
        np.random.seed(0)

    # ---------------------- utilities ----------------------
    def _standardize(self, obs: np.ndarray) -> np.ndarray:
        if self.scaler_mean is None or self.scaler_std is None:
            return obs
        std = np.clip(self.scaler_std, 1e-6, None)
        return (obs - self.scaler_mean) / std

    def _de_standardize(self, obs: np.ndarray) -> np.ndarray:
        if self.scaler_mean is None or self.scaler_std is None:
            return obs
        return obs * self.scaler_std + self.scaler_mean

    def _build_inputs(self, obs_now: np.ndarray) -> np.ndarray:
        # push to buffers
        self.obs_buffer.append(obs_now.astype(np.float32))
        if len(self.act_buffer) == 0:
            # pad zero action for the very first step if needed
            self.act_buffer.append(np.zeros((self.cfg["act_dim"],), dtype=np.float32))

        # build stacked input
        K = int(self.cfg.get("stack", 1))
        inc_act = bool(self.cfg.get("include_prev_actions", False))

        obs_list = list(self.obs_buffer)[-K:]
        while len(obs_list) < K:
            obs_list.insert(0, obs_list[0])  # left-pad with oldest

        if inc_act:
            act_list = list(self.act_buffer)[-K:]
            while len(act_list) < K:
                act_list.insert(0, act_list[0])

        x = np.concatenate(obs_list, axis=-1)
        if inc_act:
            x = np.concatenate([x, np.concatenate(act_list, axis=-1)], axis=-1)
        return x

    # ---------------------- API ----------------------
    def load(self, ckpt_dir: str):
        ckpt_path = Path(ckpt_dir)
        cfg_path = ckpt_path / "config.json"
        scaler_path = ckpt_path / "scaler.npz"
        model_path = ckpt_path / "model.pth"

        # load config if exists
        if cfg_path.exists():
            try:
                self.cfg.update(json.loads(cfg_path.read_text(encoding="utf-8")))
            except Exception:
                pass

        # load scaler
        if scaler_path.exists():
            try:
                dat = np.load(scaler_path)
                self.scaler_mean = dat.get("mean")
                self.scaler_std = dat.get("std")
            except Exception:
                self.scaler_mean = None
                self.scaler_std = None

        # infer input dim
        in_dim = int(self.cfg["obs_dim"] or 5) * int(self.cfg.get("stack", 1))
        if self.cfg.get("include_prev_actions", False):
            in_dim += int(self.cfg["act_dim"] or 3) * int(self.cfg.get("stack", 1))
        act_dim = int(self.cfg["act_dim"] or 3)

        if bool(self.cfg.get("use_gru", False)):
            self.model_gru = GRUPolicy(in_dim=in_dim, act_dim=act_dim, hidden=int(self.cfg["hidden"][0]))
            self.model_gru.to(self.device).eval()
        else:
            self.model_mlp = MLPPolicy(in_dim=in_dim, act_dim=act_dim, hidden=tuple(self.cfg["hidden"]))
            self.model_mlp.to(self.device).eval()

        # load weights if present
        if model_path.exists():
            try:
                state = torch.load(model_path, map_location=self.device)
                if self.model_mlp is not None and "mlp" in state:
                    self.model_mlp.load_state_dict(state["mlp"])
                elif self.model_gru is not None and "gru" in state:
                    self.model_gru.load_state_dict(state["gru"])
                elif isinstance(state, dict):
                    # try direct
                    (self.model_mlp or self.model_gru).load_state_dict(state)
                self._loaded = True
            except Exception:
                # fall back to random weights
                self._loaded = False
        else:
            self._loaded = False

    @torch.inference_mode()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Args:
            obs: shape (obs_dim,) or (B, obs_dim)
        Returns:
            action: shape (act_dim,)
        """
        single = obs.ndim == 1
        if single:
            obs = obs[None, ...]
        obs = obs.astype(np.float32)
        # standardize
        obs = self._standardize(obs)
        # stack with history
        x = np.stack([self._build_inputs(o) for o in obs], axis=0).astype(np.float32)

        if self.model_gru is not None:
            xt = _to_tensor(x[:, None, :], self.device)   # [B, T=1, in_dim]
            a, self.h = self.model_gru(xt, self.h)
            act = a.cpu().numpy()
        else:
            xt = _to_tensor(x, self.device)                # [B, in_dim]
            act = self.model_mlp(xt).cpu().numpy()

        # clip to [-1, 1]
        act = np.clip(act, -1.0, 1.0)

        # update last action buffer
        for a in act:
            self.act_buffer.append(a.astype(np.float32))

        return act[0] if single else act

# Optional factory
def build_agent() -> PolicyAgent:
    ag = PolicyAgent()
    ag.load(str(Path(__file__).parent))
    return ag

if __name__ == "__main__":
    # Smoke test
    agent = build_agent()
    dummy_obs = np.zeros((agent.cfg.get("obs_dim") or 5,), dtype=np.float32)
    a = agent.get_action(dummy_obs)
    print("action:", a, "shape:", a.shape)
