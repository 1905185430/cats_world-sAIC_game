import os
import json
import random
from pathlib import Path
from typing import Optional, Tuple, Deque
from collections import deque
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseAgent(ABC):
    """
    基类，定义了所有 Agent 的统一接口／行为：
      - 随机种子管理
      - 设备（CPU/CUDA）选择
      - 观测与动作的历史缓存
      - 动作产生的统一流程（reshape → 前向推理 → clip → 缓存 → 返回）
    """

    def __init__(self, seed: int = None):
        if seed is not None:
            self.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_history = []
        self.act_history = []

    def seed(self, seed: int = 123) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def reset(self) -> None:
        self.obs_history.clear()
        self.act_history.clear()

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.reshape(-1).astype(np.float32)
        action = self.get_action(obs)
        action = np.clip(action, -1.0, 1.0).reshape(-1).astype(np.float32)
        self.obs_history.append(obs)
        self.act_history.append(action)
        return action

    @abstractmethod
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        ...

    def close(self) -> None:
        pass

# --- myagent的模型定义 ---
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
        return torch.tanh(self.net(x))

class GRUPolicy(nn.Module):
    def __init__(self, in_dim: int, act_dim: int, hidden=128, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, act_dim)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None):
        y, h = self.gru(x, h)
        a = torch.tanh(self.head(y[:, -1]))
        return a, h

class PolicyAgent(BaseAgent):
    """
    集成myagent功能的PolicyAgent，支持load、窗口堆叠、GRU等。
    """
    def __init__(self, device: Optional[str] = None):
        super().__init__()
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
        torch.manual_seed(0)
        np.random.seed(0)
        # 自动加载模型
        self.load(str(Path(__file__).parent))

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
            act_dim = int(self.cfg["act_dim"] or 3)
            self.act_buffer.append(np.zeros((act_dim,), dtype=np.float32))

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
                    (self.model_mlp or self.model_gru).load_state_dict(state)
                self._loaded = True
            except Exception:
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
        obs = self._standardize(obs)
        x = np.stack([self._build_inputs(o) for o in obs], axis=0).astype(np.float32)

        if self.model_gru is not None:
            xt = _to_tensor(x[:, None, :], self.device)   # [B, T=1, in_dim]
            a, self.h = self.model_gru(xt, self.h)
            act = a.cpu().numpy()
        else:
            xt = _to_tensor(x, self.device)                # [B, in_dim]
            act = self.model_mlp(xt).cpu().numpy()

        act = np.clip(act, -1.0, 1.0)
        for a in act:
            self.act_buffer.append(a.astype(np.float32))
        return act[0] if single else act

    def reset(self) -> None:
        super().reset()
        self.obs_buffer.clear()
        self.act_buffer.clear()
        self.h = None

    def act(self, obs: np.ndarray) -> np.ndarray:
        # 兼容BaseAgent接口
        return self.get_action(obs)

    def close(self) -> None:
        pass

def build_agent() -> PolicyAgent:
    ag = PolicyAgent()
    return ag

if __name__ == "__main__":
    agent = build_agent()
    dummy_obs = np.zeros((agent.cfg.get("obs_dim") or 5,), dtype=np.float32)
    a = agent.get_action(dummy_obs)
    print("action:", a, "shape:", a.shape)