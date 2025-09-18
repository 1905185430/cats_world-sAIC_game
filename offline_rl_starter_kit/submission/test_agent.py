
"""
Local quick test for the submission package.
"""
import numpy as np
from agent.agent import build_agent

if __name__ == "__main__":
    agent = build_agent()
    obs = np.random.uniform(-1, 1, size=(agent.cfg.get("obs_dim") or 5,)).astype(np.float32)
    act = agent.get_action(obs)
    print("obs:", obs)
    print("act:", act, "range:", (act.min(), act.max()))
