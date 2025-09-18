
"""
Placeholder evaluator to locally sanity-check interface & speed.
The official evaluator may differ; use this only for smoke tests.
"""
import time
import numpy as np
from agent.agent import build_agent

def rollout(num_steps=1000, obs_dim=5):
    agent = build_agent()
    obs = np.zeros((obs_dim,), dtype=np.float32)
    t0 = time.time()
    for _ in range(num_steps):
        act = agent.get_action(obs)
        # Fake next obs: add a tiny drift
        obs = np.clip(obs + 0.01 * act, -1.0, 1.0).astype(np.float32)
    dt = time.time() - t0
    print(f"[evaluator] Steps: {num_steps}, elapsed: {dt:.3f}s, avg {1e3*dt/num_steps:.3f} ms/step")

if __name__ == "__main__":
    rollout()
