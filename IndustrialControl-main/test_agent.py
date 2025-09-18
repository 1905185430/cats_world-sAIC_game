import importlib.machinery
import importlib.util
import gym
import numpy as np
from evaluator import get_env, config


def load_agent(path):
    # 动态加载 agent.py 并实例化 Agent
    loader = importlib.machinery.SourceFileLoader('agent', path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod.PolicyAgent()

def evaluate_agent(agent, env, seeds):
    results = []
    for seed in seeds:
        obs = env.reset(seed=seed)     
        agent.reset()
        total_reward = 0.0

        while True:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break

        results.append(total_reward)
    return results

def main():
    # 1. load agent
    agent = load_agent("./agent/agent.py")
    agent.seed(config["seed"])

    # 2. create one env，循环复用
    env = get_env()

    # 3. evaluate
    results = evaluate_agent(agent, env, config["seeds"])

    # 4. clean up
    env.close()
    agent.close()

    # 5. 可选：打印或返回
    mean_score = np.mean(results)
    std_score = np.std(results)
    print(f"Score: {mean_score:.4f} ± {std_score:.4f}")
    return results

if __name__ == "__main__":
    main()