import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

config = {
    "seed" : 123,
    "seeds" : list(range(100)),
}

class IndustrialControlEnv(gym.Env):
    """
    简化的高仿真工业控制模拟器,仅用于接口测试
    
    观测空间:
      - 5 维连续量，每个元素 ∈ [-1, 1]
    动作空间:
      - 3 维连续量，每个元素 ∈ [-1, 1]
    支持 reset(seed=…) 来固定随机种子。
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, max_episode_steps: int = 200):
        print("测试环境: 仅用来测试接口，分数无实际意义")
        super(IndustrialControlEnv, self).__init__()
        # 定义动作空间：3 维，取值 [-1,1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        # 定义观测空间：5 维，取值 [-1,1]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.state = None  # 当前观测
        # 初始化随机数生成器
        self.seed()

    def seed(self, seed=None):
        """
        设置随机种子，返回 [seed]
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, return_info=False, options=None):
        """
        重置环境，返回初始观测（和 info，如果 return_info=True）
        支持传入 seed，以复现初始化过程。
        """
        # 如果传了 seed，就重新设置随机种子
        if seed is not None:
            self.seed(seed)

        # 随机初始化观测
        self.state = self.np_random.uniform(
            low=self.observation_space.low,
            high=self.observation_space.high,
            size=self.observation_space.shape
        ).astype(np.float32)

        self.current_step = 0

        if return_info:
            return self.state, {}
        else:
            return self.state

    def step(self, action):
        """
        执行动作，返回 tuple(obs, reward, done, info)
        当前实现：随机转移，零奖励
        """
        assert self.action_space.contains(action), \
            f"Invalid action {action}"

        # 使用同一随机数生成器做随机转移
        next_state = self.np_random.uniform(
            low=self.observation_space.low,
            high=self.observation_space.high,
            size=self.observation_space.shape
        ).astype(np.float32)

        # 默认零奖励，用户可根据需要替换为真实奖励函数
        reward = next_state[-1]

        self.current_step += 1
        done = (self.current_step >= self.max_episode_steps)

        info = {}
        self.state = next_state
        return next_state, reward, done, info

    def render(self, mode='human'):
        """
        可视化当前观测（此处仅打印）
        """
        print(f"Step: {self.current_step}, State: {self.state}")

    def close(self):
        pass

def get_env():
    return IndustrialControlEnv()

if __name__ == "__main__":
    env = IndustrialControlEnv(max_episode_steps=10)

    # 演示：两次用相同 seed 的 reset，会得到相同的初始观测
    obs1 = env.reset(seed=123)
    obs2 = env.reset(seed=123)
    print("两次相同 seed 的初始观测相等:", np.allclose(obs1, obs2))

    # 正常交互
    obs = env.reset(seed=42)
    print("初始观测：", obs)
    for _ in range(5):
        a = env.action_space.sample()
        obs, r, done, info = env.step(a)
        print("动作", a, "->", "观测", obs, "奖励", r)
        if done:
            break
    env.close()