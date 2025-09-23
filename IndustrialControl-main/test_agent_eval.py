import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import importlib.util
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
# 已在前面设置字体，这里无需重复设置，且避免SimHei导致的警告
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False
# 获取 agent.py 的绝对路径
agent_path = str(Path(__file__).parent / "agent" / "agent.py")
spec = importlib.util.spec_from_file_location("agent", agent_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load module from {agent_path}")
agent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent_module)
build_agent = agent_module.build_agent
def main():
    # 1. 读取数据
    data_path = Path(__file__).parent.parent / 'offline_rl_starter_kit' / 'data.csv'
    df = pd.read_csv(data_path)
    obs = df[[f'obs_{i}' for i in range(1, 6)]].values.astype(np.float32)
    actions = df[[f'action_{i}' for i in range(1, 4)]].values.astype(np.float32)
    rewards = df['reward'].values.astype(np.float32)

    # 2. 加载模型
    agent = build_agent()

    # 3. 逐条推理，统计行为克隆MSE、MAE、每个分量误差、相关性、稳定性
    pred_actions = np.array([agent.act(o) for o in obs])
    mse = np.mean((pred_actions - actions) ** 2)
    mae = np.mean(np.abs(pred_actions - actions))
    print(f'行为克隆 MSE Loss: {mse:.6f}')
    print(f'行为克隆 MAE Loss: {mae:.6f}')

    # 各动作分量误差和统计信息收集
    mse_list, mae_list, corr_list, reward_corr_list = [], [], [], []
    for i in range(actions.shape[1]):
        mse_i = np.mean((pred_actions[:, i] - actions[:, i]) ** 2)
        mae_i = np.mean(np.abs(pred_actions[:, i] - actions[:, i]))
        corr = np.corrcoef(pred_actions[:, i], actions[:, i])[0, 1]
        reward_corr = np.corrcoef(pred_actions[:, i], rewards)[0, 1]
        mse_list.append(mse_i)
        mae_list.append(mae_i)
        corr_list.append(corr)
        reward_corr_list.append(reward_corr)
        print(f'Action {i+1} - MSE: {mse_i:.6f}, MAE: {mae_i:.6f}')
    for i in range(actions.shape[1]):
        print(f'Action {i+1} 与真实动作相关系数: {corr_list[i]:.4f}')
    for i in range(actions.shape[1]):
        print(f'Action {i+1} 与奖励相关系数: {reward_corr_list[i]:.4f}')
    mean_pred = np.mean(pred_actions, axis=0)
    var_pred = np.var(pred_actions, axis=0)
    print('预测动作均值:', mean_pred)
    print('预测动作方差:', var_pred)

    # 5. 可视化
    try:
        import matplotlib.pyplot as plt
        # 统计信息文本
        stats_text = f"MSE: {mse:.4f}\nMAE: {mae:.4f}\n"
        for i in range(actions.shape[1]):
            stats_text += f"A{i+1} MSE: {mse_list[i]:.4f}, MAE: {mae_list[i]:.4f}\n"
        for i in range(actions.shape[1]):
            stats_text += f"A{i+1} Corr: {corr_list[i]:.3f}, R-Corr: {reward_corr_list[i]:.3f}\n"
        stats_text += f"Mean: {mean_pred.round(3)}\nVar: {var_pred.round(3)}"

        plt.figure(figsize=(18, 8))
        # 各分量对比
        for i in range(actions.shape[1]):
            plt.subplot(2, actions.shape[1], i+1)
            plt.plot(actions[:100, i], label=f'True Action {i+1}')
            plt.plot(pred_actions[:100, i], label=f'Pred Action {i+1}')
            plt.legend()
            plt.title(f'Action {i+1} Comparison')
        # 奖励-动作散点
        for i in range(actions.shape[1]):
            plt.subplot(2, actions.shape[1], actions.shape[1]+i+1)
            plt.scatter(rewards[:100], pred_actions[:100, i], alpha=0.5)
            plt.xlabel('Reward')
            plt.ylabel(f'Pred Action {i+1}')
            plt.title(f'Reward-Action{i+1} Corr')
        # 统计信息文本框
        plt.gcf().text(0.75, 0.5, stats_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        plt.tight_layout(rect=(0,0,0.7,1))
        plt.show()

        # 动作分布直方图
        plt.figure(figsize=(12, 4))
        for i in range(actions.shape[1]):
            plt.subplot(1, actions.shape[1], i+1)
            plt.hist(pred_actions[:, i], bins=30, alpha=0.7, label='Pred')
            plt.hist(actions[:, i], bins=30, alpha=0.5, label='True')
            plt.title(f'Action {i+1} Distribution')
            plt.legend()
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass

if __name__ == "__main__":
    main()
