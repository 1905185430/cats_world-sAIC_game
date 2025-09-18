
# 离线强化学习工业应用 · Starter Kit

这个包里包含：
- ✅ `submission/agent/agent.py`：可提交的最小策略骨架（MLP/GRU、tanh 输出、标准化、窗口堆叠）。
- ✅ `submission/evaluator.py` & `submission/test_agent.py`：本地自查接口与速度（非官方评测器）。
- ✅ `starter/train_td3bc.py`：从零实现的 TD3+BC 训练脚本（保守强基线）。
- ✅ `starter/train_iql.py`：从零实现的 IQL 训练脚本（更稳健）。
- ✅ `starter/models/`, `starter/utils/`：小型网络与工具。

## 快速开始（训练 → 打包）
```bash
# 1) 准备数据（CSV 或 NPZ）
#   - CSV 需包含前缀为 obs*, action*, next_obs* 的列，以及 reward、(done/terminal)、(index/episode 可选)
#   - NPZ 需包含 observations, actions, next_observations, rewards, terminals

# 2) 训练（任选其一）
python starter/train_td3bc.py --data /path/to/dataset.csv --out /mnt/data/offline_rl_starter_kit
# 或
python starter/train_iql.py   --data /path/to/dataset.csv --out /mnt/data/offline_rl_starter_kit

# 3) 本地自查
python submission/test_agent.py
python submission/evaluator.py

# 4) 打包提交
cd /mnt/data/offline_rl_starter_kit
zip -r submission.zip submission
# 确保压缩包 ≤10MB，推理速度 < 5min
```

## 超参小贴士
- TD3+BC：`--bc_coef (默认2.5)` 越大越保守，越贴近数据分布；可从 2.5~5.0 网格。
- IQL：`--expectile` 常用 0.7~0.9；`--beta` 控制优势加权强度，常用 1~10。
- 模型体积：`--hid 128` 基本够用；如果数据更复杂可升到 256，但注意体积与推理时间。

## 提醒
- 赛题官方评测器 / 接口以主办方发布为准，这里只提供自测骨架。
- 如需 POMDP/延迟增强：把 `submission/agent/config.json` 中 `stack` 调大，或把 `use_gru` 设为 `true` 并在训练侧做对应处理。
```
