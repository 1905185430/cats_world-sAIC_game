# agent.py 代码详解

本文件实现了一个用于工业控制任务的智能体（Agent）框架，支持多种神经网络策略（MLP/GRU）、标准化、历史堆叠、模型加载等功能。以下为主要模块和核心逻辑的详细解释。

---

## 1. 基础依赖
- `os, json, random, pathlib.Path`：文件与配置管理、随机数。
- `numpy, torch`：数值计算与深度学习。
- `abc.ABC`：抽象基类。

## 2. BaseAgent（基类）
定义了所有Agent的统一接口和行为：
- 随机种子管理（`seed`）
- 设备选择（CPU/CUDA）
- 观测与动作历史缓存
- 动作产生流程（reshape→推理→clip→缓存→返回）
- `get_action`为抽象方法，需子类实现

## 3. 神经网络策略模型
- `MLPPolicy`：多层感知机，输入为观测（可堆叠），输出为动作，激活函数为tanh。
- `GRUPolicy`：带有GRU单元的策略网络，适合处理时序观测。

## 4. PolicyAgent（主智能体类）
集成了完整的策略推理、模型加载、标准化、历史堆叠等功能。

### 主要属性
- `cfg`：配置字典，包含观测/动作维度、隐藏层结构、是否用GRU、堆叠窗口数、是否包含历史动作等。
- `scaler_mean/std`：观测标准化参数。
- `model_mlp/model_gru`：策略网络实例。
- `obs_buffer/act_buffer`：历史观测/动作缓存。
- `h`：GRU的隐状态。

### 主要方法
- `__init__`：初始化配置、缓存、自动加载模型。
- `_standardize/_de_standardize`：观测标准化与反标准化。
- `_build_inputs`：构建当前时刻的输入（历史观测/动作堆叠）。
- `load`：从指定目录加载配置、标准化参数、模型权重。
- `get_action`：核心推理方法，支持MLP/GRU两种策略，自动标准化和历史堆叠。
- `reset`：重置缓存和GRU状态。
- `act`：兼容BaseAgent接口，直接调用`get_action`。

### 推理流程
1. 输入观测obs，标准化处理。
2. 构建历史堆叠输入（可包含历史动作）。
3. 输入MLP或GRU模型，输出动作。
4. 动作裁剪到[-1,1]，并缓存。

### 模型加载流程
- 自动查找并加载`config.json`、`scaler.npz`、`model.pth`。
- 支持MLP/GRU权重自动适配。

## 5. build_agent 工厂方法
用于外部快速构建PolicyAgent实例。

## 6. 测试入口
`__main__`部分可直接运行，测试模型推理输出。

---

## 典型用法
```python
from agent import build_agent
agent = build_agent()
obs = ... # 获取观测
act = agent.act(obs)
```

---

## 备注
- 支持标准化、历史堆叠、GRU时序建模。
- 兼容多种模型权重格式。
- 适合工业控制等需要时序决策的场景。
