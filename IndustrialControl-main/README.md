# 工业控制策略评测库

本库提供了一个标准化的测试框架，用于评估控制策略（Agent）在工业控制环境 `IndustrialControlEnv` 上的性能表现。

## 竞赛要求

参赛者需要提交训练好的控制策略模型，测试服务器将使用该模型与模拟器进行交互，生成评估轨迹。最终评分将基于这些轨迹的累计奖励均值。

## 提交要求

### 文件结构
参赛者需要将所有文件打包为 `.zip` 文件，并遵循以下目录结构：

```
submission.zip
├── agent/                
│   ├── agent.py          # 策略实现（继承BaseAgent类实现PolicyAgent类）
│   ├── model.pth         # 基线策略模型权重文件
├── evaluator.py          # 测试模拟器（用于本地验证策略推理实现是否正确）
├── baseline.ipynb        # 基线模型训练示例
├── test_agent.py         # 策略测试脚本
├── environment.yml       # 环境配置文件
```

### 资源限制
- 提交包大小上限：**10 MB**
- 模型测试时间限制：5分钟
- 测试环境：离线环境（无网络访问）
- 建议：对于复杂模型，可考虑导出为ONNX格式以优化性能

### 开发建议
1. 使用提供的git仓库作为开发基础
2. 保持与`environment.yml`相同的环境配置
3. 使用`test_agent.py`和`evaluator.py`进行本地验证
4. 参考`baseline.ipynb`了解模型训练流程

## 快速开始

1. 克隆代码仓库
2. 创建环境：`conda env create -f environment.yml`
3. 运行`baseline.ipynb`进行基线策略训练
5. 使用`test_agent.py`进行本地测试
6. 打包提交

## 注意事项

- 确保策略实现正确继承`BaseAgent`类, 进行推理实现
- 模型权重文件需包含在提交包中
- 建议在提交前进行充分的本地测试
- 如遇模型过大，请考虑模型压缩或精简方案
- 如果模型比较复杂，可以导出为ONNX格式
