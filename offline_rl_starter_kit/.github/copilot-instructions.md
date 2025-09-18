# Copilot Instructions - 离线强化学习工业应用 Starter Kit

This project is an **offline reinforcement learning competition starter kit** with a clean two-phase architecture: training algorithms and deployable agents.

## Architecture Overview

### Two-Phase Design
- **Training Phase (`starter/`)**: Implements TD3+BC and IQL algorithms from scratch to train on offline datasets
- **Submission Phase (`submission/`)**: Contains minimal, deployable agent implementation with strict constraints (≤10MB, <5min inference)

### Key Components
- `starter/train_td3bc.py` & `starter/train_iql.py`: Complete training pipelines that produce compatible agent artifacts
- `submission/agent/agent.py`: Production-ready `PolicyAgent` class with standardized interface
- `starter/models/mlp.py`: Reusable neural network components (Actor, Critic, MLP)
- `starter/utils/normalization.py`: Data preprocessing utilities

## Critical Patterns

### Agent Interface Contract
The `PolicyAgent` class follows a strict interface for competition submission:
```python
agent = PolicyAgent()
agent.load('agent')  # loads from config.json, model.pth, scaler.npz
action = agent.get_action(obs)  # single-step inference
```

### Data Flow & Standardization
- **Input**: CSV (obs*, action*, next_obs*, reward, done/terminal) or NPZ format
- **Preprocessing**: Z-score normalization with `Scaler` class, stored in `scaler.npz`
- **Actions**: Always in [-1, 1] range with tanh activation
- **Observation stacking**: Configurable window history via `stack` parameter for POMDP scenarios

### Model Architecture Conventions
- **Default hidden layers**: [128, 128] for submission (balance size vs performance)
- **Training hidden layers**: [256, 256] for better capacity during training
- **Activation**: ReLU for hidden layers, tanh for actor output
- **Optional GRU**: Set `use_gru: true` in config.json for sequential/POMDP tasks

### Configuration System
Agent behavior controlled via `submission/agent/config.json`:
- `obs_dim`, `act_dim`: Environment dimensions
- `stack`: Number of past observations to concatenate
- `include_prev_actions`: Whether to include action history
- `use_gru`: Switch between MLP and GRU policy architectures

## Development Workflows

### Training to Submission Pipeline
```bash
# Train algorithm (saves to specified output directory)
python starter/train_td3bc.py --data dataset.csv --out /path/to/output
python starter/train_iql.py --data dataset.csv --out /path/to/output

# Test locally before submission
python submission/test_agent.py      # Quick smoke test
python submission/evaluator.py      # Performance timing test

# Package for competition
zip -r submission.zip submission/
```

### Hyperparameter Guidelines
- **TD3+BC**: `--bc_coef` (2.5-5.0) controls conservatism; higher = more conservative
- **IQL**: `--expectile` (0.7-0.9) for value learning; `--beta` (1-10) for advantage weighting
- **Model size**: `--hid 128` sufficient for most tasks; increase to 256 for complex domains

### File Compatibility
- Training scripts output directly compatible artifacts for `submission/agent/`
- Model weights saved with algorithm-specific keys ("mlp", "gru") in `model.pth`
- Graceful degradation: agent falls back to deterministic random policy if files missing

## Common Modification Patterns

### Adding New Algorithms
1. Create training script in `starter/` following `train_td3bc.py` pattern
2. Use same `load_dataset()` function for data loading consistency
3. Save actor network to `model.pth` with appropriate key ("mlp" or "gru")
4. Ensure output dimensions and activation match agent expectations

### Extending Agent Capabilities
- Modify `PolicyAgent._build_inputs()` for custom observation preprocessing
- Update `config.json` schema and loading logic in `agent.load()`
- Maintain backward compatibility with existing model files

### POMDP/Sequential Enhancements
- Set `use_gru: true` and adjust `stack` parameter
- Training scripts must handle sequential data appropriately
- GRU maintains hidden state across `get_action()` calls

## Testing & Validation

### Local Validation Stack
- `test_agent.py`: Instantiation and basic functionality
- `evaluator.py`: Performance timing and interface compliance
- Both scripts designed for smoke testing, not official evaluation

### Debugging Common Issues
- **Missing files**: Agent provides fallback behavior with deterministic random actions
- **Dimension mismatches**: Check `obs_dim`/`act_dim` in config vs training data
- **Performance**: Monitor inference time via `evaluator.py` before submission