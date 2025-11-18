# Typilus 调参实验工具

**精简版**参数调优工具，包含：
- 🚀 自动化实验执行
- 📊 训练日志和可视化  
- 📈 结果分析和对比

## 快速开始

### 方式1: 快捷启动（推荐）

**Linux/Mac:**
```bash
cd run/type_prediction/typilus/experiment_tools
chmod +x start.sh
./start.sh
```

**Windows:**
```powershell
cd run\type_prediction\typilus\experiment_tools
.\start.ps1
```

### 方式2: 全自动运行

```bash
python run_experiments.py
```

适合在screen中运行（**服务器推荐**）：
```bash
screen -S typilus_exp
python run_experiments.py
# Ctrl+A D (detach)

# 后续查看
screen -r typilus_exp
```

### 方式3: 单独训练

```bash
python train_enhanced.py --yaml_file ../config/typilus
```

### 方式4: 仅分析结果

```bash
python run_experiments.py --analyze
```

## 预设实验

- **baseline** - 基线（默认配置）
- **exp_lr_1e-3** - 学习率 1e-3
- **exp_lr_1e-4** - 学习率 1e-4  
- **exp_batch_64** - 批量大小 64
- **exp_hidden_128** - 隐藏层 128
- **exp_best** - 推荐配置组合

## 自定义实验

编辑 `run_experiments.py` 中的 `EXPERIMENTS` 列表：

```python
EXPERIMENTS = [
    {
        "name": "my_exp",
        "params": {
            "optimization": {"lrs": [5e-4]},
            "dataset": {"max_sentences": 32},
            "model": {"encoder_hidden_size": 96}
        }
    }
]
```

## 输出结构

```
~/naturalcc/typilus/experiments/
├── baseline/
│   ├── config.yml
│   ├── checkpoints/
│   └── logs/
│       ├── metrics.json
│       └── plots/
└── comparison_report.md
```

## 常用命令

```bash
# 仅运行实验
python run_experiments.py --run-only

# 仅分析结果
python run_experiments.py --analyze

# 运行并分析
python run_experiments.py

# 指定GPU
export CUDA_VISIBLE_DEVICES=0
```

## 关键参数

- `lrs`: 学习率 [1e-5, 1e-3]
- `max_sentences`: 批量大小 [8, 128]
- `encoder_hidden_size`: 隐藏层 [32, 256]
- `encoder_layers`: 层数 [1, 8]
- `encoder_dropout`: Dropout [0.0, 0.3]

## 问题排查

**显存不足**: 减小 `max_sentences` 或 `encoder_hidden_size`  
**训练太慢**: 增大 `max_sentences`（如果显存允许）  
**继续训练**: 脚本会自动加载最新检查点
