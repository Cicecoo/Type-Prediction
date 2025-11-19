# Transformer类型预测实验套件使用指南

类似Typilus的完整实验管理系统，用于Transformer模型的类型预测任务。

## 快速开始

### 1. 生成实验配置

```bash
# 生成所有实验
python generate_experiment_suite.py

# 只生成特定组
python generate_experiment_suite.py --groups baseline model_size lr
```

这会创建 `experiments/transformer_series/` 目录，包含所有实验配置。

### 2. 运行实验

#### 单个实验
```bash
python run_transformer_experiment.py --language python --exp-name exp_baseline
```

#### 批量运行（串行）
```bash
# 运行baseline
python run_batch_experiments.py --group baseline

# 运行模型大小实验
python run_batch_experiments.py --group model_size

# 运行所有实验
python run_batch_experiments.py --all
```

#### 批量运行（并行，多GPU）
```bash
# 使用GPU 0和1并行运行
python run_batch_experiments.py --group model_size --mode parallel --gpus 0 1
```

#### 预览命令（不实际运行）
```bash
python run_batch_experiments.py --all --dry-run
```

### 3. 分析结果

```bash
# 生成完整报告（HTML + Markdown + 图表）
python analyze_results.py --plot

# 只生成Markdown表格
python analyze_results.py --format markdown

# 自定义输出文件
python analyze_results.py --output my_report.html
```

## 实验设计

### 实验组

1. **Baseline** (`exp_baseline`)
   - 默认配置，作为对比基准

2. **模型大小** (`exp_d_model_*`)
   - d_model = 256, 512, 1024
   - 研究模型容量对性能的影响

3. **层数** (`exp_layers_*`)
   - n_layers = 4, 6, 8
   - 研究模型深度的影响

4. **学习率** (`exp_lr_*`)
   - lr = 1e-4, 5e-4, 1e-3
   - 寻找最优学习率

5. **Dropout** (`exp_dropout_*`)
   - dropout = 0.0, 0.1, 0.2
   - 研究正则化效果

6. **Encoder类型** (`exp_encoder_*`)
   - Transformer vs LSTM
   - 对比不同架构

7. **Batch Size** (`exp_batch_*`)
   - batch_size = 16, 32, 64
   - 研究训练效率和性能平衡

### 默认超参数

```yaml
model:
  arch: typetransformer
  encoder_type: lstm
  d_model: 512
  d_rep: 128
  n_head: 8
  n_encoder_layers: 6
  d_ff: 2048
  dropout: 0.1

optimizer:
  name: fairseq_adam
  lr: 5e-4
  weight_decay: 0.0001

training:
  max_epoch: 50
  batch_size: 32
  patience: 10  # early stopping
```

## 目录结构

```
experiments/transformer_series/
├── experiment_index.json       # 实验索引
├── README.md                   # 自动生成的说明
├── results_comparison.md       # 结果对比表格
├── report.html                 # HTML报告
├── visualizations/             # 可视化图表
│   ├── accuracy_model_size.png
│   ├── accuracy_layers.png
│   └── learning_curves.png
│
└── exp_baseline/              # 单个实验目录
    ├── config.yml             # 实验配置
    ├── meta.json              # 元数据（状态、运行信息）
    ├── checkpoints/           # 模型检查点
    │   ├── checkpoint_best.pt
    │   └── checkpoint_last.pt
    ├── logs/                  # 训练日志
    │   ├── train.log
    │   └── tensorboard/
    ├── results/               # 评估结果
    │   ├── test_results.json
    │   ├── predictions.txt
    │   └── confusion_matrix.png
    └── visualizations/        # 实验专属图表
        ├── loss_curve.png
        └── accuracy_curve.png
```

## 监控实验状态

### 查看实验元数据

```bash
# 查看某个实验的状态
cat experiments/transformer_series/exp_baseline/meta.json
```

输出示例：
```json
{
  "name": "exp_baseline",
  "description": "Baseline configuration",
  "created": "2025-11-20T10:00:00",
  "status": "completed",
  "last_updated": "2025-11-20T15:30:00",
  "run_info": {
    "start_time": "2025-11-20T10:00:00",
    "end_time": "2025-11-20T15:30:00",
    "elapsed_seconds": 19800,
    "return_code": 0
  }
}
```

### 实时查看训练日志

```bash
# 实时查看训练进度
tail -f experiments/transformer_series/exp_baseline/logs/train.log
```

### 使用TensorBoard（如果配置了）

```bash
tensorboard --logdir experiments/transformer_series/exp_baseline/logs/tensorboard
```

## 评估指标

所有实验会跟踪以下指标：

- **Accuracy**: Token级别的类型预测准确率
- **Top-5 Accuracy**: Top-5预测准确率
- **Precision/Recall/F1**: 按类型统计
- **Loss**: 训练和验证损失
- **Training Time**: 总训练时间
- **Memory Usage**: GPU峰值内存使用

## 与Typilus对比

### 生成对比报告

如果您有Typilus的结果文件，可以生成对比报告：

```bash
python compare_with_typilus.py \
    --transformer-results experiments/transformer_series/results_comparison.md \
    --typilus-results path/to/typilus/results.json \
    --output comparison_report.html
```

### 预期结果格式

Typilus结果文件应包含：
```json
{
  "accuracy": 0.xx,
  "top5_accuracy": 0.xx,
  "precision": 0.xx,
  "recall": 0.xx,
  "f1": 0.xx
}
```

## 高级用法

### 自定义实验

创建自定义实验配置：

```python
from generate_experiment_suite import TransformerExperimentSuite

suite = TransformerExperimentSuite()
suite.create_experiment(
    name="exp_custom",
    modifications={
        'model': {'d_model': 768, 'n_encoder_layers': 8},
        'optimizer': {'lr': 3e-4},
        'training': {'batch_size': 48}
    },
    description="Custom configuration with larger model"
)
```

### 继续中断的实验

如果实验中断，可以从最新检查点继续：

```bash
python run_transformer_experiment.py \
    --language python \
    --exp-name exp_baseline \
    --resume-from experiments/transformer_series/exp_baseline/checkpoints/checkpoint_last.pt
```

### 提取特定实验的结果

```python
import json
from pathlib import Path

exp_dir = Path("experiments/transformer_series/exp_baseline")
results_file = exp_dir / "results" / "test_results.json"

with open(results_file) as f:
    results = json.load(f)
    
print(f"Accuracy: {results['accuracy']*100:.2f}%")
print(f"Top-5 Accuracy: {results['top5_accuracy']*100:.2f}%")
```

## 故障排除

### 实验失败

检查失败原因：
```bash
# 查看错误日志
tail -100 experiments/transformer_series/exp_failed/logs/train.log

# 检查状态
cat experiments/transformer_series/exp_failed/meta.json
```

### 重新运行失败的实验

```bash
# 清理失败实验的输出
rm -rf experiments/transformer_series/exp_failed/checkpoints/*
rm -rf experiments/transformer_series/exp_failed/logs/*

# 重新运行
python run_batch_experiments.py --names exp_failed
```

### GPU内存不足

如果遇到OOM错误，尝试减小batch size：

```bash
# 修改实验配置
vi experiments/transformer_series/exp_xxx/config.yml
# 将 batch_size 改为更小的值（如16或8）

# 重新运行
python run_batch_experiments.py --names exp_xxx
```

## 最佳实践

1. **先运行Baseline**: 确保基础配置工作正常
2. **逐组实验**: 不要一次运行所有实验，按组运行便于分析
3. **监控资源**: 使用 `nvidia-smi` 监控GPU使用情况
4. **定期备份**: 备份重要的检查点和结果文件
5. **版本控制**: 将配置文件和分析脚本加入git版本控制

## 常见问题

**Q: 如何调整实验超参数？**
A: 编辑 `experiments/transformer_series/exp_xxx/config.yml` 文件，或使用 `generate_experiment_suite.py` 创建新实验。

**Q: 结果保存在哪里？**
A: 每个实验的结果保存在对应的 `results/` 目录中，包括JSON格式的指标和可视化图表。

**Q: 如何对比两个实验？**
A: 使用 `analyze_results.py` 生成对比报告，或手动对比各实验的 `results/test_results.json` 文件。

**Q: 实验可以暂停吗？**
A: 可以使用Ctrl+C终止训练，之后使用 `--resume-from` 参数从检查点继续。

## 参考资源

- **Typilus论文**: [Graph Neural Networks for Type Inference](https://arxiv.org/abs/2004.10657)
- **NaturalCC文档**: 训练脚本基于NaturalCC框架
- **数据格式**: 参考 `convert_typilus_to_transformer.py` 了解数据转换

## 贡献

如需添加新的实验类型或改进分析工具，请参考现有代码结构：

1. 实验生成: `generate_experiment_suite.py`
2. 批量运行: `run_batch_experiments.py`
3. 结果分析: `analyze_results.py`

---

**版本**: 1.0  
**更新日期**: 2025-11-20
