# Transformer 类型预测实验 - 使用指南

## 概述

这套实验工具用于运行基于 Transformer 的类型预测实验，并与 Typilus (GNN) 进行对比。

**核心区别:**
- **Typilus**: 使用图神经网络 (GNN)，输入数据为 `nodes + edges`
- **Transformer**: 使用序列模型，输入数据为 `token-sequence`

两种方法使用相同的原始数据，只是表示方式不同，非常适合对比图建模 vs 序列建模的效果。

---

## 服务器操作步骤

### 1. 准备环境

```bash
# SSH登录服务器
ssh dlserver6

# 激活conda环境
conda activate naturalcc

# 进入工作目录
cd ~/workspace/type_pred/naturalcc/run/type_prediction/transformer/experiment_tools
```

### 2. 检查数据

确保数据已准备好：
```bash
ls ~/workspace/type_pred/naturalcc/data-mmap/train.token-sequence
# 应该看到文件存在
```

如果数据不存在，需要先运行 Typilus 的数据预处理（token-sequence 会自动生成）。

### 3. 查看可用实验

```bash
python run_experiments.py --list
```

输出示例：
```
可用实验:
--------------------------------------------------------------------------------
transformer_baseline           Transformer基线 - 使用token-sequence
transformer_lr_5e5             学习率 5e-5
transformer_lr_7p5e5           学习率 7.5e-5
transformer_lr_1e4             学习率 1e-4 (baseline重复验证)
transformer_lr_2e4             学习率 2e-4
transformer_deep               更深的Transformer (6层)
transformer_wide               更宽的Transformer (d_model=512)
```

### 4. 运行单个实验

```bash
# 运行baseline实验
python run_experiments.py --exp transformer_baseline
```

实验会输出详细的训练日志，训练完成后会在以下位置生成日志文件：
```
~/workspace/type_pred/naturalcc/run/type_prediction/transformer/checkpoints/baseline/
├── metrics.json              # 简化指标（用于绘图）
├── detailed_metrics.txt      # 详细指标（所有epoch数据）
├── training_output.log       # 原始训练输出
└── plots/
    └── training_curves.png   # 训练曲线图
```

### 5. 后台运行所有实验

```bash
# 后台运行所有实验
nohup python run_experiments.py > run_all_transformer.log 2>&1 &

# 记下进程ID
echo $!

# 查看总进度
tail -f run_all_transformer.log

# 查看特定实验的详细日志
tail -f ~/workspace/type_pred/naturalcc/run/type_prediction/transformer/checkpoints/baseline/training_output.log
```

### 6. 分析结果

```bash
# 分析所有已完成的实验
python run_experiments.py --analyze
```

输出示例：
```
实验结果汇总
================================================================================

实验名称                         最佳Epoch    训练Loss      验证Loss      Gap
--------------------------------------------------------------------------------
transformer_lr_1e4              8            3.245        3.512         0.267
transformer_baseline            10           3.289        3.534         0.245
transformer_lr_2e4              6            3.312        3.601         0.289
```

### 7. 与 Typilus 对比

```bash
# 对比 Transformer 和 Typilus 的结果
python compare_with_typilus.py
```

输出会显示：
- Transformer 各实验的结果
- Typilus 各实验的结果
- 最佳模型对比
- 性能差异分析
- 过拟合情况对比

---

## 日志文件说明

每个实验会生成3个日志文件，格式与 Typilus 一致：

### 1. metrics.json
简化指标，JSON格式，方便程序读取和绘图：
```json
{
  "epochs": [1, 2, 3, ...],
  "train_loss": [4.5, 3.8, 3.2, ...],
  "valid_loss": [4.7, 4.0, 3.5, ...],
  "train_ppl": [90.2, 44.7, 24.5, ...],
  "valid_ppl": [109.9, 54.6, 33.1, ...],
  "learning_rate": [0.0001, 0.0001, 0.0001, ...]
}
```

### 2. detailed_metrics.txt
详细指标，文本格式，包含所有统计信息：
```
================================================================================
Epoch 1 | LR: 0.000100
--------------------------------------------------------------------------------
TRAIN:
  loss                          : 4.5231
  nll_loss                      : 4.5231
  ppl                           : 92.15
  wps                           : 12543.2
  ups                           : 45.3
  wpb                           : 276.8
  bsz                           : 16
  num_updates                   : 100
  ...

VALID:
  loss                          : 4.2156
  nll_loss                      : 4.2156
  ppl                           : 67.89
  ...
```

### 3. training_output.log
原始训练输出，包含所有print和LOGGER信息：
```
================================================================================
Transformer Type Prediction Training
================================================================================
[TrainingLogger] 日志目录: /home/.../checkpoints/baseline
[TrainingLogger] 原始输出保存至: .../training_output.log
Config: {...}
Task: TypeTransformerTask
Model parameters: 15,234,567 (trainable: 15,234,567)
...
```

### 4. plots/training_curves.png
自动生成的可视化图表，包含4个子图：
- Loss曲线（训练 vs 验证）
- Perplexity曲线
- 学习率变化
- Train-Valid Gap（过拟合指示器）

---

## 实验配置说明

### 基础配置 (config_base.yml)

关键参数：
```yaml
task:
  data: '~/workspace/type_pred/naturalcc/data-mmap'
  source_langs: ['token-sequence']  # 使用token序列
  target_langs: ['supernodes']      # 预测类型

model:
  arch: 'typetransformer'
  encoder_type: 'transformer'
  d_model: 256          # Transformer隐藏维度
  n_encoder_layers: 4   # Transformer层数
  d_ff: 1024           # FFN维度
  dropout: 0.1

optimization:
  max_epoch: 50
  lrs: [1e-4]          # 学习率
  warmup_updates: 2000 # Warmup步数
  
checkpoint:
  patience: 3          # 早停patience
```

### 实验配置 (experiments_lr.yml)

每个实验基于 `config_base.yml`，只修改特定参数：

```yaml
experiments:
  - name: transformer_baseline
    description: "Transformer基线"
    changes:
      optimization:
        lrs: [1e-4]
      checkpoint:
        save_dir: '.../baseline'
        patience: 3
```

---

## 常见问题

### Q1: 如何修改batch size？

修改 `config_base.yml`:
```yaml
dataset:
  max_sentences: 16  # 改小以减少GPU内存使用
```

### Q2: 如何添加新实验？

编辑 `experiments_lr.yml`，添加新配置：
```yaml
  - name: my_experiment
    description: "我的实验"
    changes:
      optimization:
        lrs: [5e-5]
      model:
        d_model: 512
```

### Q3: 训练中断了怎么办？

训练会自动保存checkpoint，重新运行相同命令会从上次中断处继续。

### Q4: 如何只运行某几个实验？

```bash
# 运行单个
python run_experiments.py --exp transformer_baseline

# 或修改 experiments_lr.yml，注释掉不需要的实验
```

### Q5: GPU内存不足？

降低以下参数：
- `dataset.max_sentences`: 16 → 8
- `model.d_model`: 256 → 128
- `model.n_encoder_layers`: 4 → 2

### Q6: 如何查看实时训练进度？

```bash
# 方法1: 查看原始日志
tail -f ~/workspace/type_pred/naturalcc/run/type_prediction/transformer/checkpoints/baseline/training_output.log

# 方法2: 查看metrics文件
watch -n 5 'cat ~/workspace/type_pred/naturalcc/run/type_prediction/transformer/checkpoints/baseline/detailed_metrics.txt | tail -30'
```

---

## 实验时间估算

基于典型配置（Typilus数据集）：

| 实验 | 预计时间 | GPU内存 |
|------|---------|---------|
| baseline (4层, d_model=256) | 3-5小时 | ~6GB |
| deep (6层) | 5-7小时 | ~8GB |
| wide (d_model=512) | 6-8小时 | ~10GB |

总计运行所有7个实验：约 30-40 小时

---

## 实验结果分析

### 分析指标

1. **验证Loss**: 越低越好，主要性能指标
2. **训练-验证Gap**: 越小越好，衡量过拟合
3. **最佳Epoch**: 早停epoch，衡量收敛速度
4. **Perplexity**: 越低越好，语言模型评估指标

### 对比维度

运行 `compare_with_typilus.py` 后，从以下维度对比：

1. **性能**: Transformer vs GNN 谁的验证loss更低？
2. **泛化**: 谁的train-valid gap更小？
3. **收敛**: 谁收敛更快？
4. **稳定性**: 谁对超参数更敏感？

### 预期发现

根据类似研究，可能的结果：
- **GNN (Typilus)**: 可能在利用代码结构方面更有优势
- **Transformer**: 可能在捕捉长距离依赖方面更强
- **数据依赖**: 表现可能取决于具体任务和数据特征

---

## 下一步

1. **运行基础实验**: 先运行 baseline 验证流程
2. **参数调优**: 根据baseline结果调整学习率、模型大小
3. **深入分析**: 对比不同架构的优劣
4. **撰写报告**: 整理实验结果，分析图神经网络的优势

祝实验顺利！
