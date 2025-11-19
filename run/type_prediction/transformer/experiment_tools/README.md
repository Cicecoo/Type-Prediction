# Transformer Type Prediction 实验工具

这个目录包含了用于运行 Transformer 类型预测实验的工具，设计与 Typilus 实验保持一致。

## 目录结构

```
experiment_tools/
├── config_base.yml          # 基础配置
├── experiments_lr.yml       # 实验配置（学习率搜索等）
├── train_enhanced.py        # 增强版训练脚本（带详细日志）
├── run_experiments.py       # 实验管理工具
└── README.md               # 本文件
```

## 快速开始

### 1. 准备数据

确保数据已经处理好：
- Token-sequence 数据位于: `~/workspace/type_pred/naturalcc/data-mmap`
- 数据包含: train.token-sequence, valid.token-sequence, test.token-sequence

### 2. 运行单个实验

```bash
# 在服务器上
cd ~/workspace/type_pred/naturalcc/run/type_prediction/transformer/experiment_tools

# 运行baseline实验
python run_experiments.py --exp transformer_baseline
```

### 3. 运行所有实验

```bash
# 运行experiments_lr.yml中定义的所有实验
python run_experiments.py
```

### 4. 查看实验列表

```bash
python run_experiments.py --list
```

### 5. 分析结果

```bash
# 分析所有实验结果
python run_experiments.py --analyze
```

## 实验配置说明

### experiments_lr.yml

包含多个实验配置：

- **transformer_baseline**: 基线配置 (lr=1e-4, 4层Transformer)
- **transformer_lr_5e5**: 学习率 5e-5
- **transformer_lr_7p5e5**: 学习率 7.5e-5
- **transformer_lr_1e4**: 学习率 1e-4 (验证)
- **transformer_lr_2e4**: 学习率 2e-4
- **transformer_deep**: 更深的模型 (6层)
- **transformer_wide**: 更宽的模型 (d_model=512)

所有实验都使用 `patience=3` 进行早停。

## 日志文件

每个实验会生成三个日志文件（与 Typilus 一致）：

1. **metrics.json**: 简化指标，用于绘图
   ```json
   {
     "epochs": [1, 2, 3, ...],
     "train_loss": [...],
     "valid_loss": [...],
     "train_ppl": [...],
     "valid_ppl": [...],
     "learning_rate": [...]
   }
   ```

2. **detailed_metrics.txt**: 详细的文本格式日志
   ```
   ================================================================================
   Epoch 1 | LR: 0.000100
   --------------------------------------------------------------------------------
   TRAIN:
     loss                          : 4.5231
     ppl                           : 92.15
     ...
   VALID:
     loss                          : 4.2156
     ppl                           : 67.89
     ...
   ```

3. **training_output.log**: 原始训练输出（所有print和log信息）

4. **plots/training_curves.png**: 自动生成的训练曲线图

## 对比 Typilus

| 特性 | Typilus | Transformer |
|------|---------|-------------|
| 输入数据 | nodes + edges (图) | token-sequence (序列) |
| 模型架构 | GatedGNN (图神经网络) | Transformer (序列模型) |
| 学习率 | 1e-3 (较大) | 1e-4 (较小) |
| Warmup | 无 | 2000 steps |
| 早停patience | 3 | 3 |
| 日志格式 | 3个文件 + plots | 3个文件 + plots |

## 实验流程

1. **数据准备**: Token-sequence 已在 Typilus 预处理时生成
2. **运行实验**: `python run_experiments.py`
3. **监控训练**: 查看 training_output.log
4. **分析结果**: `python run_experiments.py --analyze`
5. **对比模型**: 将结果与 Typilus 实验对比

## 服务器命令示例

```bash
# SSH连接服务器
ssh dlserver6

# 激活环境
conda activate naturalcc

# 进入工作目录
cd ~/workspace/type_pred/naturalcc/run/type_prediction/transformer/experiment_tools

# 后台运行所有实验
nohup python run_experiments.py > run_all.log 2>&1 &

# 查看进度
tail -f run_all.log

# 或者查看特定实验的日志
tail -f ~/workspace/type_pred/naturalcc/run/type_prediction/transformer/checkpoints/baseline/training_output.log
```

## 注意事项

1. **数据路径**: 确保 config_base.yml 中的 `task.data` 路径正确
2. **GPU内存**: 如果OOM，减小 `dataset.max_sentences` 或 `model.d_model`
3. **训练时间**: 每个实验预计需要 3-8 小时（取决于数据规模）
4. **早停**: patience=3 意味着验证loss连续3个epoch不降就停止

## 结果分析

实验结束后，使用以下命令分析：

```bash
python run_experiments.py --analyze
```

输出示例：
```
实验名称                         最佳Epoch    训练Loss      验证Loss      Gap
--------------------------------------------------------------------------------
transformer_lr_1e4              8            3.245        3.512         0.267
transformer_baseline            10           3.289        3.534         0.245
transformer_lr_2e4              6            3.312        3.601         0.289
...
```

## 与 Typilus 实验对比

运行完 Transformer 实验后，可以对比：

1. **模型性能**: Transformer vs GNN 在类型预测任务上的效果
2. **收敛速度**: 哪个模型收敛更快
3. **过拟合情况**: Train-Valid Gap 的差异
4. **计算效率**: 训练时间和GPU内存使用

这样可以验证图神经网络在代码理解任务中的优势！
