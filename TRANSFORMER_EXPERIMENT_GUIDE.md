# Transformer 类型预测实验工具集

完整的实验管理、评估和可视化工具，类似typilus的实验管理方式。

## 📁 目录结构

实验运行后会自动创建以下目录结构：

```
experiments/transformer/
├── baseline/
│   ├── checkpoints/          # 模型checkpoint
│   │   ├── checkpoint_best.pt
│   │   └── checkpoint_last.pt
│   ├── logs/                 # 训练日志
│   │   ├── train.log
│   │   ├── eval.log
│   │   └── tensorboard/      # TensorBoard日志
│   ├── results/              # 测试结果
│   │   ├── metrics.json
│   │   └── evaluation_report.txt
│   ├── config.yml            # 实验配置
│   └── info.txt              # 实验信息
├── exp_larger_model/
│   └── ...
└── batch_summary.txt         # 批量实验汇总
```

## 🚀 快速开始

### 1. 运行单个实验

```bash
python run_transformer_experiment.py \
  --exp-name baseline \
  --base-dir /mnt/data1/zhaojunzhang/experiments/transformer \
  --data-dir /mnt/data1/zhaojunzhang/typilus-data/transformer \
  --encoder-layers 2 \
  --encoder-embed-dim 512 \
  --dropout 0.1 \
  --lr 0.0001 \
  --batch-size 16 \
  --max-epoch 50
```

**自动执行**：
- ✅ 创建实验目录结构
- ✅ 生成训练配置文件
- ✅ 保存实验信息
- ✅ 执行训练
- ✅ 训练后自动评估
- ✅ 记录所有结果

### 2. 运行批量实验

#### 使用预定义配置

```bash
python batch_experiments.py \
  --mode predefined \
  --configs baseline larger_model high_dropout
```

预定义的实验配置：
- `baseline`: 基础配置（2层，512维，dropout=0.1）
- `larger_model`: 更大模型（4层，768维）
- `high_dropout`: 高dropout（dropout=0.3）
- `higher_lr`: 更高学习率（lr=0.0005）
- `lower_lr`: 更低学习率（lr=0.00005）
- `larger_batch`: 更大batch（batch_size=32）

#### 网格搜索

```bash
python batch_experiments.py \
  --mode grid \
  --grid-lr 0.0001 0.0005 0.001 \
  --grid-dropout 0.1 0.2 0.3 \
  --grid-layers 2 4 \
  --grid-dim 512 768
```

这会自动生成所有组合并运行（3×3×2×2=36个实验）。

### 3. 可视化训练过程

#### 单个实验

```bash
python visualize_training.py \
  --log-file /path/to/exp/logs/train.log \
  --output-dir /path/to/exp/plots \
  --exp-name "My Experiment"
```

**生成图表**：
- `loss_curve.png` - 训练loss曲线
- `accuracy_curve.png` - 准确率曲线  
- `lr_curve.png` - 学习率变化曲线

#### 对比多个实验

```bash
python visualize_training.py \
  --compare \
  --exp-dirs \
    /path/to/exp1 \
    /path/to/exp2 \
    /path/to/exp3 \
  --output-dir ./comparison_plots \
  --metric accuracy
```

**生成图表**：
- `accuracy_comparison.png` - 准确率对比曲线
- `metrics_summary.png` - 指标汇总条形图

### 4. 详细评估

```bash
python evaluate_predictions.py \
  --pred-file /path/to/predictions.txt \
  --ref-file /path/to/test.type \
  --output-dir /path/to/results
```

**生成报告**：
- `evaluation_report.txt` - 详细评估报告
- `metrics.json` - JSON格式指标

**计算指标**：
- Token级准确率
- 序列级准确率（完全匹配）
- Precision/Recall/F1
- 每个类型的统计信息

## 📊 实验配置说明

### 模型参数

| 参数 | 说明 | 默认值 | 范围 |
|------|------|--------|------|
| `--encoder-type` | 编码器类型 | lstm | lstm, transformer |
| `--encoder-layers` | 编码器层数 | 2 | 1-12 |
| `--encoder-embed-dim` | 嵌入维度 | 512 | 128-1024 |
| `--dropout` | Dropout率 | 0.1 | 0.0-0.5 |

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--lr` | 学习率 | 0.0001 |
| `--batch-size` | 批次大小 | 16 |
| `--max-epoch` | 最大epoch数 | 50 |
| `--warmup-updates` | 预热步数 | 1000 |

### 控制选项

| 参数 | 说明 |
|------|------|
| `--skip-train` | 跳过训练（只评估） |
| `--skip-eval` | 跳过评估 |

## 📈 查看实验结果

### 1. 查看实验信息

```bash
cat experiments/transformer/baseline/info.txt
```

### 2. 查看训练日志

```bash
tail -f experiments/transformer/baseline/logs/train.log
```

### 3. 查看评估结果

```bash
cat experiments/transformer/baseline/results/evaluation_report.txt
```

### 4. 查看TensorBoard

```bash
tensorboard --logdir experiments/transformer/baseline/logs/tensorboard
```

### 5. 查看批量实验汇总

```bash
cat experiments/transformer/batch_summary.txt
```

## 🔄 完整工作流示例

### 示例1：快速测试

```bash
# 1. 运行快速测试（只训练100步）
python run_transformer_experiment.py \
  --exp-name quick_test \
  --max-epoch 1 \
  --batch-size 8

# 2. 查看结果
cat experiments/transformer/quick_test/results/evaluation_report.txt
```

### 示例2：正式实验

```bash
# 1. 运行基准实验
python run_transformer_experiment.py \
  --exp-name baseline_50epochs \
  --max-epoch 50 \
  --batch-size 16

# 2. 可视化
python visualize_training.py \
  --log-file experiments/transformer/baseline_50epochs/logs/train.log \
  --output-dir experiments/transformer/baseline_50epochs/plots
```

### 示例3：超参数调优

```bash
# 1. 运行网格搜索
python batch_experiments.py \
  --mode grid \
  --grid-lr 0.00005 0.0001 0.0002 \
  --grid-dropout 0.1 0.2

# 2. 查看汇总
cat experiments/transformer/batch_summary.txt

# 3. 可视化对比
python visualize_training.py \
  --compare \
  --exp-dirs experiments/transformer/grid_* \
  --output-dir experiments/transformer/comparison_plots
```

### 示例4：与typilus对比

```bash
# 1. 运行多个配置
python batch_experiments.py \
  --mode predefined \
  --configs baseline larger_model

# 2. 从typilus获取结果
# (typilus的结果在 /mnt/data1/zhaojunzhang/typilus-data/typilus/type_inference/...)

# 3. 手动对比结果
# Typilus: 查看其evaluation结果
# Transformer: cat experiments/transformer/batch_summary.txt
```

## 🛠️ 故障排查

### 问题1：训练卡住不动

**原因**：数据加载慢或batch太大  
**解决**：
```bash
# 减小batch size
--batch-size 8

# 或增加workers
修改config.yml中的 dataset.num_workers
```

### 问题2：OOM (内存不足)

**解决**：
```bash
# 减小模型或batch
--encoder-embed-dim 256 \
--batch-size 8

# 或使用梯度累积
修改config.yml中的 optimization.update_freq: [4]
```

### 问题3：准确率很低

**检查**：
1. 数据是否正确转换
2. 词典是否匹配
3. 学习率是否合适

```bash
# 尝试调整学习率
--lr 0.00005  # 降低
--lr 0.0005   # 提高
```

## 📝 输出文件说明

### config.yml
完整的训练配置，可以用于复现实验。

### info.txt
实验元信息：创建时间、数据路径、主要超参数。

### train.log
完整的训练日志，包含每个epoch的loss、accuracy等。

### eval.log
评估日志，包含测试集上的性能指标。

### metrics.json
JSON格式的评估指标，方便程序读取。

### evaluation_report.txt
人类可读的详细评估报告：
- 总体指标（准确率、F1等）
- 混淆矩阵统计
- 每个类型的详细统计

## 🎯 实验建议

### 基础实验
1. **baseline**: 先跑基准配置，建立baseline
2. **quick_test**: 用小epoch快速验证数据和代码

### 模型大小
1. 逐步增大：2层→4层→6层
2. 维度：512→768→1024

### 正则化
1. Dropout: 0.1→0.2→0.3
2. Weight decay: 0.0→0.01→0.1

### 学习率
1. 从0.0001开始
2. 如果不收敛，降低到0.00005
3. 如果收敛太慢，提高到0.0003

## 📚 相关脚本

| 脚本 | 功能 |
|------|------|
| `convert_typilus_to_transformer.py` | 数据格式转换 |
| `prepare_transformer_dict.py` | 词典格式转换 |
| `run_transformer_experiment.py` | 单个实验管理 |
| `batch_experiments.py` | 批量实验运行 |
| `evaluate_predictions.py` | 详细评估 |
| `visualize_training.py` | 训练可视化 |

## 💡 最佳实践

1. **命名规范**：使用描述性的实验名称
   - ✅ `baseline_50ep_bs16`
   - ✅ `lstm4_dim768_drop0.2`
   - ❌ `exp1`, `test`

2. **保存配置**：每个实验都会自动保存config.yml，方便复现

3. **记录笔记**：在info.txt中添加实验目的和观察

4. **定期备份**：重要实验的checkpoint要备份

5. **对比分析**：使用可视化工具对比不同配置

## 🤝 与Typilus集成

这套工具的设计思路参考了typilus的实验管理方式：

- ✅ 自动创建规范的目录结构
- ✅ 保存完整的配置和日志
- ✅ 训练后自动评估
- ✅ 生成详细的评估报告
- ✅ 支持批量实验对比

可以直接与typilus的结果进行对比，完成"挑战3"。
