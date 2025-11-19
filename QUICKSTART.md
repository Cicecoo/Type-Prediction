# 快速开始 - 5分钟完成第一个实验

## 步骤1: 生成实验配置（30秒）

```bash
cd ~/workspace/type_pred/naturalcc
python generate_experiment_suite.py
```

这会创建约15个实验配置，包括baseline和各种超参数变体。

## 步骤2: 运行Baseline实验（2-3小时）

```bash
# 使用GPU 0运行baseline
python run_batch_experiments.py --group baseline --gpus 0
```

或者使用交互式脚本：

```bash
chmod +x experiment_suite.sh
./experiment_suite.sh
# 选择 "2" 运行baseline
```

## 步骤3: 查看训练进度

在另一个终端监控日志：

```bash
tail -f experiments/transformer_series/exp_baseline/logs/train.log
```

## 步骤4: 实验完成后分析结果

```bash
python analyze_results.py --plot --format both
```

生成的文件：
- `report.html` - 完整HTML报告
- `results_comparison.md` - Markdown表格
- `visualizations/` - 对比图表

## 一键运行所有实验（推荐使用多GPU）

如果有多个GPU可用：

```bash
# 使用4个GPU并行运行所有实验
python run_batch_experiments.py --all --mode parallel --gpus 0 1 2 3
```

## 查看结果示例

### 命令行查看

```bash
# 查看baseline结果
cat experiments/transformer_series/exp_baseline/results/test_results.json
```

输出类似：
```json
{
  "accuracy": 0.7234,
  "top5_accuracy": 0.8912,
  "precision": 0.7156,
  "recall": 0.7089,
  "f1": 0.7122,
  "loss": 0.8234
}
```

### 浏览器查看

```bash
# 生成报告后，在浏览器中打开
firefox report.html  # 或 chrome, edge等
```

## 实验组说明

- **baseline**: 默认配置（必须先运行）
- **model_size**: 测试不同的d_model（256, 512, 1024）
- **layers**: 测试不同层数（4, 6, 8）
- **lr**: 测试不同学习率（1e-4, 5e-4, 1e-3）
- **dropout**: 测试正则化（0.0, 0.1, 0.2）
- **encoder**: 对比Transformer vs LSTM
- **batch_size**: 测试训练效率（16, 32, 64）

## 推荐运行顺序

1. **第一天**: 运行baseline（验证环境正常）
2. **第二天**: 运行model_size和layers（找到最佳模型容量）
3. **第三天**: 运行lr和dropout（调优训练策略）
4. **第四天**: 运行encoder和batch_size（对比架构和效率）
5. **第五天**: 分析所有结果，撰写报告

## 常用命令速查

```bash
# 生成配置
python generate_experiment_suite.py

# 运行单组实验
python run_batch_experiments.py --group <group_name> --gpus 0

# 运行特定实验
python run_batch_experiments.py --names exp_baseline exp_layers_8 --gpus 0

# 查看状态（不运行）
python run_batch_experiments.py --all --dry-run

# 生成分析报告
python analyze_results.py --plot

# 查看实验状态
./experiment_suite.sh  # 选择 "8"
```

## 预期时间

基于典型服务器（Tesla V100 GPU）：

- **单个实验**: 2-4小时（取决于模型大小）
- **一组实验（3个）**: 6-12小时（串行）或 2-4小时（并行）
- **所有实验（15个）**: 30-60小时（串行）或 8-15小时（4GPU并行）

## 遇到问题？

1. **训练失败**: 检查 `experiments/transformer_series/exp_xxx/logs/train.log`
2. **GPU OOM**: 减小batch_size或d_model
3. **数据不存在**: 确认数据路径正确
4. **环境问题**: 重新运行 `conda activate naturalcc`

## 下一步

完成baseline后，建议：

1. 分析baseline结果，确定性能瓶颈
2. 优先运行与瓶颈相关的实验组
3. 根据初步结果调整后续实验的超参数范围
4. 记录实验日志和观察结果

---

**提示**: 所有脚本都支持 `--help` 查看详细参数说明！

```bash
python generate_experiment_suite.py --help
python run_batch_experiments.py --help
python analyze_results.py --help
```
