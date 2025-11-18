# 类型预测调参实验

## 核心工具

- `run_experiments.py` - 实验管理，生成训练命令
- `monitor.py` - 实时监控训练进度
- `log_parser.py` - 解析训练日志
- `visualize_results.py` - 生成可视化图表（可选，需安装matplotlib等）

## 实验配置

| 实验名称 | 修改参数 | 目的 |
|---------|---------|------|
| exp_lr_2e4 | lr: 2e-4 | 解决loss波动 ⭐ |
| exp_lr_1e4 | lr: 1e-4 | 更保守的学习率 |
| exp_dropout_02 | dropout: 0.2 | 增强正则化 |
| exp_hidden_128 | hidden: 128 | 增大模型容量 |
| exp_layers_4 | layers: 4 | 增加网络深度 |

基线: Top-1 Acc=22.54%, Top-5 Acc=54.89%

## 使用方法

### 1. 查看训练命令
```bash
python run/type_prediction/typilus/experiments/run_experiments.py train exp_lr_2e4
```

### 2. 启动训练（按输出的命令）
```bash
# 必须在项目根目录执行！
cd /path/to/Type-Prediction

conda activate naturalcc
export NCC=/path/to/typilus-data

screen -L -Logfile ./screen/log_exp_lr_2e4.txt -S exp_lr_2e4
python run/type_prediction/typilus/train.py -f experiments/exp_lr_2e4/config
```

### 3. 监控进度
```bash
# 退出screen但不停止训练: Ctrl+A, D
# 新终端监控
python run/type_prediction/typilus/experiments/monitor.py exp_lr_2e4
```

### 4. 解析结果
```bash
python run/type_prediction/typilus/experiments/log_parser.py screen/log_exp_lr_2e4.txt
```

## 注意事项

1. **工作目录**: 所有命令必须在项目根目录 `/path/to/Type-Prediction` 执行
2. **路径格式**: `-f` 参数是相对于 `train.py` 的路径，不含 `.yml` 后缀
   - 正确: `-f experiments/exp_lr_2e4/config`
   - 错误: `-f run/type_prediction/typilus/experiments/exp_lr_2e4/config`
3. **环境变量**: 确保设置 `NCC` 指向typilus数据目录
