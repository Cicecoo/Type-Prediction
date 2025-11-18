# 类型预测调参实验

## 🚀 快速开始（推荐）

### 一键启动所有实验
```bash
cd /path/to/Type-Prediction
conda activate naturalcc
export NCC=/path/to/typilus-data

bash run/type_prediction/typilus/experiments/run_all.sh
```

### 启动单个实验
```bash
bash run/type_prediction/typilus/experiments/start_single.sh exp_lr_2e4
```

## 核心工具

- `run_all.sh` - **一键启动所有实验**（自动创建结果目录）
- `start_single.sh` - 启动单个实验
- `run_experiments.py` - 查看实验信息，生成训练命令
- `monitor.py` - 实时监控训练进度
- `log_parser.py` - 解析训练日志
- `visualize_results.py` - 生成可视化图表（可选）

## 实验配置

| 实验名称 | 修改参数 | 目的 |
|---------|---------|------|
| exp_lr_2e4 | lr: 2e-4 | 解决loss波动 ⭐ |
| exp_lr_1e4 | lr: 1e-4 | 更保守的学习率 |
| exp_dropout_02 | dropout: 0.2 | 增强正则化 |
| exp_hidden_128 | hidden: 128 | 增大模型容量 |
| exp_layers_4 | layers: 4 | 增加网络深度 |

基线: Top-1 Acc=22.54%, Top-5 Acc=54.89%

## 详细使用方法

### 方式1: 自动化脚本（推荐）

**批量启动所有实验**:
```bash
bash run/type_prediction/typilus/experiments/run_all.sh
# 会自动创建 results/ 目录保存所有结果
# 自动生成 watch_all.sh 监控脚本
```

**启动单个实验**:
```bash
bash run/type_prediction/typilus/experiments/start_single.sh exp_lr_2e4
```

**监控所有实验**:
```bash
./watch_all.sh  # 由 run_all.sh 自动生成
```

### 方式2: 手动操作

**1. 查看训练命令**:
```bash
python run/type_prediction/typilus/experiments/run_experiments.py train exp_lr_2e4
```

**2. 启动训练**:
```bash
cd /path/to/Type-Prediction
conda activate naturalcc
export NCC=/path/to/typilus-data

screen -L -Logfile ./screen/log_exp_lr_2e4.txt -S exp_lr_2e4
python run/type_prediction/typilus/train.py -f experiments/exp_lr_2e4/config
```

**3. 监控进度**:
```bash
# 退出screen: Ctrl+A, D
python run/type_prediction/typilus/experiments/monitor.py exp_lr_2e4
```

**4. 解析结果**:
```bash
python run/type_prediction/typilus/experiments/log_parser.py screen/log_exp_lr_2e4.txt
```

## 结果目录结构

使用 `run_all.sh` 或 `start_single.sh` 后，会自动创建：

```
results/
├── checkpoints/          # 训练好的模型
│   ├── exp_lr_2e4/
│   ├── exp_lr_1e4/
│   └── ...
├── logs/                 # 元数据和状态
│   ├── exp_lr_2e4/
│   │   ├── exit_code.txt
│   │   └── finish_time.txt
│   └── ...
└── parsed/              # 解析后的结果（手动运行log_parser.py生成）

screen/                  # Screen日志
├── log_exp_lr_2e4.txt
├── log_exp_lr_1e4.txt
└── ...
```

## Screen常用命令

```bash
screen -ls                      # 查看所有会话
screen -r exp_lr_2e4           # 连接到某个实验
# 在screen内: Ctrl+A, D        # 退出但不停止训练

screen -X -S exp_lr_2e4 quit   # 停止某个实验
```

## 注意事项

1. **工作目录**: 所有命令必须在项目根目录执行
2. **路径格式**: `-f` 参数相对于 `train.py`，不含 `.yml` 后缀
3. **环境变量**: 确保设置 `NCC` 指向typilus数据目录
4. **GPU资源**: 批量启动会间隔2秒，避免同时占满GPU
