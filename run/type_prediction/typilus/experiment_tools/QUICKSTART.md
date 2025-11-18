# Typilus 实验工具 - 快速参考

## 📦 文件说明

```
experiment_tools/          ← 所有工具都在这里
├── README.md             详细使用说明
├── run_experiments.py    主脚本（实验+分析一体）
├── train_enhanced.py     增强版训练（带日志可视化）
└── start.ps1             Windows快速启动
```

## 🚀 快速开始

### Linux服务器（推荐）
```bash
cd experiment_tools
chmod +x start.sh
./start.sh    # 交互式菜单
```

**或直接运行：**
```bash
# screen中后台运行（推荐）
screen -S typilus_exp
python run_experiments.py
# Ctrl+A D (detach)

# 后续查看
screen -r typilus_exp
```

### Windows本地
```powershell
cd experiment_tools
.\start.ps1    # 交互式菜单
```

### 命令行直接运行
```bash
cd experiment_tools

# 运行全部实验（6个预设）
python run_experiments.py

# 仅分析结果
python run_experiments.py --analyze

# 单独训练
python train_enhanced.py --yaml_file ../config/typilus
```

## 📊 预设实验

- **baseline** - 基线（默认配置）
- **exp_lr_1e-3** - 学习率1e-3
- **exp_lr_1e-4** - 学习率1e-4
- **exp_batch_64** - 批量64
- **exp_hidden_128** - 隐藏层128
- **exp_best** - 推荐配置

修改: 编辑 `run_experiments.py` 的 `EXPERIMENTS` 列表

## 📁 输出

```
~/naturalcc/typilus/experiments/
├── baseline/
│   ├── config.yml
│   ├── checkpoints/
│   └── logs/
│       ├── metrics.json
│       └── plots/training.png  # 4合1图
├── comparison.png              # 对比图
└── report.md                   # 分析报告
```

## ⚙️ 常见配置

**指定GPU:**
```python
# 在实验配置中添加
"params": {
    "distributed_training": {"device_id": 1}
}
```

**减少显存:**
```python
"params": {
    "dataset": {"max_sentences": 16},      # 减小批量
    "model": {"encoder_hidden_size": 32}   # 减小模型
}
```

## 🔄 精简对比

**之前:** 10个文件分散
**现在:** 4个文件集中

✅ 功能完整
✅ 更易使用
✅ 便于维护

---

**立即开始:**
```bash
cd experiment_tools && python run_experiments.py
```
