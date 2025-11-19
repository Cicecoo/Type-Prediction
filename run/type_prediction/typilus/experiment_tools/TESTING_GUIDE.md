# Typilus 实验结果分析与测试工具

## 📊 实验结果分析

### 当前实验结果

基于已拉取的学习率实验数据：

| 学习率 | 最佳Epoch | 最佳训练Loss | 最佳验证Loss | Gap | 状态 |
|--------|-----------|--------------|--------------|-----|------|
| 5e-4   | 4 | 3.6480 | 3.7850 | 0.1370 | ⏳ 待测试 |
| 7.5e-4 | 4 | 3.5530 | 3.7560 | 0.2030 | ⏳ 待测试 |
| 1e-3   | 4 | 3.5320 | 3.7180 | 0.1860 | ⏳ 待测试 |
| 1.25e-3| 4 | 3.5090 | 3.7340 | 0.2250 | ⏳ 待测试 |
| 1.5e-3 | 4 | 3.5200 | 3.7210 | 0.2010 | ⏳ 待测试 |

### 初步观察

1. **最佳验证Loss**: `lr=1e-3` (3.7180)
2. **最小Gap**: `lr=5e-4` (0.1370) - 泛化能力最好
3. **训练最快**: `lr=1.25e-3` (训练loss最低)
4. **早停触发**: 所有实验都在第4轮停止，说明patience=3设置合理

### 趋势分析

- **学习率 < 1e-3**: 收敛慢，但泛化好（gap小）
- **学习率 = 1e-3**: 性能最佳，平衡点
- **学习率 > 1e-3**: 训练更快，但可能轻微过拟合

---

## 🧪 测试工具使用指南

### 1. 分析实验结果

查看所有实验的训练结果和对比：

```bash
cd ~/workspace/type_pred/naturalcc/run/type_prediction/typilus/experiment_tools

# 运行分析脚本
python analyze_results.py
```

**输出内容**:
- 所有实验的训练指标表格
- 测试结果表格（如果已测试）
- 最佳配置推荐
- 对比图表（保存为 `../experiments/comparison.png`）
- 分析报告（保存为 `../experiments/analysis_report.md`）

### 2. 批量运行测试

为所有未测试的实验自动运行测试：

```bash
# 查看需要测试的实验（不实际运行）
python batch_test.py --dry_run

# 运行所有未测试的实验
python batch_test.py

# 只测试指定实验
python batch_test.py --exp lr_1e-3
```

**测试流程**:
1. 自动查找有checkpoint但没有测试结果的实验
2. 为每个实验创建测试配置
3. 加载最佳checkpoint运行测试
4. 保存测试结果到 `checkpoints/res.txt`
5. 自动更新训练日志

### 3. 单个实验测试

如果想手动测试某个实验：

```bash
cd ~/workspace/type_pred/naturalcc/run/type_prediction/typilus

# 方法1: 使用原始测试脚本
python type_predict.py -f experiments/lr_1e-3/config

# 方法2: 使用批量测试工具
python experiment_tools/batch_test.py --exp lr_1e-3
```

### 4. 新实验自动测试

训练新实验时，测试会自动运行：

```bash
cd experiment_tools

# 训练会自动在完成后运行测试
python train_enhanced.py -f config_base

# 或使用实验管理工具
python run_experiments.py --exp some_experiment
```

**自动测试特性**:
- ✅ 训练完成后自动加载最佳checkpoint
- ✅ 自动在test集上评估
- ✅ 计算 Acc@1, Acc@5, Loss
- ✅ 保存结果到 `checkpoints/res.txt`
- ✅ 更新训练日志文件

---

## 📝 测试结果格式

### res.txt 文件格式

```
avg_loss: 3.5234
acc1: 45.67
acc5: 68.92
acc1_any: 52.34
acc5_any: 75.21
```

### metrics.json 新增字段

```json
{
  "epochs": [...],
  "train_loss": [...],
  "valid_loss": [...],
  "test_results": {
    "avg_loss": 3.5234,
    "acc1": 45.67,
    "acc5": 68.92,
    "acc1_any": 52.34,
    "acc5_any": 75.21
  }
}
```

### detailed_metrics.txt 新增部分

```
================================================================================
TEST RESULTS
================================================================================
avg_loss                      : 3.5234
acc1                          : 45.67%
acc5                          : 68.92%
acc1_any                      : 52.34%
acc5_any                      : 75.21%
```

---

## 📈 完整工作流程

### 在服务器上运行

```bash
# 1. SSH登录
ssh dlserver6

# 2. 激活环境
conda activate naturalcc

# 3. 进入目录
cd ~/workspace/type_pred/naturalcc/run/type_prediction/typilus/experiment_tools

# 4. 拉取最新代码
git pull

# 5. 为现有实验补充测试
python batch_test.py

# 6. 查看完整分析
python analyze_results.py

# 7. 查看生成的图表和报告
ls ../experiments/
# - comparison.png (对比图)
# - analysis_report.md (分析报告)
```

### 后台运行测试

```bash
# 如果有很多实验需要测试
nohup python batch_test.py > test_all.log 2>&1 &

# 查看进度
tail -f test_all.log

# 测试完成后分析
python analyze_results.py
```

---

## 🎯 关键指标说明

### Acc@1 vs Acc@5

- **Acc@1**: 模型预测的第1名是否正确（严格准确率）
- **Acc@5**: 模型预测的前5名中是否包含正确答案（宽松准确率）

### 含any vs 不含any

- **不含any**: 排除 `$any$` 类型（更严格的评估）
- **含any**: 包含所有类型（完整评估）

在类型预测任务中，`$any$` 是一个特殊标记，通常表示任意类型。

### Gap (Train-Valid Loss差值)

- **Gap < 0.2**: 泛化很好，模型稳定
- **0.2 ≤ Gap < 0.3**: 轻微过拟合，可接受
- **Gap ≥ 0.3**: 明显过拟合，需要调整

---

## 📊 预期测试结果

基于类似研究，Typilus在类型预测任务上的典型表现：

| 指标 | 预期范围 |
|------|----------|
| Acc@1 | 40-50% |
| Acc@5 | 65-75% |
| 测试Loss | 3.5-4.0 |

如果你的结果：
- **显著高于预期**: 🎉 太好了！可能数据集较简单或模型优化很好
- **在预期范围内**: ✅ 正常表现
- **低于预期**: 需要检查数据、模型配置或训练过程

---

## 🔧 故障排查

### 问题1: batch_test.py 找不到实验

```bash
# 检查实验目录结构
ls -R ../experiments/

# 应该看到:
# lr_1e-3/
# ├── checkpoints/
# │   ├── checkpoint_best.pt
# │   └── checkpoint_last.pt
# ├── logs/
# ├── config.yml
# └── ...
```

### 问题2: 测试失败 "找不到模块"

```bash
# 确保在正确的conda环境
conda activate naturalcc

# 检查Python路径
python -c "import ncc; print(ncc.__file__)"
```

### 问题3: CUDA内存不足

编辑测试配置，减小batch size：
```yaml
dataset:
  max_sentences: 16  # 改为 8 或 4
```

### 问题4: 测试结果异常（准确率过低）

可能原因：
1. Checkpoint损坏 - 尝试使用 `checkpoint_last.pt`
2. 数据路径错误 - 检查 `task.data` 配置
3. 字典不匹配 - 重新生成词典

---

## 📚 下一步工作

完成所有测试后：

1. **对比分析**: 
   ```bash
   python analyze_results.py
   ```

2. **选择最佳模型**:
   - 根据验证loss和测试准确率综合判断
   - 考虑泛化能力（gap值）

3. **撰写实验报告**:
   - 使用生成的 `analysis_report.md` 作为基础
   - 添加 `comparison.png` 图表
   - 解释最佳学习率的选择依据

4. **准备对比实验**:
   - 将结果与 Transformer 实验对比
   - 使用 `compare_with_typilus.py` （在transformer目录）

---

## 💡 提示

1. **测试时间**: 每个实验约5-15分钟（取决于数据集大小）
2. **保存结果**: 所有结果会自动保存，无需手动记录
3. **重复测试**: 如果需要重新测试，删除 `res.txt` 再运行
4. **日志完整性**: 测试后训练日志会自动更新，保持记录完整

祝测试顺利！🎉
