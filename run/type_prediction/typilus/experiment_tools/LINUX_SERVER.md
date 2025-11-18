# Typilus 实验工具 - Linux服务器使用指南

## 🖥️ 受限远程Linux服务器快速开始

### 一、准备工作

1. **上传文件到服务器**
```bash
# 使用scp上传整个文件夹
scp -r experiment_tools/ user@server:/path/to/typilus/

# 或使用rsync
rsync -avz experiment_tools/ user@server:/path/to/typilus/experiment_tools/
```

2. **检查环境**
```bash
# 登录服务器
ssh user@server

# 进入目录
cd /path/to/typilus/experiment_tools

# 检查Python
python --version  # 需要3.8+

# 检查GPU
nvidia-smi

# 安装依赖
pip install matplotlib pyyaml
```

### 二、运行实验（推荐方式）

#### 🌟 方式1: 使用screen（最推荐）

```bash
# 1. 创建screen会话
screen -S typilus_exp

# 2. 运行实验
cd experiment_tools
python run_experiments.py

# 3. Detach（保持运行）
# 按 Ctrl+A，然后按 D

# 4. 断开SSH连接也没关系，实验继续运行

# 5. 后续重新连接
ssh user@server
screen -r typilus_exp

# 6. 查看所有screen会话
screen -ls
```

#### 方式2: 使用nohup

```bash
cd experiment_tools
nohup python run_experiments.py > training.log 2>&1 &

# 查看进程
ps aux | grep run_experiments

# 查看日志
tail -f training.log
```

#### 方式3: 使用tmux

```bash
# 创建会话
tmux new -s typilus_exp

# 运行实验
cd experiment_tools
python run_experiments.py

# Detach: Ctrl+B，然后按 D

# 重新连接
tmux attach -t typilus_exp
```

### 三、监控实验

#### 监控GPU
```bash
# 实时监控
watch -n 1 nvidia-smi

# 或使用gpustat（更友好）
pip install gpustat
watch -n 1 gpustat -cpu
```

#### 监控日志
```bash
# 查看实验日志
tail -f ~/naturalcc/typilus/experiments/baseline/training.log

# 查看最新10行
tail -n 10 ~/naturalcc/typilus/experiments/baseline/logs/metrics.json
```

#### 查看进度
```bash
# 查看实验目录
ls -lh ~/naturalcc/typilus/experiments/

# 查看训练指标
cat ~/naturalcc/typilus/experiments/baseline/info.txt
```

### 四、常见场景

#### 场景1: 只运行特定实验

编辑 `run_experiments.py`，注释掉不需要的实验：
```python
EXPERIMENTS = [
    {
        "name": "baseline",
        "desc": "基线实验",
        "params": {}
    },
    # {  # 注释掉不运行
    #     "name": "exp_lr_1e-3",
    #     ...
    # },
]
```

#### 场景2: 指定GPU

**方法1: 环境变量**
```bash
export CUDA_VISIBLE_DEVICES=0
python run_experiments.py
```

**方法2: 修改配置**
在实验配置中添加：
```python
"params": {
    "distributed_training": {"device_id": 0}
}
```

#### 场景3: 减少显存占用

```python
"params": {
    "dataset": {"max_sentences": 16},      # 减小batch
    "model": {
        "encoder_hidden_size": 32,         # 减小模型
        "encoder_layers": 2
    }
}
```

#### 场景4: 中断后继续

实验会自动保存checkpoint，直接重新运行即可：
```bash
python run_experiments.py
# 会自动加载最新检查点继续
```

### 五、传输结果

#### 下载实验结果到本地

```bash
# 下载所有结果
scp -r user@server:~/naturalcc/typilus/experiments/ ./local_results/

# 只下载图表和报告
scp user@server:~/naturalcc/typilus/experiments/comparison.png ./
scp user@server:~/naturalcc/typilus/experiments/report.md ./

# 下载单个实验
scp -r user@server:~/naturalcc/typilus/experiments/baseline/ ./
```

### 六、故障排查

#### 问题1: screen会话丢失
```bash
# 查找所有screen会话
screen -ls

# 如果显示Detached，重新连接
screen -r typilus_exp

# 如果显示Attached（被占用），强制连接
screen -d -r typilus_exp
```

#### 问题2: 显存不足
```bash
# 查看显存使用
nvidia-smi

# 杀死其他进程（谨慎）
kill -9 PID

# 或减小batch size（推荐）
# 编辑实验配置，设置 max_sentences: 8
```

#### 问题3: 权限问题
```bash
# 给脚本执行权限
chmod +x start.sh

# 检查文件权限
ls -la
```

#### 问题4: Python包缺失
```bash
# 安装到用户目录（无需root）
pip install --user matplotlib pyyaml

# 或使用conda
conda install matplotlib pyyaml
```

### 七、完整示例流程

```bash
# 1. 登录服务器
ssh user@gpu-server

# 2. 进入目录
cd /path/to/typilus/experiment_tools

# 3. 检查环境
nvidia-smi
python --version

# 4. 创建screen会话
screen -S typilus_exp

# 5. 可选：指定GPU
export CUDA_VISIBLE_DEVICES=0

# 6. 运行实验
python run_experiments.py

# 7. Detach（保持运行）
# Ctrl+A D

# 8. 断开SSH
exit

# ===== 几小时或几天后 =====

# 9. 重新登录
ssh user@gpu-server

# 10. 重新连接screen
screen -r typilus_exp

# 11. 查看结果
python run_experiments.py --analyze

# 12. 下载结果（在本地机器执行）
scp -r user@gpu-server:~/naturalcc/typilus/experiments/ ./results/
```

### 八、提示和技巧

✅ **使用screen或tmux** - 最可靠的方式
✅ **定期备份** - 重要实验及时下载到本地
✅ **监控资源** - 使用watch、htop等工具
✅ **记录日志** - 所有输出都会自动保存
✅ **分批实验** - 可以先运行1-2个测试

❌ 不要直接在SSH会话中运行（断开就中止）
❌ 不要忘记保存重要结果
❌ 不要在生产服务器上占满GPU

### 九、快速命令参考

```bash
# 启动实验
screen -S exp && cd experiment_tools && python run_experiments.py

# 查看GPU
watch -n 1 nvidia-smi

# 查看日志
tail -f ~/naturalcc/typilus/experiments/*/training.log

# 分析结果
python run_experiments.py --analyze

# 下载结果（本地执行）
scp -r user@server:~/naturalcc/typilus/experiments/ ./
```

---

**立即开始:**
```bash
screen -S typilus_exp
cd experiment_tools
python run_experiments.py
```

祝实验顺利！🚀
