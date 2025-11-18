#!/bin/bash
# 自动运行所有类型预测调参实验
# 使用方法: bash run_all.sh

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Typilus 超参数调优实验批量启动脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查当前目录
if [ ! -f "run/type_prediction/typilus/train.py" ]; then
    echo -e "${RED}错误: 请在项目根目录执行此脚本${NC}"
    echo "当前目录: $(pwd)"
    echo "应该在: /path/to/Type-Prediction"
    exit 1
fi

# 检查conda环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${YELLOW}警告: conda环境未激活${NC}"
    echo "请先运行: conda activate naturalcc"
    exit 1
fi

# 检查NCC环境变量
if [ -z "$NCC" ]; then
    echo -e "${YELLOW}警告: NCC环境变量未设置${NC}"
    echo "请先运行: export NCC=/path/to/typilus-data"
    exit 1
fi

echo -e "${GREEN}✓ 环境检查通过${NC}"
echo "  Conda环境: $CONDA_DEFAULT_ENV"
echo "  NCC路径: $NCC"
echo ""

# 创建必要的目录
mkdir -p screen
mkdir -p results/{checkpoints,logs,parsed}

echo -e "${GREEN}✓ 目录创建完成${NC}"
echo ""

# 定义实验列表（按优先级排序）
# 格式: "实验名称:优先级:预计时长"
experiments=(
    "exp_lr_2e4:高:30min"
    "exp_dropout_02:高:30min"
    "exp_lr_1e4:中:30min"
    "exp_batch_8:中:40min"
    "exp_hidden_128:低:45min"
    "exp_layers_4:低:50min"
)

# 显示实验计划
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}实验计划（共${#experiments[@]}个）${NC}"
echo -e "${GREEN}========================================${NC}"
for exp in "${experiments[@]}"; do
    IFS=':' read -r name priority duration <<< "$exp"
    printf "%-20s 优先级:%-4s 预计:%s\n" "$name" "$priority" "$duration"
done
echo ""

# 检查是否需要交互确认
if [ "$1" != "-y" ] && [ "$1" != "--yes" ]; then
    read -p "是否启动所有实验？[y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "已取消"
        exit 0
    fi
fi

# 启动所有实验
echo -e "${GREEN}开始启动实验（后台运行）...${NC}"
echo ""

for exp in "${experiments[@]}"; do
    IFS=':' read -r name priority duration <<< "$exp"
    
    echo -e "${YELLOW}[启动] $name${NC}"
    
    # 检查screen会话是否已存在
    if screen -list | grep -q "\.${name}\s"; then
        echo -e "${RED}  ✗ Screen会话已存在，跳过${NC}"
        continue
    fi
    
    # 创建实验结果目录
    mkdir -p "results/checkpoints/${name}"
    mkdir -p "results/logs/${name}"
    
    # 启动screen会话
    screen -dmS "$name" -L -Logfile "./screen/log_${name}.txt" bash -c "
        export NCC=$NCC
        echo '开始训练: $name'
        echo '时间: \$(date)'
        echo '环境: $CONDA_DEFAULT_ENV'
        echo '========================================='
        
        python run/type_prediction/typilus/train.py -f experiments/${name}/config
        
        exit_code=\$?
        echo ''
        echo '========================================='
        echo '训练完成: $name'
        echo '退出码: '\$exit_code
        echo '时间: \$(date)'
        
        # 保存退出码
        echo \$exit_code > results/logs/${name}/exit_code.txt
        
        # 如果训练成功，复制checkpoint
        if [ \$exit_code -eq 0 ]; then
            echo '复制checkpoint到results目录...'
            if [ -d ~/naturalcc/typilus/checkpoints/${name} ]; then
                cp -r ~/naturalcc/typilus/checkpoints/${name}/* results/checkpoints/${name}/ 2>/dev/null || true
            fi
        fi
        
        exec bash
    "
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✓ Screen会话已创建: $name${NC}"
    else
        echo -e "${RED}  ✗ 启动失败${NC}"
    fi
    
    # 间隔2秒，避免同时启动占满GPU
    sleep 2
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}所有实验已启动完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 显示运行中的实验
echo "运行中的Screen会话:"
screen -ls | grep -E "exp_" || echo "  无"
echo ""

# 使用提示
echo -e "${YELLOW}后续操作:${NC}"
echo "  实时监控所有:    ./watch_all.sh"
echo "  查看所有会话:    screen -ls"
echo "  连接某个实验:    screen -r exp_lr_2e4"
echo "  退出但不停止:    Ctrl+A, D"
echo "  监控单个实验:    python run/type_prediction/typilus/experiments/monitor.py exp_lr_2e4"
echo "  查看日志文件:    tail -f screen/log_exp_lr_2e4.txt"
echo ""
echo "  停止某个实验:    screen -X -S exp_lr_2e4 quit"
echo "  停止所有实验:    pkill -f 'SCREEN.*exp_'"
echo ""

# 创建监控脚本
cat > watch_all.sh << 'EOF'
#!/bin/bash
# 监控所有实验进度

while true; do
    clear
    echo "=========================================="
    echo "实验监控 - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # 显示screen会话
    echo "【运行中的实验】"
    screen -ls | grep exp_ || echo "  无运行中的实验"
    echo ""
    
    # 显示各实验的最新日志
    for log in screen/log_exp_*.txt; do
        if [ -f "$log" ]; then
            exp_name=$(basename "$log" .txt | sed 's/log_//')
            echo "【${exp_name}】"
            echo "  日志大小: $(ls -lh "$log" | awk '{print $5}')"
            
            # 提取最新的loss/accuracy信息
            tail -n 50 "$log" | grep -E "loss|acc|epoch" | tail -n 2 | sed 's/^/  /'
            
            # 检查退出码
            if [ -f "results/logs/${exp_name}/exit_code.txt" ]; then
                exit_code=$(cat "results/logs/${exp_name}/exit_code.txt")
                if [ "$exit_code" -eq 0 ]; then
                    echo "  状态: ✓ 完成"
                else
                    echo "  状态: ✗ 失败 (退出码: $exit_code)"
                fi
            fi
            echo ""
        fi
    done
    
    echo "按 Ctrl+C 退出监控"
    sleep 10
done
EOF

chmod +x watch_all.sh

echo -e "${GREEN}✓ 已创建监控脚本: watch_all.sh${NC}"
echo "  运行 ./watch_all.sh 可以监控所有实验进度"
echo ""

echo -e "${GREEN}实验已在后台运行，可以安全退出终端${NC}"
