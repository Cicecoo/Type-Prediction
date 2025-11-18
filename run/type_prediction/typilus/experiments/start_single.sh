#!/bin/bash
# 启动单个实验
# 使用方法: bash start_single.sh exp_lr_2e4

if [ $# -eq 0 ]; then
    echo "使用方法: bash start_single.sh <实验名称>"
    echo ""
    echo "可用的实验:"
    echo "  exp_lr_2e4      - 学习率 2e-4 (推荐首选)"
    echo "  exp_lr_1e4      - 学习率 1e-4"
    echo "  exp_dropout_02  - Dropout 0.2"
    echo "  exp_hidden_128  - 隐藏层 128"
    echo "  exp_layers_4    - 层数 4"
    echo "  exp_batch_8     - Batch size 8"
    exit 1
fi

EXP_NAME=$1

# 检查实验配置是否存在
if [ ! -f "run/type_prediction/typilus/experiments/${EXP_NAME}/config.yml" ]; then
    echo "错误: 实验配置不存在: experiments/${EXP_NAME}/config.yml"
    exit 1
fi

# 检查环境
if [ ! -f "run/type_prediction/typilus/train.py" ]; then
    echo "错误: 请在项目根目录执行此脚本"
    exit 1
fi

if [ -z "$NCC" ]; then
    echo "警告: NCC环境变量未设置"
    echo "请先运行: export NCC=/path/to/typilus-data"
    exit 1
fi

# 创建必要的目录
mkdir -p screen
mkdir -p results/{checkpoints,logs,parsed}/${EXP_NAME}

# 检查screen会话是否已存在
if screen -list | grep -q "\.${EXP_NAME}\s"; then
    echo "错误: Screen会话已存在"
    echo "可以使用以下命令:"
    echo "  连接: screen -r ${EXP_NAME}"
    echo "  停止: screen -X -S ${EXP_NAME} quit"
    exit 1
fi

echo "=========================================="
echo "启动实验: $EXP_NAME"
echo "=========================================="
echo "配置文件: experiments/${EXP_NAME}/config.yml"
echo "日志文件: screen/log_${EXP_NAME}.txt"
echo "结果目录: results/checkpoints/${EXP_NAME}/"
echo ""

# 启动screen会话
screen -dmS "$EXP_NAME" -L -Logfile "./screen/log_${EXP_NAME}.txt" bash -c "
    # 重新激活conda环境
    source \$(conda info --base)/etc/profile.d/conda.sh
    conda activate $CONDA_DEFAULT_ENV
    
    # 设置环境变量
    export NCC=$NCC
    
    echo '========================================='
    echo '实验: $EXP_NAME'
    echo '时间: \$(date)'
    echo 'Conda环境: '\$CONDA_DEFAULT_ENV
    echo 'NCC路径: '\$NCC
    echo '配置: experiments/${EXP_NAME}/config'
    echo '========================================='
    echo ''
    
    python run/type_prediction/typilus/train.py -f experiments/${EXP_NAME}/config
    
    exit_code=\$?
    echo ''
    echo '========================================='
    echo '训练完成'
    echo '退出码: '\$exit_code
    echo '结束时间: \$(date)'
    echo '========================================='
    
    # 保存退出码和完成时间
    echo \$exit_code > results/logs/${EXP_NAME}/exit_code.txt
    date > results/logs/${EXP_NAME}/finish_time.txt
    
    # 如果成功，复制checkpoint
    if [ \$exit_code -eq 0 ]; then
        echo '复制checkpoint...'
        if [ -d ~/naturalcc/typilus/checkpoints/${EXP_NAME} ]; then
            cp -r ~/naturalcc/typilus/checkpoints/${EXP_NAME}/* results/checkpoints/${EXP_NAME}/ 2>/dev/null || true
            echo '✓ Checkpoint已保存到 results/checkpoints/${EXP_NAME}/'
        fi
    fi
    
    exec bash
"

if [ $? -eq 0 ]; then
    echo "✓ 实验已启动！"
    echo ""
    echo "后续操作:"
    echo "  连接会话: screen -r ${EXP_NAME}"
    echo "  监控进度: python run/type_prediction/typilus/experiments/monitor.py ${EXP_NAME}"
    echo "  查看日志: tail -f screen/log_${EXP_NAME}.txt"
    echo "  停止实验: screen -X -S ${EXP_NAME} quit"
else
    echo "✗ 启动失败"
    exit 1
fi
