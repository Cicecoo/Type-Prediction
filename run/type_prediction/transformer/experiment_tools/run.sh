#!/bin/bash
# Transformer 实验快速启动脚本

echo "=================================================="
echo "  Transformer Type Prediction 实验启动"
echo "=================================================="
echo ""

# 检查conda环境
if [ "$CONDA_DEFAULT_ENV" != "naturalcc" ]; then
    echo "警告: 当前不在 naturalcc 环境中"
    echo "正在激活 naturalcc 环境..."
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate naturalcc
fi

# 检查当前目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "当前目录: $SCRIPT_DIR"
echo ""

# 显示可用实验
echo "可用实验:"
echo "------------------------------------------------"
python run_experiments.py --list
echo ""

# 询问用户
echo "请选择操作:"
echo "  1) 运行单个实验"
echo "  2) 运行所有实验"
echo "  3) 分析已有结果"
echo "  4) 退出"
echo ""
read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        echo ""
        read -p "请输入实验名称 (例如: transformer_baseline): " exp_name
        echo ""
        echo "开始运行实验: $exp_name"
        echo "日志将保存到各实验的 checkpoint 目录"
        echo ""
        python run_experiments.py --exp "$exp_name"
        ;;
    2)
        echo ""
        echo "开始运行所有实验..."
        echo "这可能需要很长时间（每个实验 3-8 小时）"
        read -p "是否继续? (y/n): " confirm
        if [ "$confirm" == "y" ] || [ "$confirm" == "Y" ]; then
            echo ""
            nohup python run_experiments.py > run_all_transformer.log 2>&1 &
            echo "所有实验已在后台启动"
            echo "查看进度: tail -f run_all_transformer.log"
            echo "进程ID: $!"
        else
            echo "已取消"
        fi
        ;;
    3)
        echo ""
        echo "分析实验结果..."
        python run_experiments.py --analyze
        ;;
    4)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "完成!"
