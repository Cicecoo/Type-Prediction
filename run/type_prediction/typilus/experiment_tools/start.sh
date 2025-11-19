#!/bin/bash
# Typilus 实验工具 - Linux快速启动脚本

echo ""
echo "========================================"
echo "  Typilus 参数调优实验工具"
echo "========================================"
echo ""

echo "选择操作:"
echo "1. 运行所有实验（自动化）"
echo "2. 分析已有结果"
echo "3. 查看帮助"
echo "0. 退出"
echo ""

read -p "请输入 [0-3]: " choice

case $choice in
    1)
        echo ""
        echo "启动自动化实验..."
        python run_experiments.py
        ;;
    2)
        echo ""
        echo "分析结果..."
        python run_experiments.py --analyze
        ;;
    3)
        echo ""
        cat README.md
        ;;
    0)
        echo "退出"
        ;;
    *)
        echo "无效选项"
        ;;
esac
