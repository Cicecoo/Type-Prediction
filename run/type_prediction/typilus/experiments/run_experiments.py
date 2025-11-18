#!/usr/bin/env python3
"""
Typilus 超参数调优实验管理脚本
用于启动、监控和比较不同的实验配置
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 实验根目录
EXP_ROOT = Path(__file__).parent
PROJECT_ROOT = EXP_ROOT.parent.parent.parent.parent

# 所有实验配置
EXPERIMENTS = {
    "baseline": {
        "config": "run/type_prediction/typilus/config/typilus.yml",
        "description": "基线配置 (lr=4e-4, hidden=64, layers=2, dropout=0.1)"
    },
    "exp_lr_2e4": {
        "config": "run/type_prediction/typilus/experiments/exp_lr_2e4/config.yml",
        "description": "降低学习率至2e-4"
    },
    "exp_lr_1e4": {
        "config": "run/type_prediction/typilus/experiments/exp_lr_1e4/config.yml",
        "description": "降低学习率至1e-4"
    },
    "exp_dropout_02": {
        "config": "run/type_prediction/typilus/experiments/exp_dropout_02/config.yml",
        "description": "增加dropout至0.2"
    },
    "exp_hidden_128": {
        "config": "run/type_prediction/typilus/experiments/exp_hidden_128/config.yml",
        "description": "增大隐藏层至128"
    },
    "exp_layers_4": {
        "config": "run/type_prediction/typilus/experiments/exp_layers_4/config.yml",
        "description": "增加层数至4"
    }
}


def list_experiments():
    """列出所有可用的实验"""
    print("\n" + "="*80)
    print("可用的实验配置:")
    print("="*80)
    for exp_name, exp_info in EXPERIMENTS.items():
        print(f"\n实验名称: {exp_name}")
        print(f"描述: {exp_info['description']}")
        print(f"配置文件: {exp_info['config']}")
        
        # 检查是否有结果文件
        exp_dir = EXP_ROOT / exp_name
        if exp_dir.exists():
            info_file = exp_dir / "experiment_info.json"
            result_file = exp_dir / "results.json"
            
            if info_file.exists():
                with open(info_file) as f:
                    info = json.load(f)
                print(f"状态: {info.get('status', 'unknown')}")
            
            if result_file.exists():
                with open(result_file) as f:
                    results = json.load(f)
                print(f"结果: acc1={results.get('acc1', 'N/A'):.2f}%, "
                      f"acc5={results.get('acc5', 'N/A'):.2f}%")
    print("\n" + "="*80 + "\n")


def train_experiment(exp_name):
    """启动训练实验"""
    if exp_name not in EXPERIMENTS:
        print(f"错误: 实验 '{exp_name}' 不存在")
        print("使用 'python run_experiments.py list' 查看所有实验")
        return
    
    exp_info = EXPERIMENTS[exp_name]
    config_path = exp_info['config']
    
    print(f"\n启动实验: {exp_name}")
    print(f"描述: {exp_info['description']}")
    print(f"配置: {config_path}")
    print("\n提示: 请在服务器上使用以下命令:")
    print("="*80)
    print(f"# 1. 进入项目根目录")
    print(f"cd /path/to/Type-Prediction")
    print(f"\n# 2. 激活conda环境")
    print(f"conda activate naturalcc")
    print(f"\n# 3. 设置环境变量")
    print(f"export NCC=/mnt/data1/zhaojunzhang/typilus-data")
    print(f"\n# 4. 创建screen会话")
    print(f"screen -L -Logfile ./screen/log_{exp_name}.txt -S {exp_name}")
    print(f"\n# 5. 运行训练")
    # train.py用os.path.dirname(__file__)拼接，所以路径是相对于train.py所在目录
    # train.py在 run/type_prediction/typilus/
    # 配置在 run/type_prediction/typilus/experiments/exp_lr_2e4/config.yml
    # 所以-f参数应该是: experiments/exp_lr_2e4/config (不含.yml)
    config_relative = config_path.replace('run/type_prediction/typilus/', '').replace('.yml', '')
    print(f"python run/type_prediction/typilus/train.py -f {config_relative}")
    print("="*80 + "\n")


def evaluate_experiment(exp_name):
    """运行推理评估"""
    if exp_name not in EXPERIMENTS:
        print(f"错误: 实验 '{exp_name}' 不存在")
        return
    
    exp_info = EXPERIMENTS[exp_name]
    config_path = exp_info['config']
    
    print(f"\n评估实验: {exp_name}")
    print(f"配置: {config_path}")
    print("\n提示: 请在服务器上使用以下命令:")
    print("="*80)
    print(f"# 1. 进入项目根目录")
    print(f"cd /path/to/Type-Prediction")
    print(f"\n# 2. 激活环境并运行推理")
    print(f"conda activate naturalcc")
    print(f"export NCC=/mnt/data1/zhaojunzhang/typilus-data")
    config_relative = config_path.replace('run/type_prediction/typilus/', '').replace('.yml', '')
    print(f"python run/type_prediction/typilus/type_predict.py -f {config_relative}")
    print("="*80 + "\n")


def compare_results():
    """比较所有实验的结果"""
    print("\n" + "="*100)
    print("实验结果对比")
    print("="*100)
    print(f"{'实验名称':<20} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Loss':<10} {'状态':<10}")
    print("-"*100)
    
    results = []
    for exp_name in EXPERIMENTS.keys():
        exp_dir = EXP_ROOT / exp_name
        result_file = exp_dir / "results.json"
        
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)
            results.append({
                'name': exp_name,
                'acc1': data.get('acc1', 0),
                'acc5': data.get('acc5', 0),
                'loss': data.get('avg_loss', 0),
                'status': 'completed'
            })
        else:
            results.append({
                'name': exp_name,
                'acc1': 0,
                'acc5': 0,
                'loss': 0,
                'status': 'pending'
            })
    
    # 按Top-1准确率排序
    results.sort(key=lambda x: x['acc1'], reverse=True)
    
    for r in results:
        print(f"{r['name']:<20} {r['acc1']:>10.2f}% {r['acc5']:>10.2f}% "
              f"{r['loss']:>10.4f} {r['status']:<10}")
    
    print("="*100 + "\n")
    
    # 找出最佳实验
    best = max(results, key=lambda x: x['acc1'])
    if best['acc1'] > 0:
        print(f"🏆 最佳实验: {best['name']}")
        print(f"   Top-1 Accuracy: {best['acc1']:.2f}%")
        print(f"   Top-5 Accuracy: {best['acc5']:.2f}%")
        print(f"   Loss: {best['loss']:.4f}\n")


def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  python run_experiments.py list              - 列出所有实验")
        print("  python run_experiments.py train <exp_name>  - 训练指定实验")
        print("  python run_experiments.py eval <exp_name>   - 评估指定实验")
        print("  python run_experiments.py compare           - 比较所有实验结果")
        print("\n示例:")
        print("  python run_experiments.py train exp_lr_2e4")
        return
    
    command = sys.argv[1]
    
    if command == "list":
        list_experiments()
    elif command == "train":
        if len(sys.argv) < 3:
            print("错误: 请指定实验名称")
            print("示例: python run_experiments.py train exp_lr_2e4")
        else:
            train_experiment(sys.argv[2])
    elif command == "eval":
        if len(sys.argv) < 3:
            print("错误: 请指定实验名称")
            print("示例: python run_experiments.py eval exp_lr_2e4")
        else:
            evaluate_experiment(sys.argv[2])
    elif command == "compare":
        compare_results()
    else:
        print(f"未知命令: {command}")
        print("使用 'python run_experiments.py' 查看帮助")


if __name__ == "__main__":
    main()
