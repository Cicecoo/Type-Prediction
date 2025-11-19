#!/usr/bin/env python3
"""
训练日志可视化工具

功能：
1. 从训练日志中提取loss、accuracy等指标
2. 绘制训练曲线
3. 对比多个实验的结果
"""

import os
import re
import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 无GUI环境
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_train_log(log_file):
    """从训练日志中提取指标"""
    metrics = defaultdict(list)
    
    # 正则表达式模式
    patterns = {
        'epoch': r'epoch (\d+)',
        'loss': r'loss[:\s]+([0-9.]+)',
        'accuracy': r'accuracy[:\s]+([0-9.]+)',
        'lr': r'lr[:\s]+([0-9.e\-]+)',
        'update': r'update (\d+)',
    }
    
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            
            # 提取各种指标
            for key, pattern in patterns.items():
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1))
                        metrics[key].append(value)
                    except ValueError:
                        continue
    
    return metrics


def plot_training_curves(metrics, output_dir, exp_name=''):
    """绘制训练曲线"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制loss曲线
    if 'loss' in metrics and metrics['loss']:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['loss'], linewidth=2)
        plt.xlabel('Update Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'Training Loss{" - " + exp_name if exp_name else ""}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        loss_file = os.path.join(output_dir, 'loss_curve.png')
        plt.savefig(loss_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Loss curve saved to: {loss_file}")
    
    # 绘制accuracy曲线
    if 'accuracy' in metrics and metrics['accuracy']:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['accuracy'], linewidth=2, color='green')
        plt.xlabel('Update Steps', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'Training Accuracy{" - " + exp_name if exp_name else ""}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        acc_file = os.path.join(output_dir, 'accuracy_curve.png')
        plt.savefig(acc_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Accuracy curve saved to: {acc_file}")
    
    # 绘制learning rate曲线
    if 'lr' in metrics and metrics['lr']:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['lr'], linewidth=2, color='orange')
        plt.xlabel('Update Steps', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title(f'Learning Rate Schedule{" - " + exp_name if exp_name else ""}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        lr_file = os.path.join(output_dir, 'lr_curve.png')
        plt.savefig(lr_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Learning rate curve saved to: {lr_file}")


def compare_experiments(exp_dirs, metric_key='accuracy', output_file='comparison.png'):
    """对比多个实验的结果"""
    plt.figure(figsize=(12, 7))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for idx, exp_dir in enumerate(exp_dirs):
        exp_path = Path(exp_dir)
        exp_name = exp_path.name
        
        # 查找训练日志
        log_file = exp_path / 'logs' / 'train.log'
        if not log_file.exists():
            print(f"Warning: Log file not found for {exp_name}")
            continue
        
        # 提取指标
        metrics = parse_train_log(log_file)
        
        if metric_key in metrics and metrics[metric_key]:
            color = colors[idx % len(colors)]
            plt.plot(metrics[metric_key], label=exp_name, 
                    linewidth=2, color=color, alpha=0.8)
    
    plt.xlabel('Update Steps', fontsize=12)
    plt.ylabel(metric_key.capitalize(), fontsize=12)
    plt.title(f'{metric_key.capitalize()} Comparison', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Comparison plot saved to: {output_file}")


def plot_metrics_summary(exp_dirs, output_file='metrics_summary.png'):
    """绘制多个实验的指标汇总条形图"""
    exp_names = []
    accuracies = []
    f1_scores = []
    
    for exp_dir in exp_dirs:
        exp_path = Path(exp_dir)
        exp_name = exp_path.name
        
        # 读取metrics.json
        metrics_file = exp_path / 'results' / 'metrics.json'
        if not metrics_file.exists():
            print(f"Warning: Metrics file not found for {exp_name}")
            continue
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        exp_names.append(exp_name)
        accuracies.append(metrics.get('token_accuracy', 0))
        f1_scores.append(metrics.get('f1', 0))
    
    if not exp_names:
        print("No valid metrics found")
        return
    
    # 绘制条形图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy
    ax1.bar(range(len(exp_names)), accuracies, color='steelblue', alpha=0.8)
    ax1.set_xticks(range(len(exp_names)))
    ax1.set_xticklabels(exp_names, rotation=45, ha='right')
    ax1.set_ylabel('Token Accuracy', fontsize=12)
    ax1.set_title('Token Accuracy Comparison', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    
    # F1 Score
    ax2.bar(range(len(exp_names)), f1_scores, color='forestgreen', alpha=0.8)
    ax2.set_xticks(range(len(exp_names)))
    ax2.set_xticklabels(exp_names, rotation=45, ha='right')
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('F1 Score Comparison', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, v in enumerate(f1_scores):
        ax2.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Metrics summary saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Visualize training logs')
    parser.add_argument('--log-file', type=str,
                       help='Single training log file to visualize')
    parser.add_argument('--output-dir', type=str, default='./plots',
                       help='Output directory for plots')
    parser.add_argument('--exp-name', type=str, default='',
                       help='Experiment name for plot title')
    
    # 对比模式
    parser.add_argument('--compare', action='store_true',
                       help='Compare multiple experiments')
    parser.add_argument('--exp-dirs', nargs='+',
                       help='List of experiment directories to compare')
    parser.add_argument('--metric', type=str, default='accuracy',
                       choices=['loss', 'accuracy', 'lr'],
                       help='Metric to compare')
    
    args = parser.parse_args()
    
    if args.compare:
        if not args.exp_dirs:
            print("Error: --exp-dirs required for comparison mode")
            return
        
        print(f"Comparing {len(args.exp_dirs)} experiments...")
        
        # 绘制训练曲线对比
        compare_file = os.path.join(args.output_dir, f'{args.metric}_comparison.png')
        compare_experiments(args.exp_dirs, args.metric, compare_file)
        
        # 绘制指标汇总
        summary_file = os.path.join(args.output_dir, 'metrics_summary.png')
        plot_metrics_summary(args.exp_dirs, summary_file)
        
    else:
        if not args.log_file:
            print("Error: --log-file required for single experiment visualization")
            return
        
        if not os.path.exists(args.log_file):
            print(f"Error: Log file not found: {args.log_file}")
            return
        
        print(f"Parsing log file: {args.log_file}")
        metrics = parse_train_log(args.log_file)
        
        print(f"Found metrics: {list(metrics.keys())}")
        for key, values in metrics.items():
            print(f"  {key}: {len(values)} data points")
        
        print(f"\nGenerating plots...")
        plot_training_curves(metrics, args.output_dir, args.exp_name)
        
        # 保存原始数据
        metrics_json = os.path.join(args.output_dir, 'training_metrics.json')
        with open(metrics_json, 'w') as f:
            json.dump({k: v for k, v in metrics.items()}, f, indent=2)
        print(f"✓ Metrics data saved to: {metrics_json}")
    
    print("\n✓ Visualization completed!")


if __name__ == '__main__':
    main()
