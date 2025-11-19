#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析 Typilus 学习率实验结果
"""

import json
import os
from pathlib import Path
from tabulate import tabulate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_experiment_results(experiments_dir):
    """加载所有实验结果"""
    experiments_dir = Path(experiments_dir)
    results = []
    
    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        metrics_file = exp_dir / 'logs' / 'metrics.json'
        test_result_file = exp_dir / 'checkpoints' / 'res.txt'
        
        result = {
            'name': exp_name,
            'exp_dir': str(exp_dir)
        }
        
        # 加载训练指标
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            if metrics['epochs']:
                # 找到最佳 epoch (验证loss最低)
                valid_losses = metrics['valid_loss']
                if valid_losses and any(v > 0 for v in valid_losses):
                    best_idx = min(range(len(valid_losses)), 
                                 key=lambda i: valid_losses[i] if valid_losses[i] > 0 else float('inf'))
                    
                    result.update({
                        'total_epochs': len(metrics['epochs']),
                        'best_epoch': metrics['epochs'][best_idx],
                        'train_loss': metrics['train_loss'][best_idx],
                        'valid_loss': valid_losses[best_idx],
                        'gap': metrics['train_loss'][best_idx] - valid_losses[best_idx],
                        'final_train_loss': metrics['train_loss'][-1],
                        'final_valid_loss': valid_losses[-1],
                        'learning_rate': metrics['learning_rate'][best_idx] if metrics['learning_rate'] else 0,
                    })
        
        # 加载测试结果
        if test_result_file.exists():
            test_metrics = {}
            with open(test_result_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if ':' in line:
                        key, value = line.split(':', 1)
                        try:
                            test_metrics[key.strip()] = float(value.strip())
                        except:
                            test_metrics[key.strip()] = value.strip()
            
            result.update({
                'test_loss': test_metrics.get('avg_loss', None),
                'test_acc1': test_metrics.get('acc1', None),
                'test_acc5': test_metrics.get('acc5', None),
                'test_acc1_any': test_metrics.get('acc1_any', None),
                'test_acc5_any': test_metrics.get('acc5_any', None),
            })
        
        results.append(result)
    
    return results


def print_analysis(results):
    """打印分析结果"""
    print("\n" + "="*120)
    print("Typilus 学习率实验结果分析")
    print("="*120 + "\n")
    
    # 过滤有效结果
    valid_results = [r for r in results if 'valid_loss' in r]
    
    if not valid_results:
        print("未找到有效的实验结果")
        return
    
    # 表1: 训练结果
    print("## 训练结果")
    print("-"*120)
    
    table_data = []
    for r in valid_results:
        lr_str = r['name'].replace('lr_', '')
        table_data.append([
            lr_str,
            r.get('best_epoch', 'N/A'),
            f"{r.get('train_loss', 0):.4f}",
            f"{r.get('valid_loss', 0):.4f}",
            f"{r.get('gap', 0):.4f}",
            f"{r.get('final_train_loss', 0):.4f}",
            f"{r.get('final_valid_loss', 0):.4f}",
        ])
    
    headers = ['学习率', '最佳Epoch', '最佳训练Loss', '最佳验证Loss', 'Gap', 
               '最终训练Loss', '最终验证Loss']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # 表2: 测试结果
    print("\n## 测试结果")
    print("-"*120)
    
    tested_results = [r for r in valid_results if r.get('test_acc1') is not None]
    if tested_results:
        table_data = []
        for r in tested_results:
            lr_str = r['name'].replace('lr_', '')
            table_data.append([
                lr_str,
                f"{r.get('test_loss', 0):.4f}",
                f"{r.get('test_acc1', 0):.2f}%",
                f"{r.get('test_acc5', 0):.2f}%",
                f"{r.get('test_acc1_any', 0):.2f}%",
                f"{r.get('test_acc5_any', 0):.2f}%",
            ])
        
        headers = ['学习率', '测试Loss', 'Acc@1', 'Acc@5', 
                   'Acc@1(含any)', 'Acc@5(含any)']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
    else:
        print("没有实验完成测试")
        print("需要测试的实验:", [r['name'] for r in valid_results])
    
    # 排序和推荐
    print("\n## 最佳配置")
    print("-"*120)
    
    # 按验证loss排序
    sorted_by_valid = sorted(valid_results, key=lambda x: x.get('valid_loss', float('inf')))
    best_valid = sorted_by_valid[0]
    
    print(f"\n最佳验证Loss: {best_valid['name']} (lr={best_valid['name'].replace('lr_', '')})")
    print(f"  - 最佳Epoch: {best_valid.get('best_epoch')}")
    print(f"  - 训练Loss: {best_valid.get('train_loss', 0):.4f}")
    print(f"  - 验证Loss: {best_valid.get('valid_loss', 0):.4f}")
    print(f"  - Gap: {best_valid.get('gap', 0):.4f}")
    if best_valid.get('test_acc1'):
        print(f"  - 测试Acc@1: {best_valid.get('test_acc1', 0):.2f}%")
        print(f"  - 测试Acc@5: {best_valid.get('test_acc5', 0):.2f}%")
    
    # 按gap排序（泛化能力）
    sorted_by_gap = sorted(valid_results, key=lambda x: x.get('gap', float('inf')))
    best_gap = sorted_by_gap[0]
    
    if best_gap['name'] != best_valid['name']:
        print(f"\n最佳泛化能力: {best_gap['name']} (Gap={best_gap.get('gap', 0):.4f})")
        print(f"  - 验证Loss: {best_gap.get('valid_loss', 0):.4f}")
    
    # 如果有测试结果，显示测试最佳
    if tested_results:
        sorted_by_test = sorted(tested_results, key=lambda x: -x.get('test_acc1', 0))
        best_test = sorted_by_test[0]
        
        print(f"\n最佳测试准确率: {best_test['name']}")
        print(f"  - Acc@1: {best_test.get('test_acc1', 0):.2f}%")
        print(f"  - Acc@5: {best_test.get('test_acc5', 0):.2f}%")
    
    print("\n" + "="*120)
    
    return valid_results, tested_results


def plot_results(results, output_dir):
    """绘制对比图"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    valid_results = [r for r in results if 'valid_loss' in r]
    if not valid_results:
        return
    
    # 提取学习率值
    lrs = []
    for r in valid_results:
        lr_str = r['name'].replace('lr_', '').replace('e-', 'e-0')
        try:
            lrs.append(float(lr_str))
        except:
            lrs.append(0)
    
    train_losses = [r.get('train_loss', 0) for r in valid_results]
    valid_losses = [r.get('valid_loss', 0) for r in valid_results]
    gaps = [r.get('gap', 0) for r in valid_results]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss对比
    axes[0, 0].plot(lrs, train_losses, 'o-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(lrs, valid_losses, 's-', label='Valid Loss', linewidth=2)
    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss vs Learning Rate')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].set_xscale('log')
    
    # Gap对比
    axes[0, 1].plot(lrs, gaps, 'o-', color='red', linewidth=2)
    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Learning Rate')
    axes[0, 1].set_ylabel('Train - Valid Loss')
    axes[0, 1].set_title('Generalization Gap vs Learning Rate')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].set_xscale('log')
    
    # 测试准确率对比
    tested_results = [r for r in valid_results if r.get('test_acc1') is not None]
    if tested_results:
        test_lrs = []
        for r in tested_results:
            lr_str = r['name'].replace('lr_', '').replace('e-', 'e-0')
            try:
                test_lrs.append(float(lr_str))
            except:
                test_lrs.append(0)
        
        test_acc1 = [r.get('test_acc1', 0) for r in tested_results]
        test_acc5 = [r.get('test_acc5', 0) for r in tested_results]
        
        axes[1, 0].plot(test_lrs, test_acc1, 'o-', label='Acc@1', linewidth=2)
        axes[1, 0].plot(test_lrs, test_acc5, 's-', label='Acc@5', linewidth=2)
        axes[1, 0].set_xlabel('Learning Rate')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_title('Test Accuracy vs Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].set_xscale('log')
    else:
        axes[1, 0].text(0.5, 0.5, 'No Test Results Yet', 
                       ha='center', va='center', fontsize=14)
        axes[1, 0].set_title('Test Accuracy (Pending)')
    
    # 最佳epoch对比
    best_epochs = [r.get('best_epoch', 0) for r in valid_results]
    axes[1, 1].bar(range(len(lrs)), best_epochs, color='green', alpha=0.7)
    axes[1, 1].set_xlabel('Experiment')
    axes[1, 1].set_ylabel('Best Epoch')
    axes[1, 1].set_title('Best Epoch by Learning Rate')
    axes[1, 1].set_xticks(range(len(lrs)))
    axes[1, 1].set_xticklabels([f"{lr:.1e}" for lr in lrs], rotation=45)
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_file = output_dir / 'comparison.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n对比图已保存: {plot_file}")


def save_report(results, output_file):
    """保存分析报告"""
    import time
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Typilus 学习率实验结果报告\n\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 训练结果表
        f.write("## 训练结果\n\n")
        valid_results = [r for r in results if 'valid_loss' in r]
        
        if valid_results:
            f.write("| 学习率 | 最佳Epoch | 最佳训练Loss | 最佳验证Loss | Gap | 最终训练Loss | 最终验证Loss |\n")
            f.write("|--------|-----------|--------------|--------------|-----|--------------|-------------|\n")
            
            for r in valid_results:
                lr_str = r['name'].replace('lr_', '')
                f.write(f"| {lr_str} | {r.get('best_epoch', 'N/A')} | "
                       f"{r.get('train_loss', 0):.4f} | {r.get('valid_loss', 0):.4f} | "
                       f"{r.get('gap', 0):.4f} | {r.get('final_train_loss', 0):.4f} | "
                       f"{r.get('final_valid_loss', 0):.4f} |\n")
        
        # 测试结果表
        f.write("\n## 测试结果\n\n")
        tested_results = [r for r in valid_results if r.get('test_acc1') is not None]
        
        if tested_results:
            f.write("| 学习率 | 测试Loss | Acc@1 | Acc@5 | Acc@1(含any) | Acc@5(含any) |\n")
            f.write("|--------|----------|-------|-------|--------------|-------------|\n")
            
            for r in tested_results:
                lr_str = r['name'].replace('lr_', '')
                f.write(f"| {lr_str} | {r.get('test_loss', 0):.4f} | "
                       f"{r.get('test_acc1', 0):.2f}% | {r.get('test_acc5', 0):.2f}% | "
                       f"{r.get('test_acc1_any', 0):.2f}% | {r.get('test_acc5_any', 0):.2f}% |\n")
        else:
            f.write("暂无测试结果\n\n")
            f.write("需要测试的实验:\n")
            for r in valid_results:
                f.write(f"- {r['name']}\n")
        
        # 最佳配置
        f.write("\n## 最佳配置\n\n")
        if valid_results:
            sorted_by_valid = sorted(valid_results, key=lambda x: x.get('valid_loss', float('inf')))
            best_valid = sorted_by_valid[0]
            
            f.write(f"**最佳验证Loss**: {best_valid['name']}\n")
            f.write(f"- 最佳Epoch: {best_valid.get('best_epoch')}\n")
            f.write(f"- 训练Loss: {best_valid.get('train_loss', 0):.4f}\n")
            f.write(f"- 验证Loss: {best_valid.get('valid_loss', 0):.4f}\n")
            f.write(f"- Gap: {best_valid.get('gap', 0):.4f}\n")
            
            if best_valid.get('test_acc1'):
                f.write(f"- 测试Acc@1: {best_valid.get('test_acc1', 0):.2f}%\n")
                f.write(f"- 测试Acc@5: {best_valid.get('test_acc5', 0):.2f}%\n")
    
    print(f"报告已保存: {output_file}")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='分析 Typilus 实验结果')
    parser.add_argument('--exp_dir', '-e', type=str, 
                       default='../experiments',
                       help='实验目录路径')
    
    args = parser.parse_args()
    
    exp_dir = Path(__file__).parent.parent / 'experiments'
    if not exp_dir.exists():
        exp_dir = Path(args.exp_dir)
    
    print(f"分析目录: {exp_dir}")
    
    # 加载结果
    results = load_experiment_results(exp_dir)
    
    if not results:
        print("未找到任何实验结果")
        return
    
    # 打印分析
    valid_results, tested_results = print_analysis(results)
    
    # 绘制图表
    plot_results(results, exp_dir)
    
    # 保存报告
    report_file = exp_dir / 'analysis_report.md'
    save_report(results, report_file)
    
    # 返回需要测试的实验列表
    untested = [r for r in valid_results if r.get('test_acc1') is None]
    if untested:
        print(f"\n需要测试的实验 ({len(untested)}):")
        for r in untested:
            print(f"  - {r['name']}: {r['exp_dir']}")
    
    return untested


if __name__ == '__main__':
    main()
