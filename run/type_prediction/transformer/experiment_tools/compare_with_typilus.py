#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比 Transformer 和 Typilus 实验结果
"""

import json
import sys
from pathlib import Path
from tabulate import tabulate


def load_metrics(checkpoint_dir):
    """从checkpoint目录加载metrics"""
    checkpoint_dir = Path(checkpoint_dir).expanduser()
    metrics_file = checkpoint_dir / 'metrics.json'
    
    if not metrics_file.exists():
        return None
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    if not metrics['epochs']:
        return None
    
    # 找到最佳epoch
    valid_losses = metrics['valid_loss']
    if not valid_losses or not any(v > 0 for v in valid_losses):
        return None
    
    best_idx = min(range(len(valid_losses)), 
                   key=lambda i: valid_losses[i] if valid_losses[i] > 0 else float('inf'))
    
    return {
        'total_epochs': len(metrics['epochs']),
        'best_epoch': metrics['epochs'][best_idx],
        'train_loss': metrics['train_loss'][best_idx],
        'valid_loss': valid_losses[best_idx],
        'train_ppl': metrics['train_ppl'][best_idx] if metrics['train_ppl'] else 0,
        'valid_ppl': metrics['valid_ppl'][best_idx] if metrics['valid_ppl'] else 0,
        'gap': metrics['train_loss'][best_idx] - valid_losses[best_idx],
    }


def compare_experiments():
    """对比实验结果"""
    
    # Transformer实验
    transformer_base_dir = Path('~/workspace/type_pred/naturalcc/run/type_prediction/transformer/checkpoints').expanduser()
    transformer_experiments = {
        'transformer_baseline': transformer_base_dir / 'baseline',
        'transformer_lr_5e5': transformer_base_dir / 'lr_5e5',
        'transformer_lr_1e4': transformer_base_dir / 'lr_1e4',
        'transformer_deep': transformer_base_dir / 'deep',
        'transformer_wide': transformer_base_dir / 'wide',
    }
    
    # Typilus实验
    typilus_base_dir = Path('~/workspace/type_pred/naturalcc/run/type_prediction/typilus/checkpoints').expanduser()
    typilus_experiments = {
        'typilus_baseline': typilus_base_dir / 'baseline',
        'typilus_lr_1e3': typilus_base_dir / 'lr_1e3',
        'typilus_batch_64': typilus_base_dir / 'batch_64',
    }
    
    print("\n" + "="*100)
    print("Transformer vs Typilus - 实验结果对比")
    print("="*100 + "\n")
    
    # 加载Transformer结果
    print("## Transformer 实验 (使用 token-sequence)")
    print("-"*100)
    transformer_results = []
    for name, checkpoint_dir in transformer_experiments.items():
        metrics = load_metrics(checkpoint_dir)
        if metrics:
            transformer_results.append([
                name.replace('transformer_', ''),
                metrics['best_epoch'],
                f"{metrics['train_loss']:.4f}",
                f"{metrics['valid_loss']:.4f}",
                f"{metrics['gap']:.4f}",
                f"{metrics['train_ppl']:.2f}" if metrics['train_ppl'] > 0 else 'N/A',
                f"{metrics['valid_ppl']:.2f}" if metrics['valid_ppl'] > 0 else 'N/A',
            ])
    
    if transformer_results:
        headers = ['实验名', '最佳Epoch', '训练Loss', '验证Loss', 'Gap', '训练PPL', '验证PPL']
        print(tabulate(transformer_results, headers=headers, tablefmt='grid'))
    else:
        print("未找到Transformer实验结果")
    
    print("\n")
    
    # 加载Typilus结果
    print("## Typilus 实验 (使用 nodes + edges)")
    print("-"*100)
    typilus_results = []
    for name, checkpoint_dir in typilus_experiments.items():
        metrics = load_metrics(checkpoint_dir)
        if metrics:
            typilus_results.append([
                name.replace('typilus_', ''),
                metrics['best_epoch'],
                f"{metrics['train_loss']:.4f}",
                f"{metrics['valid_loss']:.4f}",
                f"{metrics['gap']:.4f}",
                f"{metrics['train_ppl']:.2f}" if metrics['train_ppl'] > 0 else 'N/A',
                f"{metrics['valid_ppl']:.2f}" if metrics['valid_ppl'] > 0 else 'N/A',
            ])
    
    if typilus_results:
        headers = ['实验名', '最佳Epoch', '训练Loss', '验证Loss', 'Gap', '训练PPL', '验证PPL']
        print(tabulate(typilus_results, headers=headers, tablefmt='grid'))
    else:
        print("未找到Typilus实验结果")
    
    print("\n")
    
    # 关键对比
    if transformer_results and typilus_results:
        print("## 关键对比")
        print("-"*100)
        
        # 找到各自最佳
        transformer_best = min(transformer_results, key=lambda x: float(x[3]))
        typilus_best = min(typilus_results, key=lambda x: float(x[3]))
        
        print(f"\n最佳 Transformer 实验: {transformer_best[0]}")
        print(f"  - 验证Loss: {transformer_best[3]}")
        print(f"  - 训练-验证Gap: {transformer_best[4]}")
        print(f"  - 最佳Epoch: {transformer_best[1]}")
        
        print(f"\n最佳 Typilus 实验: {typilus_best[0]}")
        print(f"  - 验证Loss: {typilus_best[3]}")
        print(f"  - 训练-验证Gap: {typilus_best[4]}")
        print(f"  - 最佳Epoch: {typilus_best[1]}")
        
        # 性能对比
        trans_loss = float(transformer_best[3])
        typi_loss = float(typilus_best[3])
        
        print(f"\n性能差异:")
        if trans_loss < typi_loss:
            improvement = ((typi_loss - trans_loss) / typi_loss) * 100
            print(f"  ✓ Transformer 更好 (降低 {improvement:.2f}%)")
        elif typi_loss < trans_loss:
            improvement = ((trans_loss - typi_loss) / trans_loss) * 100
            print(f"  ✓ Typilus (GNN) 更好 (降低 {improvement:.2f}%)")
        else:
            print(f"  = 性能相近")
        
        # 过拟合对比
        trans_gap = float(transformer_best[4])
        typi_gap = float(typilus_best[4])
        
        print(f"\n过拟合情况:")
        print(f"  - Transformer Gap: {trans_gap:.4f}")
        print(f"  - Typilus Gap: {typi_gap:.4f}")
        if trans_gap < typi_gap:
            print(f"  → Transformer 泛化更好")
        elif typi_gap < trans_gap:
            print(f"  → Typilus 泛化更好")
        else:
            print(f"  → 泛化能力相近")
    
    print("\n" + "="*100)
    print("说明:")
    print("  - Loss越低越好")
    print("  - Gap (训练Loss - 验证Loss) 越小说明过拟合越少")
    print("  - Perplexity (PPL) 越低越好")
    print("="*100 + "\n")


if __name__ == '__main__':
    try:
        compare_experiments()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
