#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Typilus 参数调优实验工具 - 一体化脚本
支持自动化实验、日志记录、可视化和结果分析
"""

import os
import sys
import json
import yaml
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# ==================== 实验配置 ====================

EXPERIMENTS = [
    {
        "name": "baseline",
        "desc": "基线实验",
        "params": {}  # 使用默认配置
    },
    {
        "name": "exp_lr_1e-3",
        "desc": "学习率1e-3",
        "params": {"optimization": {"lrs": [1e-3]}}
    },
    {
        "name": "exp_lr_1e-4",
        "desc": "学习率1e-4",
        "params": {"optimization": {"lrs": [1e-4]}}
    },
    {
        "name": "exp_batch_64",
        "desc": "批量大小64",
        "params": {"dataset": {"max_sentences": 64}}
    },
    {
        "name": "exp_hidden_128",
        "desc": "隐藏层128",
        "params": {"model": {"encoder_embed_dim": 128, "encoder_hidden_size": 128}}
    },
    {
        "name": "exp_best",
        "desc": "推荐配置",
        "params": {
            "optimization": {"lrs": [5e-4]},
            "dataset": {"max_sentences": 64},
            "model": {"encoder_embed_dim": 128, "encoder_hidden_size": 128, "encoder_layers": 4}
        }
    }
]

# ==================== 工具函数 ====================

def deep_update(base_dict, update_dict):
    """递归更新字典"""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def create_config(base_config, exp_config, output_path):
    """创建实验配置"""
    with open(base_config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if exp_config.get("params"):
        config = deep_update(config, exp_config["params"])
    
    exp_dir = Path(output_path).parent
    config["checkpoint"]["save_dir"] = str(exp_dir / "checkpoints")
    config["common"]["tensorboard_logdir"] = str(exp_dir / "tensorboard")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config


def run_single_experiment(exp_config, base_config, train_script):
    """运行单个实验"""
    name = exp_config["name"]
    print(f"\n{'='*60}")
    print(f"实验: {name} - {exp_config['desc']}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    exp_dir = Path("~/naturalcc/typilus/experiments").expanduser() / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = exp_dir / "config.yml"
    create_config(base_config, exp_config, str(config_path))
    
    # 保存实验信息
    with open(exp_dir / "info.txt", 'w', encoding='utf-8') as f:
        f.write(f"实验: {name}\n描述: {exp_config['desc']}\n")
        f.write(f"开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 运行训练（使用增强版脚本）
    log_file = exp_dir / "training.log"
    cmd = [sys.executable, train_script, "--yaml_file", f"experiments/{name}/config"]
    
    start_time = time.time()
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()
            process.wait()
        
        success = process.returncode == 0
        status = "成功" if success else f"失败({process.returncode})"
        print(f"\n{'✓' if success else '✗'} {name} {status}")
        
    except Exception as e:
        success = False
        status = f"异常: {str(e)}"
        print(f"\n✗ {name} {status}")
    
    # 更新信息
    duration = time.time() - start_time
    with open(exp_dir / "info.txt", 'a', encoding='utf-8') as f:
        f.write(f"结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"耗时: {duration/3600:.2f}小时\n状态: {status}\n")
    
    return success


def visualize_experiment(exp_dir):
    """为单个实验生成可视化"""
    metrics_file = Path(exp_dir) / "logs" / "metrics.json"
    if not metrics_file.exists():
        return
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        with open(metrics_file) as f:
            data = json.load(f)
        
        if not data.get('epochs'):
            return
        
        plots_dir = Path(exp_dir) / "logs" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        epochs = data['epochs']
        
        # Loss图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(epochs, data['train_loss'], 'o-', label='Train')
        if any(data['valid_loss']):
            ax1.plot(epochs, data['valid_loss'], 's-', label='Valid')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # PPL图
        if any(data.get('train_ppl', [])):
            ax2.plot(epochs, data['train_ppl'], 'o-', label='Train')
        if any(data.get('valid_ppl', [])):
            ax2.plot(epochs, data['valid_ppl'], 's-', label='Valid')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Perplexity')
        ax2.set_title('Perplexity')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'training.png', dpi=120, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        pass
    except Exception as e:
        print(f"可视化错误: {e}")


def analyze_all_experiments(exp_base_dir):
    """分析所有实验结果"""
    exp_base_dir = Path(exp_base_dir).expanduser()
    if not exp_base_dir.exists():
        print("实验目录不存在")
        return
    
    results = []
    for exp_dir in exp_base_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        metrics_file = exp_dir / "logs" / "metrics.json"
        if not metrics_file.exists():
            continue
        
        with open(metrics_file) as f:
            data = json.load(f)
        
        if data.get('valid_loss'):
            valid_losses = [v for v in data['valid_loss'] if v > 0]
            best_loss = min(valid_losses) if valid_losses else 0
        else:
            best_loss = 0
        
        results.append({
            'name': exp_dir.name,
            'epochs': len(data.get('epochs', [])),
            'best_loss': best_loss,
            'final_loss': data['train_loss'][-1] if data.get('train_loss') else 0
        })
    
    if not results:
        print("没有找到实验结果")
        return
    
    # 打印对比表
    print("\n" + "="*80)
    print("实验结果对比")
    print("="*80)
    print(f"{'实验名称':<25} {'轮数':<8} {'最终Loss':<12} {'最佳验证Loss':<15}")
    print("-"*80)
    
    for r in sorted(results, key=lambda x: x['best_loss'] if x['best_loss'] > 0 else 999):
        loss_str = f"{r['best_loss']:.4f}" if r['best_loss'] > 0 else "N/A"
        print(f"{r['name']:<25} {r['epochs']:<8} {r['final_loss']:<12.4f} {loss_str:<15}")
    
    print("="*80)
    
    # 生成对比图
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 验证Loss对比
        for exp_dir in exp_base_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            metrics_file = exp_dir / "logs" / "metrics.json"
            if not metrics_file.exists():
                continue
            
            with open(metrics_file) as f:
                data = json.load(f)
            
            if data.get('epochs') and data.get('valid_loss'):
                epochs = data['epochs']
                valid_loss = [v if v > 0 else None for v in data['valid_loss']]
                axes[0].plot(epochs, valid_loss, 'o-', label=exp_dir.name, linewidth=2)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Validation Loss')
        axes[0].set_title('验证Loss对比')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # 最佳Loss柱状图
        names = [r['name'] for r in results if r['best_loss'] > 0]
        losses = [r['best_loss'] for r in results if r['best_loss'] > 0]
        axes[1].bar(range(len(names)), losses)
        axes[1].set_xticks(range(len(names)))
        axes[1].set_xticklabels(names, rotation=45, ha='right')
        axes[1].set_ylabel('Best Validation Loss')
        axes[1].set_title('最佳验证Loss对比')
        axes[1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(exp_base_dir / 'comparison.png', dpi=120, bbox_inches='tight')
        print(f"\n对比图已保存: {exp_base_dir / 'comparison.png'}")
        
    except ImportError:
        print("\n提示: 安装matplotlib可生成对比图 (pip install matplotlib)")
    except Exception as e:
        print(f"\n生成对比图错误: {e}")
    
    # 生成简单报告
    report_file = exp_base_dir / "report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 实验结果报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| 实验名称 | 轮数 | 最终Loss | 最佳验证Loss |\n")
        f.write("|---------|------|---------|-------------|\n")
        for r in results:
            loss_str = f"{r['best_loss']:.4f}" if r['best_loss'] > 0 else "N/A"
            f.write(f"| {r['name']} | {r['epochs']} | {r['final_loss']:.4f} | {loss_str} |\n")
    
    print(f"报告已保存: {report_file}\n")


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description='Typilus 参数调优实验工具')
    parser.add_argument('--analyze', action='store_true', help='仅分析结果')
    parser.add_argument('--run-only', action='store_true', help='仅运行实验')
    parser.add_argument('--exp-dir', default='~/naturalcc/typilus/experiments', help='实验目录')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent
    base_config = script_dir / "config" / "typilus.yml"
    
    # 优先使用增强版脚本
    train_script = Path(__file__).parent / "train_enhanced.py"
    if not train_script.exists():
        train_script = script_dir / "train.py"
    
    if args.analyze:
        print("分析实验结果...\n")
        analyze_all_experiments(args.exp_dir)
        return
    
    if not args.run_only:
        print("="*60)
        print("Typilus 参数调优实验系统")
        print("="*60)
        print(f"\n共 {len(EXPERIMENTS)} 个实验:\n")
        for i, exp in enumerate(EXPERIMENTS, 1):
            print(f"{i}. {exp['name']}: {exp['desc']}")
        print("\n" + "="*60)
        input("\n按 Enter 开始实验...")
    
    # 运行实验
    results = []
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n进度: {i}/{len(EXPERIMENTS)}")
        success = run_single_experiment(exp, str(base_config), str(train_script))
        results.append((exp['name'], success))
        
        if not success:
            response = input("\n实验失败，继续? (y/n): ")
            if response.lower() != 'y':
                break
        
        if i < len(EXPERIMENTS):
            print("\n等待3秒...")
            time.sleep(3)
    
    # 打印总结
    print("\n" + "="*60)
    print("实验完成")
    print("="*60)
    for name, success in results:
        print(f"{'✓' if success else '✗'} {name}")
    print("="*60)
    
    # 自动分析（如果不是只运行模式）
    if not args.run_only:
        print("\n正在分析结果...")
        analyze_all_experiments(args.exp_dir)


if __name__ == "__main__":
    main()
