#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transformer 实验管理工具
与 Typilus 实验工具保持一致的接口和日志格式
"""

import os
import sys
import yaml
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
from copy import deepcopy


def load_experiments(config_file='experiments_lr.yml'):
    """加载实验配置"""
    config_path = Path(__file__).parent / config_file
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载基础配置
    base_config_file = config.get('base_config', 'config_base.yml')
    base_config_path = Path(__file__).parent / base_config_file
    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    experiments = []
    for exp_cfg in config['experiments']:
        # 深拷贝基础配置
        exp_config = deepcopy(base_config)
        
        # 递归合并changes
        def merge_dict(base, updates):
            for key, value in updates.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        
        if 'changes' in exp_cfg:
            merge_dict(exp_config, exp_cfg['changes'])
        
        experiments.append({
            'name': exp_cfg['name'],
            'description': exp_cfg.get('description', ''),
            'config': exp_config
        })
    
    return experiments


def save_experiment_config(experiment, output_dir):
    """保存实验配置到临时文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = output_dir / f"{experiment['name']}_config.yml"
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(experiment['config'], f, default_flow_style=False, allow_unicode=True)
    
    return config_file


def run_single_experiment(experiment, working_dir=None):
    """运行单个实验"""
    print("\n" + "="*80)
    print(f"实验: {experiment['name']}")
    print(f"描述: {experiment['description']}")
    print("="*80 + "\n")
    
    # 保存实验配置
    if working_dir is None:
        working_dir = Path(__file__).parent
    else:
        working_dir = Path(working_dir)
    
    config_file = save_experiment_config(experiment, working_dir / 'temp_configs')
    print(f"配置文件: {config_file}")
    
    # 构建训练命令
    train_script = Path(__file__).parent / 'train_enhanced.py'
    
    cmd = [
        sys.executable,
        str(train_script),
        '--config', str(config_file)
    ]
    
    print(f"命令: {' '.join(cmd)}\n")
    
    # 记录实验开始时间
    start_time = time.time()
    start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        # 运行实验
        result = subprocess.run(
            cmd,
            cwd=str(working_dir),
            check=True,
            capture_output=False,
            text=True
        )
        
        # 记录实验结束
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n实验 {experiment['name']} 完成")
        print(f"耗时: {duration/3600:.2f} 小时")
        
        return {
            'name': experiment['name'],
            'status': 'success',
            'start_time': start_datetime,
            'duration_seconds': duration,
            'duration_hours': duration / 3600
        }
        
    except subprocess.CalledProcessError as e:
        print(f"\n实验 {experiment['name']} 失败")
        print(f"错误: {e}")
        return {
            'name': experiment['name'],
            'status': 'failed',
            'start_time': start_datetime,
            'error': str(e)
        }
    except KeyboardInterrupt:
        print(f"\n实验 {experiment['name']} 被用户中断")
        return {
            'name': experiment['name'],
            'status': 'interrupted',
            'start_time': start_datetime
        }


def analyze_experiment_results(experiment_name, checkpoint_dir):
    """分析单个实验的结果"""
    checkpoint_dir = Path(checkpoint_dir).expanduser()
    
    results = {
        'name': experiment_name,
        'checkpoint_dir': str(checkpoint_dir)
    }
    
    # 读取metrics.json
    metrics_file = checkpoint_dir / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        if metrics['epochs']:
            # 找到最佳epoch
            valid_losses = metrics['valid_loss']
            if valid_losses and any(v > 0 for v in valid_losses):
                best_idx = min(range(len(valid_losses)), key=lambda i: valid_losses[i] if valid_losses[i] > 0 else float('inf'))
                
                results.update({
                    'total_epochs': len(metrics['epochs']),
                    'best_epoch': metrics['epochs'][best_idx],
                    'best_valid_loss': valid_losses[best_idx],
                    'best_train_loss': metrics['train_loss'][best_idx],
                    'final_lr': metrics['learning_rate'][-1] if metrics['learning_rate'] else 0,
                    'train_valid_gap': metrics['train_loss'][best_idx] - valid_losses[best_idx]
                })
    
    return results


def analyze_all_experiments(experiments):
    """分析所有实验结果"""
    print("\n" + "="*80)
    print("实验结果汇总")
    print("="*80 + "\n")
    
    all_results = []
    for exp in experiments:
        checkpoint_dir = exp['config']['checkpoint']['save_dir']
        results = analyze_experiment_results(exp['name'], checkpoint_dir)
        all_results.append(results)
    
    # 按最佳验证损失排序
    valid_results = [r for r in all_results if 'best_valid_loss' in r]
    if valid_results:
        valid_results.sort(key=lambda x: x['best_valid_loss'])
        
        print(f"{'实验名称':<30} {'最佳Epoch':<10} {'训练Loss':<12} {'验证Loss':<12} {'Gap':<10}")
        print("-" * 80)
        for r in valid_results:
            print(f"{r['name']:<30} "
                  f"{r['best_epoch']:<10} "
                  f"{r['best_train_loss']:<12.4f} "
                  f"{r['best_valid_loss']:<12.4f} "
                  f"{r['train_valid_gap']:<10.4f}")
        
        # 保存结果
        results_file = Path(__file__).parent / 'experiment_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存至: {results_file}")
    else:
        print("没有找到有效的实验结果")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='Transformer实验管理工具')
    parser.add_argument('--config', '-c', default='experiments_lr.yml',
                        help='实验配置文件')
    parser.add_argument('--exp', '-e', type=str,
                        help='运行指定实验（不指定则运行所有）')
    parser.add_argument('--analyze', '-a', action='store_true',
                        help='只分析结果，不运行实验')
    parser.add_argument('--list', '-l', action='store_true',
                        help='列出所有实验')
    
    args = parser.parse_args()
    
    # 加载实验配置
    experiments = load_experiments(args.config)
    
    if args.list:
        print("\n可用实验:")
        print("-" * 80)
        for exp in experiments:
            print(f"{exp['name']:<30} {exp['description']}")
        return
    
    if args.analyze:
        analyze_all_experiments(experiments)
        return
    
    # 运行实验
    if args.exp:
        # 运行指定实验
        exp = next((e for e in experiments if e['name'] == args.exp), None)
        if exp is None:
            print(f"错误: 找不到实验 '{args.exp}'")
            print("\n可用实验:")
            for e in experiments:
                print(f"  - {e['name']}")
            return
        
        experiments_to_run = [exp]
    else:
        # 运行所有实验
        experiments_to_run = experiments
    
    # 执行实验
    run_results = []
    for exp in experiments_to_run:
        result = run_single_experiment(exp)
        run_results.append(result)
    
    # 打印运行总结
    print("\n" + "="*80)
    print("实验运行总结")
    print("="*80)
    for result in run_results:
        status_symbol = "✓" if result['status'] == 'success' else "✗"
        print(f"{status_symbol} {result['name']:<30} {result['status']}")
        if 'duration_hours' in result:
            print(f"  耗时: {result['duration_hours']:.2f} 小时")
    
    # 分析结果
    if all(r['status'] == 'success' for r in run_results):
        print("\n分析实验结果...")
        analyze_all_experiments(experiments_to_run)


if __name__ == '__main__':
    main()
