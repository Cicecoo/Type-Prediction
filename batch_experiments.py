#!/usr/bin/env python3
"""
批量实验运行工具

功能：
1. 定义多组超参数配置
2. 自动运行所有实验
3. 汇总对比所有实验结果
4. 生成实验报告
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from itertools import product


# 预定义的实验配置模板
EXPERIMENT_CONFIGS = {
    'baseline': {
        'encoder_type': 'lstm',
        'encoder_layers': 2,
        'encoder_embed_dim': 512,
        'dropout': 0.1,
        'lr': 0.0001,
        'batch_size': 16,
        'max_epoch': 50,
    },
    
    'larger_model': {
        'encoder_type': 'lstm',
        'encoder_layers': 4,
        'encoder_embed_dim': 768,
        'dropout': 0.1,
        'lr': 0.0001,
        'batch_size': 16,
        'max_epoch': 50,
    },
    
    'high_dropout': {
        'encoder_type': 'lstm',
        'encoder_layers': 2,
        'encoder_embed_dim': 512,
        'dropout': 0.3,
        'lr': 0.0001,
        'batch_size': 16,
        'max_epoch': 50,
    },
    
    'higher_lr': {
        'encoder_type': 'lstm',
        'encoder_layers': 2,
        'encoder_embed_dim': 512,
        'dropout': 0.1,
        'lr': 0.0005,
        'batch_size': 16,
        'max_epoch': 50,
    },
    
    'lower_lr': {
        'encoder_type': 'lstm',
        'encoder_layers': 2,
        'encoder_embed_dim': 512,
        'dropout': 0.1,
        'lr': 0.00005,
        'batch_size': 16,
        'max_epoch': 50,
    },
    
    'larger_batch': {
        'encoder_type': 'lstm',
        'encoder_layers': 2,
        'encoder_embed_dim': 512,
        'dropout': 0.1,
        'lr': 0.0001,
        'batch_size': 32,
        'max_epoch': 50,
    },
}


class BatchExperimentRunner:
    """批量实验运行器"""
    
    def __init__(self, base_dir, data_dir, python_path='python'):
        self.base_dir = Path(base_dir)
        self.data_dir = Path(data_dir)
        self.python_path = python_path
        self.results = []
        
        # 确保基础目录存在
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def run_experiment(self, exp_name, config):
        """运行单个实验"""
        print(f"\n{'='*80}")
        print(f"Running experiment: {exp_name}")
        print(f"{'='*80}")
        print(f"Config: {json.dumps(config, indent=2)}")
        
        # 构建命令
        cmd = [
            self.python_path,
            'run_transformer_experiment.py',
            '--exp-name', exp_name,
            '--base-dir', str(self.base_dir),
            '--data-dir', str(self.data_dir),
        ]
        
        # 添加配置参数
        for key, value in config.items():
            cmd.extend([f'--{key.replace("_", "-")}', str(value)])
        
        print(f"\nCommand: {' '.join(cmd)}\n")
        
        # 运行实验
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                text=True
            )
            
            print(f"\n✓ Experiment {exp_name} completed successfully!")
            
            # 记录结果
            exp_dir = self.base_dir / exp_name
            metrics_file = exp_dir / 'results' / 'metrics.json'
            
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                self.results.append({
                    'name': exp_name,
                    'config': config,
                    'metrics': metrics,
                    'status': 'success'
                })
            else:
                self.results.append({
                    'name': exp_name,
                    'config': config,
                    'status': 'success',
                    'metrics': None
                })
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Experiment {exp_name} failed!")
            self.results.append({
                'name': exp_name,
                'config': config,
                'status': 'failed',
                'error': str(e)
            })
            return False
    
    def run_all_experiments(self, configs):
        """运行所有实验"""
        total = len(configs)
        success_count = 0
        
        print(f"\n{'='*80}")
        print(f"Starting batch experiments: {total} experiments")
        print(f"{'='*80}\n")
        
        for idx, (exp_name, config) in enumerate(configs.items(), 1):
            print(f"\n[{idx}/{total}] {exp_name}")
            
            if self.run_experiment(exp_name, config):
                success_count += 1
        
        print(f"\n{'='*80}")
        print(f"Batch experiments completed!")
        print(f"Success: {success_count}/{total}")
        print(f"{'='*80}\n")
    
    def generate_summary_report(self, output_file):
        """生成汇总报告"""
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Batch Experiments Summary Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Experiments: {len(self.results)}\n")
            success = sum(1 for r in self.results if r['status'] == 'success')
            failed = len(self.results) - success
            f.write(f"Successful: {success}\n")
            f.write(f"Failed: {failed}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("Experiment Results:\n")
            f.write("-"*80 + "\n\n")
            
            # 按准确率排序
            sorted_results = sorted(
                [r for r in self.results if r.get('metrics')],
                key=lambda x: x['metrics'].get('token_accuracy', 0),
                reverse=True
            )
            
            for idx, result in enumerate(sorted_results, 1):
                f.write(f"{idx}. {result['name']}\n")
                f.write(f"   Status: {result['status']}\n")
                
                if result.get('metrics'):
                    metrics = result['metrics']
                    f.write(f"   Token Accuracy: {metrics.get('token_accuracy', 0):.4f}\n")
                    f.write(f"   F1 Score:       {metrics.get('f1', 0):.4f}\n")
                    f.write(f"   Precision:      {metrics.get('precision', 0):.4f}\n")
                    f.write(f"   Recall:         {metrics.get('recall', 0):.4f}\n")
                
                f.write(f"   Config:\n")
                for key, value in result['config'].items():
                    f.write(f"     {key}: {value}\n")
                f.write("\n")
            
            # 失败的实验
            if failed > 0:
                f.write("-"*80 + "\n")
                f.write("Failed Experiments:\n")
                f.write("-"*80 + "\n\n")
                
                for result in self.results:
                    if result['status'] == 'failed':
                        f.write(f"- {result['name']}\n")
                        if 'error' in result:
                            f.write(f"  Error: {result['error']}\n")
                        f.write("\n")
        
        print(f"✓ Summary report saved to: {output_file}")
    
    def save_results_json(self, output_file):
        """保存JSON格式的结果"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Results JSON saved to: {output_file}")
    
    def visualize_comparison(self):
        """生成可视化对比"""
        exp_dirs = [str(self.base_dir / r['name']) for r in self.results if r['status'] == 'success']
        
        if not exp_dirs:
            print("No successful experiments to visualize")
            return
        
        output_dir = self.base_dir / 'comparison_plots'
        output_dir.mkdir(exist_ok=True)
        
        cmd = [
            self.python_path,
            'visualize_training.py',
            '--compare',
            '--exp-dirs'] + exp_dirs + [
            '--output-dir', str(output_dir),
        ]
        
        print(f"\nGenerating comparison plots...")
        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Comparison plots saved to: {output_dir}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Visualization failed: {e}")


def grid_search(param_grid):
    """生成网格搜索的所有配置组合"""
    keys = param_grid.keys()
    values = param_grid.values()
    
    configs = {}
    for idx, combination in enumerate(product(*values)):
        config_dict = dict(zip(keys, combination))
        
        # 生成配置名称
        name_parts = []
        for key, value in config_dict.items():
            if key == 'encoder_layers':
                name_parts.append(f"layers{value}")
            elif key == 'encoder_embed_dim':
                name_parts.append(f"dim{value}")
            elif key == 'dropout':
                name_parts.append(f"drop{value}")
            elif key == 'lr':
                name_parts.append(f"lr{value}")
            elif key == 'batch_size':
                name_parts.append(f"bs{value}")
        
        exp_name = f"grid_{'_'.join(name_parts)}"
        configs[exp_name] = config_dict
    
    return configs


def main():
    parser = argparse.ArgumentParser(description='Run batch experiments')
    
    parser.add_argument('--base-dir', type=str,
                       default='/mnt/data1/zhaojunzhang/experiments/transformer',
                       help='Base directory for experiments')
    parser.add_argument('--data-dir', type=str,
                       default='/mnt/data1/zhaojunzhang/typilus-data/transformer',
                       help='Data directory')
    parser.add_argument('--python', type=str, default='python',
                       help='Python executable')
    
    # 实验选择
    parser.add_argument('--mode', type=str, default='predefined',
                       choices=['predefined', 'grid'],
                       help='Experiment mode')
    parser.add_argument('--configs', nargs='+',
                       help='Config names to run (for predefined mode)')
    
    # 网格搜索参数
    parser.add_argument('--grid-lr', nargs='+', type=float,
                       help='Learning rates for grid search')
    parser.add_argument('--grid-dropout', nargs='+', type=float,
                       help='Dropout rates for grid search')
    parser.add_argument('--grid-layers', nargs='+', type=int,
                       help='Number of layers for grid search')
    parser.add_argument('--grid-dim', nargs='+', type=int,
                       help='Embedding dimensions for grid search')
    
    args = parser.parse_args()
    
    # 创建运行器
    runner = BatchExperimentRunner(args.base_dir, args.data_dir, args.python)
    
    # 选择实验配置
    if args.mode == 'predefined':
        if args.configs:
            configs = {name: EXPERIMENT_CONFIGS[name] 
                      for name in args.configs 
                      if name in EXPERIMENT_CONFIGS}
        else:
            configs = EXPERIMENT_CONFIGS
        
        print(f"Running {len(configs)} predefined experiments:")
        for name in configs.keys():
            print(f"  - {name}")
    
    elif args.mode == 'grid':
        # 构建参数网格
        param_grid = {
            'encoder_type': ['lstm'],
            'encoder_layers': args.grid_layers or [2],
            'encoder_embed_dim': args.grid_dim or [512],
            'dropout': args.grid_dropout or [0.1],
            'lr': args.grid_lr or [0.0001],
            'batch_size': [16],
            'max_epoch': [50],
        }
        
        configs = grid_search(param_grid)
        print(f"Generated {len(configs)} experiments from grid search")
    
    # 运行实验
    runner.run_all_experiments(configs)
    
    # 生成报告
    summary_file = runner.base_dir / 'batch_summary.txt'
    runner.generate_summary_report(summary_file)
    
    results_file = runner.base_dir / 'batch_results.json'
    runner.save_results_json(results_file)
    
    # 生成可视化
    runner.visualize_comparison()
    
    print(f"\n{'='*80}")
    print("All experiments completed!")
    print(f"Summary: {summary_file}")
    print(f"Results: {results_file}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
