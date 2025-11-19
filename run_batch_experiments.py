#!/usr/bin/env python3
"""
批量运行Transformer实验
支持串行、并行、队列模式
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
import multiprocessing as mp

class BatchExperimentRunner:
    def __init__(self, exp_dir="experiments/transformer_series", 
                 data_dir="/mnt/data1/zhaojunzhang/typilus-data/transformer"):
        self.exp_dir = Path(exp_dir)
        self.base_dir = self.exp_dir  # 别名
        self.data_dir = Path(data_dir)
        self.index_file = self.exp_dir / "experiment_index.json"
        
        if not self.index_file.exists():
            raise FileNotFoundError(f"Experiment index not found: {self.index_file}")
        
        with open(self.index_file) as f:
            self.index = json.load(f)
    
    def get_experiments(self, group=None, names=None):
        """获取实验列表"""
        experiments = self.index['experiments']
        
        if names:
            experiments = [exp for exp in experiments if exp['name'] in names]
        elif group:
            # 按组过滤
            group_prefixes = {
                'baseline': ['exp_baseline'],
                'model_size': ['exp_d_model_'],
                'layers': ['exp_layers_'],
                'lr': ['exp_lr_'],
                'dropout': ['exp_dropout_'],
                'encoder': ['exp_encoder_'],
                'batch_size': ['exp_batch_'],
            }
            
            if group in group_prefixes:
                prefixes = group_prefixes[group]
                experiments = [exp for exp in experiments 
                             if any(exp['name'].startswith(p) for p in prefixes)]
        
        return experiments
    
    def run_experiment(self, exp_name, gpu_id=0, dry_run=False):
        """运行单个实验"""
        exp_path = self.exp_dir / exp_name
        
        if not exp_path.exists():
            print(f"❌ Experiment not found: {exp_name}")
            return False
        
        # 读取配置
        config_path = exp_path / "config.yml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # 构建命令
        cmd = self._build_command(exp_name, config, gpu_id)
        
        print(f"\n{'='*60}")
        print(f"Running: {exp_name}")
        print(f"GPU: {gpu_id}")
        print(f"Time: {datetime.now().isoformat()}")
        print(f"{'='*60}")
        
        if dry_run:
            print(f"Command: {' '.join(cmd)}")
            return True
        
        # 更新状态
        self._update_status(exp_name, 'running', {'start_time': datetime.now().isoformat()})
        
        # 运行实验
        log_file = exp_path / "logs" / "train.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        try:
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu_id)}
                )
                
                # 实时输出并保存
                for line in process.stdout:
                    print(line, end='')
                    f.write(line)
                    f.flush()
                
                process.wait()
            
            elapsed = time.time() - start_time
            success = process.returncode == 0
            
            if success:
                self._update_status(exp_name, 'completed', {
                    'end_time': datetime.now().isoformat(),
                    'elapsed_seconds': elapsed,
                    'return_code': 0
                })
                print(f"\n✓ {exp_name} completed in {elapsed/3600:.2f} hours")
            else:
                self._update_status(exp_name, 'failed', {
                    'end_time': datetime.now().isoformat(),
                    'elapsed_seconds': elapsed,
                    'return_code': process.returncode
                })
                print(f"\n❌ {exp_name} failed with code {process.returncode}")
            
            return success
            
        except Exception as e:
            self._update_status(exp_name, 'error', {
                'end_time': datetime.now().isoformat(),
                'error': str(e)
            })
            print(f"\n❌ {exp_name} error: {e}")
            return False
    
    def _build_command(self, exp_name, config, gpu_id):
        """构建训练命令"""
        cmd = [
            'python', 'run_transformer_experiment.py',
            '--exp-name', exp_name,
            '--base-dir', str(self.base_dir),
            '--data-dir', str(self.data_dir),
        ]
        
        # 模型参数（使用训练脚本实际支持的参数）
        model_config = config.get('model', {})
        
        # encoder_type: lstm/transformer
        if 'encoder_type' in model_config:
            cmd.extend(['--encoder-type', str(model_config['encoder_type'])])
        
        # encoder_layers: 层数
        if 'encoder_layers' in model_config:
            cmd.extend(['--encoder-layers', str(model_config['encoder_layers'])])
        elif 'n_encoder_layers' in model_config:
            cmd.extend(['--encoder-layers', str(model_config['n_encoder_layers'])])
        
        # encoder_embed_dim: 嵌入维度
        if 'encoder_embed_dim' in model_config:
            cmd.extend(['--encoder-embed-dim', str(model_config['encoder_embed_dim'])])
        elif 'd_model' in model_config:
            cmd.extend(['--encoder-embed-dim', str(model_config['d_model'])])
        
        # dropout
        if 'dropout' in model_config:
            cmd.extend(['--dropout', str(model_config['dropout'])])
        
        # 训练参数
        optimization = config.get('optimization', {})
        
        # 学习率
        if 'lr' in optimization:
            lr_value = optimization['lr']
            # 如果是列表，取第一个值
            if isinstance(lr_value, list):
                lr_value = lr_value[0]
            cmd.extend(['--lr', str(lr_value)])
        
        # 最大epoch
        if 'max_epoch' in optimization:
            cmd.extend(['--max-epoch', str(optimization['max_epoch'])])
        
        # warmup
        if 'warmup_updates' in optimization:
            cmd.extend(['--warmup-updates', str(optimization['warmup_updates'])])
        
        # batch size
        dataset_config = config.get('dataset', {})
        if 'max_sentences' in dataset_config:
            cmd.extend(['--batch-size', str(dataset_config['max_sentences'])])
        elif 'batch_size' in config.get('training', {}):
            cmd.extend(['--batch-size', str(config['training']['batch_size'])])
        
        return cmd
    
    def _update_status(self, exp_name, status, info=None):
        """更新实验状态"""
        meta_path = self.exp_dir / exp_name / "meta.json"
        
        with open(meta_path) as f:
            meta = json.load(f)
        
        meta['status'] = status
        meta['last_updated'] = datetime.now().isoformat()
        
        if info:
            if 'run_info' not in meta:
                meta['run_info'] = {}
            meta['run_info'].update(info)
        
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
    
    def run_batch(self, experiments, mode='serial', gpus=None, dry_run=False):
        """批量运行实验"""
        if gpus is None:
            gpus = [0]
        
        print(f"\n{'='*60}")
        print(f"Batch Experiment Runner")
        print(f"Mode: {mode}")
        print(f"Experiments: {len(experiments)}")
        print(f"GPUs: {gpus}")
        print(f"{'='*60}\n")
        
        if mode == 'serial':
            # 串行运行
            results = []
            for i, exp in enumerate(experiments):
                gpu_id = gpus[i % len(gpus)]
                success = self.run_experiment(exp['name'], gpu_id, dry_run)
                results.append((exp['name'], success))
        
        elif mode == 'parallel':
            # 并行运行（一个GPU一个实验）
            if len(experiments) > len(gpus):
                print(f"Warning: {len(experiments)} experiments but only {len(gpus)} GPUs")
                print(f"Will run in batches")
            
            results = []
            for i in range(0, len(experiments), len(gpus)):
                batch = experiments[i:i+len(gpus)]
                batch_results = []
                
                for j, exp in enumerate(batch):
                    gpu_id = gpus[j]
                    success = self.run_experiment(exp['name'], gpu_id, dry_run)
                    batch_results.append((exp['name'], success))
                
                results.extend(batch_results)
        
        # 打印总结
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results):
        """打印运行总结"""
        total = len(results)
        success = sum(1 for _, s in results if s)
        failed = total - success
        
        print(f"\n{'='*60}")
        print(f"Batch Run Summary")
        print(f"{'='*60}")
        print(f"Total: {total}")
        print(f"Success: {success}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {success/total*100:.1f}%")
        
        if failed > 0:
            print(f"\nFailed experiments:")
            for name, success in results:
                if not success:
                    print(f"  - {name}")
        
        print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description='Run batch experiments')
    parser.add_argument('--base-dir', '--exp-dir', type=str, 
                       default='experiments/transformer_series',
                       dest='exp_dir',
                       help='Experiments directory')
    parser.add_argument('--data-dir', type=str,
                       default='/mnt/data1/zhaojunzhang/typilus-data/transformer',
                       help='Data directory')
    parser.add_argument('--group', type=str, 
                       choices=['baseline', 'model_size', 'layers', 'lr', 'dropout', 
                               'encoder', 'batch_size'],
                       help='Run experiments in a specific group')
    parser.add_argument('--names', nargs='+',
                       help='Specific experiment names to run')
    parser.add_argument('--all', action='store_true',
                       help='Run all experiments')
    parser.add_argument('--mode', type=str, default='serial',
                       choices=['serial', 'parallel'],
                       help='Execution mode')
    parser.add_argument('--gpus', '--gpu', nargs='+', type=int, default=[0],
                       dest='gpus',
                       help='GPU IDs to use')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without running')
    
    args = parser.parse_args()
    
    runner = BatchExperimentRunner(exp_dir=args.exp_dir, data_dir=args.data_dir)
    
    # 获取实验列表
    if args.all:
        experiments = runner.get_experiments()
    elif args.names:
        experiments = runner.get_experiments(names=args.names)
    elif args.group:
        experiments = runner.get_experiments(group=args.group)
    else:
        print("Please specify --all, --group, or --names")
        sys.exit(1)
    
    if not experiments:
        print("No experiments found")
        sys.exit(1)
    
    # 运行实验
    runner.run_batch(experiments, mode=args.mode, gpus=args.gpus, dry_run=args.dry_run)

if __name__ == '__main__':
    main()
