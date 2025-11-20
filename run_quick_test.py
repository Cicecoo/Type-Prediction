#!/usr/bin/env python3
"""
快速测试实验 - 用于验证训练流程
使用极小的配置快速完成一轮训练+验证
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Quick test experiment')
    parser.add_argument('--exp-name', type=str, default='quick_test',
                       help='Experiment name')
    parser.add_argument('--base-dir', type=str, 
                       default='/mnt/data1/zhaojunzhang/experiments/quick_test',
                       help='Base directory')
    parser.add_argument('--data-dir', type=str,
                       default='/mnt/data1/zhaojunzhang/typilus-data/transformer',
                       help='Data directory')
    
    args = parser.parse_args()
    
    # 使用最小配置快速测试
    cmd = [
        'python', 'run_transformer_experiment.py',
        '--exp-name', args.exp_name,
        '--base-dir', args.base_dir,
        '--data-dir', args.data_dir,
        # 最小模型配置
        '--encoder-type', 'lstm',
        '--encoder-layers', '1',
        '--encoder-embed-dim', '128',
        '--dropout', '0.1',
        # 快速训练配置
        '--lr', '0.001',
        '--batch-size', '16',
        '--max-epoch', '2',  # 只训练2个epoch
        '--warmup-updates', '100',
    ]
    
    print(f"{'='*60}")
    print(f"Quick Test Experiment")
    print(f"{'='*60}")
    print(f"Experiment: {args.exp_name}")
    print(f"Base dir: {args.base_dir}")
    print(f"Data dir: {args.data_dir}")
    print(f"Config: LSTM, 1 layer, 128 dim, 2 epochs")
    print(f"{'='*60}\n")
    print(f"Command: {' '.join(cmd)}\n")
    
    import subprocess
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()
