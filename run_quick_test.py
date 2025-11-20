#!/usr/bin/env python3
"""
快速测试实验 - 用于验证训练流程
1. 创建小数据集（采样1000条）
2. 使用极小配置训练2个epoch
3. 验证
4. 测试
"""

import sys
import os
import argparse
import subprocess
import shutil
from pathlib import Path

def sample_data(src_dir, dst_dir, n_samples=1000):
    """采样创建小数据集"""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    
    print(f"\n{'='*60}")
    print(f"Sampling small dataset: {n_samples} samples")
    print(f"Source: {src_dir}")
    print(f"Target: {dst_dir}")
    print(f"{'='*60}\n")
    
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制词典文件
    for dict_file in ['nodes.dict', 'supernodes.dict', 'types.dict']:
        src_file = src_dir / dict_file
        if src_file.exists():
            shutil.copy(src_file, dst_dir / dict_file)
            print(f"✓ Copied {dict_file}")
    
    # 采样train/valid/test数据
    for split in ['train', 'valid', 'test']:
        for ext in ['nodes', 'supernodes', 'types']:
            src_file = src_dir / f"{split}.{ext}"
            dst_file = dst_dir / f"{split}.{ext}"
            
            if src_file.exists():
                with open(src_file, 'r') as f:
                    lines = f.readlines()[:n_samples]
                
                with open(dst_file, 'w') as f:
                    f.writelines(lines)
                
                print(f"✓ Sampled {split}.{ext}: {len(lines)} lines")
    
    print(f"\n✓ Small dataset created at {dst_dir}\n")

def main():
    parser = argparse.ArgumentParser(description='Quick test experiment')
    parser.add_argument('--exp-name', type=str, default='quick_test',
                       help='Experiment name')
    parser.add_argument('--base-dir', type=str, 
                       default='/mnt/data1/zhaojunzhang/experiments/quick_test',
                       help='Base directory')
    parser.add_argument('--data-dir', type=str,
                       default='/mnt/data1/zhaojunzhang/typilus-data/transformer',
                       help='Original data directory')
    parser.add_argument('--small-data-dir', type=str,
                       default='/mnt/data1/zhaojunzhang/typilus-data/transformer_small',
                       help='Small dataset directory')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of samples for quick test')
    parser.add_argument('--skip-sampling', action='store_true',
                       help='Skip data sampling (use existing small dataset)')
    parser.add_argument('--skip-eval', action='store_true',
                       help='Skip evaluation after training')
    
    args = parser.parse_args()
    
    # 1. 创建小数据集
    if not args.skip_sampling:
        sample_data(args.data_dir, args.small_data_dir, args.n_samples)
    else:
        print(f"Using existing small dataset: {args.small_data_dir}\n")
    
    # 2. 使用最小配置快速测试（使用小数据集）
    cmd = [
        'python', 'run_transformer_experiment.py',
        '--exp-name', args.exp_name,
        '--base-dir', args.base_dir,
        '--data-dir', args.small_data_dir,  # 使用小数据集
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
    
    # 如果跳过评估，添加参数
    if args.skip_eval:
        cmd.append('--skip-eval')
    
    print(f"{'='*60}")
    print(f"Quick Test Experiment")
    print(f"{'='*60}")
    print(f"Experiment: {args.exp_name}")
    print(f"Base dir: {args.base_dir}")
    print(f"Data dir: {args.small_data_dir} ({args.n_samples} samples)")
    print(f"Config: LSTM, 1 layer, 128 dim, 2 epochs")
    print(f"Steps: Train -> Validate -> Test")
    print(f"{'='*60}\n")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"\n{'='*60}")
        print(f"✓ Quick test completed successfully!")
        print(f"✓ Training pipeline is working")
        print(f"\nResults saved to: {args.base_dir}/{args.exp_name}/")
        print(f"- Checkpoints: checkpoints/")
        print(f"- Logs: logs/train.log, logs/eval.log")
        print(f"- Results: results/")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"✗ Quick test failed (exit code {result.returncode})")
        print(f"✗ Check logs: {args.base_dir}/{args.exp_name}/logs/")
        print(f"{'='*60}\n")
    
    sys.exit(result.returncode)

if __name__ == '__main__':
    main()
