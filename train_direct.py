#!/usr/bin/env python3
"""
直接调用NaturalCC训练（不依赖ncc-train命令）
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# 添加ncc到Python路径
ncc_path = Path(__file__).parent.absolute()
if str(ncc_path) not in sys.path:
    sys.path.insert(0, str(ncc_path))

try:
    from ncc.trainers import Trainer
    from ncc.cli.train import cli_main
except ImportError as e:
    print(f"Error: Cannot import NaturalCC modules: {e}")
    print("Make sure you are in the naturalcc directory and the package is properly installed")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Train transformer model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    print(f"Loading config from: {args.config}")
    
    # 构建命令行参数
    sys.argv = ['train', '--configs', args.config]
    
    # 调用NaturalCC的训练函数
    try:
        cli_main()
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
