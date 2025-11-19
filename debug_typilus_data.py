#!/usr/bin/env python3
"""
调试Typilus数据格式，查看token-sequence和nodes的对应关系
"""

import json
import sys
import os

def debug_data_format(data_dir, split='train', num_samples=5):
    """检查前几个样本的数据格式"""
    
    # 首先检查目录结构
    print(f"Data directory: {data_dir}")
    print(f"Checking directory structure...")
    
    # 可能的路径
    possible_paths = [
        f"{data_dir}/attributes/{split}.token-sequence",
        f"{data_dir}/{split}.token-sequence",
        f"{data_dir}/data/{split}.token-sequence",
    ]
    
    token_seq_file = None
    for path in possible_paths:
        if os.path.exists(path):
            token_seq_file = path
            base_dir = os.path.dirname(path)
            break
    
    if token_seq_file is None:
        print(f"ERROR: Could not find token-sequence file!")
        print(f"Tried paths:")
        for path in possible_paths:
            print(f"  - {path}")
        
        # 列出实际存在的目录
        print(f"\nActual directory contents:")
        if os.path.exists(data_dir):
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                if os.path.isdir(item_path):
                    print(f"  DIR: {item}/")
                    # 列出子目录内容
                    try:
                        sub_items = os.listdir(item_path)[:10]
                        for sub in sub_items:
                            print(f"    - {sub}")
                        if len(os.listdir(item_path)) > 10:
                            print(f"    ... and {len(os.listdir(item_path)) - 10} more")
                    except:
                        pass
                else:
                    print(f"  FILE: {item}")
        return
    
    nodes_file = os.path.join(base_dir, f"{split}.nodes")
    supernodes_file = os.path.join(base_dir, f"{split}.supernodes")
    
    print(f"Found files:")
    print(f"  - {token_seq_file}")
    print(f"  - {nodes_file}")
    print(f"  - {supernodes_file}")
    print(f"=" * 80)
    
    with open(token_seq_file, 'r') as f_seq, \
         open(nodes_file, 'r') as f_nodes, \
         open(supernodes_file, 'r') as f_super:
        
        for i in range(num_samples):
            try:
                seq_line = f_seq.readline().strip()
                nodes_line = f_nodes.readline().strip()
                super_line = f_super.readline().strip()
                
                token_ids = json.loads(seq_line)
                nodes = json.loads(nodes_line)
                supernodes = json.loads(super_line)
                
                print(f"\nSample {i}:")
                print(f"  token_ids length: {len(token_ids)}")
                print(f"  nodes length: {len(nodes) if isinstance(nodes, list) else 'N/A'}")
                print(f"  supernodes count: {len(supernodes) if isinstance(supernodes, dict) else 0}")
                
                print(f"  token_ids[:10]: {token_ids[:10]}")
                print(f"  nodes[:10]: {nodes[:10] if isinstance(nodes, list) else nodes}")
                
                if isinstance(supernodes, dict) and supernodes:
                    print(f"  supernode keys (first 10): {list(supernodes.keys())[:10]}")
                    first_key = list(supernodes.keys())[0]
                    print(f"  supernode[{first_key}]: {supernodes[first_key]}")
                
            except Exception as e:
                print(f"Error on sample {i}: {e}")
                break
    
    print(f"\n" + "=" * 80)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "/mnt/data1/zhaojunzhang/typilus-data"
    
    debug_data_format(data_dir, 'train', 5)
