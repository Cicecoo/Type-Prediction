#!/usr/bin/env python3
"""
检查训练数据加载器返回的实际token ID
"""
import sys
sys.path.insert(0, '/home/zhaojunzhang/workspace/type_pred/naturalcc')

import torch
from ncc.data.dictionary import Dictionary
from ncc.data.type_prediction import CodeTypeDataset

# 加载词典
dict_file = '/mnt/data1/zhaojunzhang/typilus-data/transformer/dict.txt'
print(f"Loading dictionaries...")
src_dict = Dictionary()
src_dict.add_from_file(dict_file)
tgt_dict = Dictionary()
tgt_dict.add_from_file(dict_file)

print(f"Source dictionary size: {len(src_dict)}")
print(f"Target dictionary size: {len(tgt_dict)}")

# 加载数据集
data_dir = '/mnt/data1/zhaojunzhang/typilus-data/transformer'
print(f"\nLoading dataset from {data_dir}")

dataset = CodeTypeDataset(
    src_file=f"{data_dir}/train.code",
    tgt_file=f"{data_dir}/train.type",
    src_dict=src_dict,
    tgt_dict=tgt_dict,
    sp=None,  # 预分词数据
    max_source_positions=512,
    max_target_positions=512,
)

print(f"Dataset size: {len(dataset)}")

# 检查前几个样本
print(f"\nChecking first 10 samples:")
for i in range(min(10, len(dataset))):
    sample = dataset[i]
    subword_ids = sample['subword_ids']
    label_segments = sample['label_segments']
    
    max_id = subword_ids.max().item() if len(subword_ids) > 0 else -1
    min_id = subword_ids.min().item() if len(subword_ids) > 0 else -1
    
    print(f"\nSample {i}:")
    print(f"  subword_ids shape: {subword_ids.shape}")
    print(f"  subword_ids range: [{min_id}, {max_id}]")
    print(f"  vocab size: {len(src_dict)}")
    
    if max_id >= len(src_dict):
        print(f"  ❌ ERROR: max_id {max_id} >= vocab_size {len(src_dict)}")
        print(f"  First 20 token IDs: {subword_ids[:20].tolist()}")
        print(f"  Out-of-range IDs: {subword_ids[subword_ids >= len(src_dict)].tolist()[:10]}")
        break
    elif min_id < 0:
        print(f"  ❌ ERROR: min_id {min_id} < 0")
        break
    else:
        print(f"  ✓ All IDs in valid range")
        print(f"  First 10 token IDs: {subword_ids[:10].tolist()}")

# 测试collater
print(f"\n\nTesting collater with batch_size=2:")
samples = [dataset[i] for i in range(2)]
batch = dataset.collater(samples)

print(f"Batch keys: {batch.keys()}")
print(f"src_tokens shape: {batch['src_tokens'].shape}")
print(f"src_tokens range: [{batch['src_tokens'].min().item()}, {batch['src_tokens'].max().item()}]")
print(f"src_lengths: {batch['src_lengths']}")

if batch['src_tokens'].max().item() >= len(src_dict):
    print(f"❌ ERROR in batch: max token ID {batch['src_tokens'].max().item()} >= vocab_size {len(src_dict)}")
else:
    print(f"✓ Batch token IDs are valid")
