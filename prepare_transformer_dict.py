#!/usr/bin/env python3
"""
将Typilus的词典格式转换为NaturalCC Transformer所需格式

Typilus格式 (每行): ["token", id]
NaturalCC格式 (每行): ["token", count]  (JSON格式)

注意：需要同时处理nodes.dict.json(代码token)和supernodes.dict.json(类型标签)
"""
import json
import argparse
from collections import Counter
import os

def convert_dict(input_file, output_file, default_count=1):
    """转换词典格式"""
    print(f"Reading from {input_file}")
    
    # NaturalCC的Dictionary会自动添加这些特殊token：[PAD], <s>, </s>, [UNK]
    # 原始Typilus词典中有 [PAD] 和 [UNK]，需要过滤掉避免重复
    # 但 <s> 和 </s> 不在原始词典中，会被Dictionary自动添加
    skip_tokens = {'[PAD]', '[UNK]'}
    
    token_dict = {}  # 使用字典去重，key=token, value=原始ID
    skipped_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                token, idx = json.loads(line)
                
                # 跳过会被Dictionary自动添加的特殊token
                if token in skip_tokens:
                    skipped_count += 1
                    print(f"Skipping special token: {token}")
                    continue
                
                # 去重：只保留第一次出现的token
                if token not in token_dict:
                    token_dict[token] = idx
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} failed to parse: {line[:50]}... Error: {e}")
                continue
    
    # 按ID排序（保持原有顺序）
    tokens = sorted(token_dict.items(), key=lambda x: x[1])
    
    print(f"Found {len(tokens)} unique tokens from original dictionary")
    print(f"Skipped {skipped_count} special tokens ([PAD], [UNK]) - will be auto-added by Dictionary")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # NaturalCC词典格式：每行是JSON数组 ["token", count]
        for token, _ in tokens:
            # 写入JSON格式：["token", count]
            json_line = json.dumps([token, default_count], ensure_ascii=False)
            f.write(json_line + '\n')
    
    print("Conversion completed!")
    print(f"Dictionary saved in NaturalCC JSON format")
    print(f"Format: each line is a JSON array [\"token\", count]")
    
    # 打印前几个token作为示例
    print("\nFirst 5 tokens:")
    for i, (token, _) in enumerate(tokens[:5]):
        print(f"  {i+1}. {token}")
    
    # 打印前几个token作为示例
    print("\nFirst 5 tokens:")
    for i, (token, _) in enumerate(tokens[:5]):
        print(f"  {i+1}. {token}")

def append_type_labels(types_dict_file, output_file, default_count=1):
    """追加类型标签到词典"""
    # 首先读取已有的token
    existing_tokens = set()
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                token, count = json.loads(line.strip())
                existing_tokens.add(token)
            except:
                continue
    
    # 读取类型标签
    skip_tokens = {'[PAD]', '[UNK]'}
    type_labels = []
    skipped = 0
    
    with open(types_dict_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                token, idx = json.loads(line)
                
                if token in skip_tokens:
                    skipped += 1
                    continue
                
                # 只添加不存在的类型标签
                if token not in existing_tokens:
                    type_labels.append((token, idx))
                    existing_tokens.add(token)
            except:
                continue
    
    # 按ID排序
    type_labels.sort(key=lambda x: x[1])
    
    print(f"Found {len(type_labels)} new type labels")
    print(f"Skipped {skipped} special tokens from type dict")
    
    # 追加到文件
    with open(output_file, 'a', encoding='utf-8') as f:
        for token, _ in type_labels:
            json_line = json.dumps([token, default_count], ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"Appended type labels. Total vocabulary size: {len(existing_tokens)}")
    
    # 打印前几个类型标签
    print("\nFirst 5 type labels:")
    for i, (token, _) in enumerate(type_labels[:5]):
        print(f"  {i+1}. {token}")

def main():
    parser = argparse.ArgumentParser(description='Convert Typilus dict to Transformer format')
    parser.add_argument('--input', type=str, required=True,
                       help='Input Typilus dict file (nodes.dict.json)')
    parser.add_argument('--types-dict', type=str, default=None,
                       help='Optional: supernodes.dict.json for type labels')
    parser.add_argument('--output', type=str, required=True,
                       help='Output Transformer dict file')
    parser.add_argument('--count', type=int, default=1,
                       help='Default count value for each token (default: 1)')
    
    args = parser.parse_args()
    
    # 如果没有指定types_dict，尝试自动查找
    if args.types_dict is None:
        input_dir = os.path.dirname(args.input)
        potential_types_dict = os.path.join(input_dir, 'supernodes.dict.json')
        if os.path.exists(potential_types_dict):
            print(f"Auto-detected types dictionary: {potential_types_dict}")
            args.types_dict = potential_types_dict
    
    # 首先处理代码token
    convert_dict(args.input, args.output, args.count)
    
    # 如果有类型词典，追加类型标签
    if args.types_dict and os.path.exists(args.types_dict):
        print(f"\nAppending type labels from {args.types_dict}")
        append_type_labels(args.types_dict, args.output, args.count)

if __name__ == '__main__':
    main()
