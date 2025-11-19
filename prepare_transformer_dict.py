#!/usr/bin/env python3
"""
将Typilus的词典格式转换为NaturalCC Transformer所需格式

Typilus格式 (每行): ["token", id]
Transformer格式: token count (空格分隔，每行一个token)
"""
import json
import argparse
from collections import Counter

def convert_dict(input_file, output_file, default_count=1):
    """转换词典格式"""
    print(f"Reading from {input_file}")
    
    tokens = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                token, idx = json.loads(line)
                tokens.append((token, idx))
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {line[:50]}... Error: {e}")
                continue
    
    # 按ID排序（保持原有顺序）
    tokens.sort(key=lambda x: x[1])
    
    print(f"Writing {len(tokens)} tokens to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for token, _ in tokens:
            # Transformer词典格式: token count
            # 这里使用固定的count值，因为我们没有真实的词频统计
            f.write(f"{token} {default_count}\n")
    
    print("Conversion completed!")

def main():
    parser = argparse.ArgumentParser(description='Convert Typilus dict to Transformer format')
    parser.add_argument('--input', type=str, required=True,
                       help='Input Typilus dict file (nodes.dict.json)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output Transformer dict file')
    parser.add_argument('--count', type=int, default=1,
                       help='Default count value for each token (default: 1)')
    
    args = parser.parse_args()
    convert_dict(args.input, args.output, args.count)

if __name__ == '__main__':
    main()
