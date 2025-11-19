#!/usr/bin/env python3
"""
将Typilus的词典格式转换为NaturalCC Transformer所需格式

Typilus格式 (每行): ["token", id]
NaturalCC格式 (每行): ["token", count]  (JSON格式)
"""
import json
import argparse
from collections import Counter

def convert_dict(input_file, output_file, default_count=1):
    """转换词典格式"""
    print(f"Reading from {input_file}")
    
    tokens = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                token, idx = json.loads(line)
                tokens.append((token, idx))
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} failed to parse: {line[:50]}... Error: {e}")
                continue
    
    # 按ID排序（保持原有顺序）
    tokens.sort(key=lambda x: x[1])
    
    print(f"Writing {len(tokens)} tokens to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # NaturalCC词典格式：每行是JSON数组 ["token", count]
        for token, _ in tokens:
            # 写入JSON格式：["token", count]
            json_line = json.dumps([token, default_count], ensure_ascii=False)
            f.write(json_line + '\n')
    
    print("Conversion completed!")
    print(f"Dictionary saved in NaturalCC JSON format")
    print(f"Format: each line is a JSON array [\"token\", count]")

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
