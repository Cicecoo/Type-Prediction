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
    
    # 注意：虽然Dictionary会自动添加特殊token，但我们的数据中已经使用了这些token
    # 所以需要在词典中包含它们，避免索引错误
    
    token_dict = {}  # 使用字典去重，key=token, value=原始ID
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                token, idx = json.loads(line)
                
                # 去重：只保留第一次出现的token
                if token not in token_dict:
                    token_dict[token] = idx
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} failed to parse: {line[:50]}... Error: {e}")
                continue
    
    # 按ID排序（保持原有顺序）
    tokens = sorted(token_dict.items(), key=lambda x: x[1])
    
    print(f"Found {len(tokens)} unique tokens from original dictionary")
    
    # 手动添加特殊token（如果不存在的话）
    special_tokens = ['<pad>', '<unk>', '<s>', '</s>']
    added_specials = []
    for special_token in special_tokens:
        if special_token not in token_dict:
            tokens.append((special_token, -1))  # 使用-1作为ID标记
            added_specials.append(special_token)
    
    if added_specials:
        print(f"Added {len(added_specials)} special tokens: {added_specials}")
    
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
