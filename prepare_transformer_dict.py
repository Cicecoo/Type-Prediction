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
    
    # 添加特殊token（如果不存在）
    special_tokens = ['<pad>', '<unk>', '<s>', '</s>']
    existing_tokens = {t for t, _ in tokens}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 先写入特殊token
        for special_token in special_tokens:
            if special_token not in existing_tokens and special_token.lower() not in existing_tokens:
                f.write(f"{special_token} {default_count}\n")
        
        # 写入所有token
        for token, _ in tokens:
            # 确保token不包含换行符，并且正确转义
            token_clean = token.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            
            # 检查token是否包含空格
            if ' ' in token_clean:
                # 如果包含空格，使用引号包裹或跳过
                print(f"Warning: Token contains space, skipping: '{token_clean[:50]}'")
                continue
            
            f.write(f"{token_clean} {default_count}\n")
    
    print("Conversion completed!")
    print(f"Note: Check first few lines of {output_file} to verify format")

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
