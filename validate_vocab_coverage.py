#!/usr/bin/env python3
"""
验证训练数据中的所有token都在词典中
"""
import json
import argparse
from collections import Counter

def load_dict(dict_file):
    """加载词典"""
    vocab = set()
    with open(dict_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                token, count = json.loads(line)
                vocab.add(token)
            except:
                continue
    return vocab

def check_file(data_file, vocab, max_lines=10000):
    """检查数据文件中的token覆盖"""
    missing_tokens = Counter()
    total_tokens = 0
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            
            line = line.strip()
            if not line:
                continue
            
            # 训练数据格式: <s> token1 token2 ... </s>
            tokens = line.split()
            
            for token in tokens:
                total_tokens += 1
                if token not in vocab:
                    missing_tokens[token] += 1
    
    return missing_tokens, total_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict', type=str, required=True, help='Dictionary file')
    parser.add_argument('--data', type=str, required=True, help='Data file to check')
    parser.add_argument('--max-lines', type=int, default=10000, help='Max lines to check')
    
    args = parser.parse_args()
    
    print(f"Loading dictionary from {args.dict}")
    vocab = load_dict(args.dict)
    
    # Dictionary会自动添加这些特殊token
    special_tokens = ['[PAD]', '<s>', '</s>', '[UNK]']
    for tok in special_tokens:
        vocab.add(tok)
    
    print(f"Dictionary has {len(vocab)} tokens (including auto-added special tokens)")
    print(f"Special tokens: {special_tokens}")
    
    print(f"\nChecking {args.data} (first {args.max_lines} lines)")
    missing, total = check_file(args.data, vocab, args.max_lines)
    
    print(f"\nTotal tokens checked: {total}")
    print(f"Missing tokens: {len(missing)}")
    
    if missing:
        print("\nTop 20 missing tokens:")
        for token, count in missing.most_common(20):
            print(f"  '{token}': {count} occurrences")
    else:
        print("\n✓ All tokens are in vocabulary!")

if __name__ == '__main__':
    main()
