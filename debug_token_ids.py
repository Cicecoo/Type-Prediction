#!/usr/bin/env python3
"""
调试token ID，检查是否有超出词典范围的ID
"""
import json
import sys

def load_dict(dict_file):
    """加载词典并返回token到ID的映射"""
    token_to_id = {}
    id_to_token = {}
    
    # Dictionary会自动添加特殊token在开头
    special_tokens = ['[PAD]', '<s>', '</s>', '[UNK]']
    for idx, token in enumerate(special_tokens):
        token_to_id[token] = idx
        id_to_token[idx] = token
    
    next_id = len(special_tokens)
    
    with open(dict_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                token, count = json.loads(line)
                if token not in token_to_id:  # 避免重复
                    token_to_id[token] = next_id
                    id_to_token[next_id] = token
                    next_id += 1
            except:
                continue
    
    return token_to_id, id_to_token, next_id

def check_token_ids(data_file, token_to_id, vocab_size, max_lines=1000):
    """检查数据文件中的token ID"""
    token_mapping = {
        '<unk>': '[UNK]',
    }
    
    max_id_seen = -1
    out_of_range_tokens = {}
    total_tokens = 0
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num > max_lines:
                break
                
            line = line.strip()
            if not line:
                continue
            
            tokens = line.split()
            
            for pos, token in enumerate(tokens):
                total_tokens += 1
                
                # 应用映射
                mapped_token = token_mapping.get(token, token)
                
                if mapped_token in token_to_id:
                    token_id = token_to_id[mapped_token]
                else:
                    # UNK token
                    token_id = token_to_id['[UNK]']
                
                max_id_seen = max(max_id_seen, token_id)
                
                if token_id >= vocab_size:
                    key = f"{token} -> {mapped_token} (ID={token_id})"
                    if key not in out_of_range_tokens:
                        out_of_range_tokens[key] = {
                            'count': 0,
                            'first_line': line_num,
                            'first_pos': pos
                        }
                    out_of_range_tokens[key]['count'] += 1
    
    return max_id_seen, out_of_range_tokens, total_tokens

def main():
    if len(sys.argv) < 3:
        print("Usage: python debug_token_ids.py <dict_file> <data_file> [max_lines]")
        sys.exit(1)
    
    dict_file = sys.argv[1]
    data_file = sys.argv[2]
    max_lines = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    
    print(f"Loading dictionary from {dict_file}")
    token_to_id, id_to_token, vocab_size = load_dict(dict_file)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Special tokens: [PAD]=0, <s>=1, </s>=2, [UNK]=3")
    
    print(f"\nFirst 10 tokens from dict:")
    for i in range(min(10, vocab_size)):
        if i in id_to_token:
            print(f"  {i}: {id_to_token[i]}")
    
    print(f"\nChecking {data_file} (first {max_lines} lines)")
    max_id, out_of_range, total = check_token_ids(data_file, token_to_id, vocab_size, max_lines)
    
    print(f"\nTotal tokens checked: {total}")
    print(f"Max token ID seen: {max_id}")
    print(f"Vocabulary size: {vocab_size}")
    
    if out_of_range:
        print(f"\n❌ ERROR: Found {len(out_of_range)} token types with IDs >= vocab_size")
        print("\nOut-of-range tokens:")
        for token_info, details in sorted(out_of_range.items(), key=lambda x: -x[1]['count'])[:20]:
            print(f"  {token_info}")
            print(f"    Count: {details['count']}, First at line {details['first_line']}, pos {details['first_pos']}")
    else:
        print(f"\n✓ All token IDs are within range [0, {vocab_size-1}]")

if __name__ == '__main__':
    main()
