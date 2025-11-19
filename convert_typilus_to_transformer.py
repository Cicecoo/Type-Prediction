#!/usr/bin/env python3
"""
将Typilus图数据转换为Transformer序列数据格式

Typilus格式：
- token-sequence: [17,20,22,...] (token ID序列)
- nodes: {"0": ["def"], "1": ["function_name"], ...} (节点token)
- supernodes: {"0": {"annotation": "int", ...}, ...} (类型标注)

Transformer格式：
- train.code: <s> token1 token2 ... </s>
- train.type: O O int O ...
"""

import os
import json
import gzip
import argparse
from tqdm import tqdm
from pathlib import Path


def load_vocab_dict(dict_file):
    """加载词典：token -> id
    
    词典格式：每行一个 [token, id] 的JSON数组
    例如：["[PAD]", 2147483647]
    """
    vocab = {}
    id2token = {}
    
    with open(dict_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                token, idx = json.loads(line)
                vocab[token] = idx
                id2token[idx] = token
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {line[:50]}... Error: {e}")
                continue
    
    return vocab, id2token


def convert_graph_to_sequence(graph_file, nodes_dict, output_code, output_type):
    """
    从graph jsonl.gz文件提取序列数据
    
    Args:
        graph_file: graph-XXX.jsonl.gz文件路径
        nodes_dict: 节点ID到token的映射
        output_code: 输出代码文件
        output_type: 输出类型文件
    """
    with gzip.open(graph_file, 'rt', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            # 获取token序列
            token_seq = data.get('token-sequence', [])
            if not token_seq:
                continue
                
            # 将token ID转换为token字符串
            tokens = []
            for tid in token_seq:
                token = nodes_dict.get(tid, '<unk>')
                tokens.append(token)
            
            # 获取类型标注
            supernodes = data.get('supernodes', {})
            
            # 初始化类型标签（全部为O）
            types = ['O'] * len(tokens)
            
            # 填充类型标注
            for node_id, node_info in supernodes.items():
                annotation = node_info.get('annotation', 'O')
                # 去掉泛型信息，只保留基本类型
                if '[' in annotation:
                    annotation = annotation.split('[')[0]
                    
                # 找到这个supernode对应的token位置
                # 这里需要根据nodes中的映射关系来确定
                # 简化处理：如果有位置信息就用，否则跳过
                if 'node_ids' in node_info:  # 如果有节点ID列表
                    node_ids = node_info['node_ids']
                    for nid in node_ids:
                        if nid < len(types):
                            types[nid] = annotation
            
            # 添加<s>和</s>标记
            tokens_str = '<s> ' + ' '.join(tokens) + ' </s>'
            types_str = 'O ' + ' '.join(types) + ' O'
            
            # 写入文件
            output_code.write(tokens_str + '\n')
            output_type.write(types_str + '\n')


def convert_attributes_to_transformer(attributes_dir, dict_file, output_dir, split='train'):
    """
    从attributes目录转换数据
    
    这个方法更简单，直接使用已经flatten的数据
    """
    print(f"Converting {split} split from attributes...")
    
    # 加载词典（如果存在）
    print("Loading vocabulary...")
    if os.path.exists(dict_file):
        vocab, id2token = load_vocab_dict(dict_file)
        print(f"Loaded {len(vocab)} tokens from vocabulary")
    else:
        print(f"Warning: Dictionary file not found: {dict_file}")
        print("Will use node strings directly from attributes/nodes files")
        vocab, id2token = {}, {}
    
    # 读取数据文件
    token_seq_file = os.path.join(attributes_dir, f'{split}.token-sequence')
    nodes_file = os.path.join(attributes_dir, f'{split}.nodes')
    supernodes_file = os.path.join(attributes_dir, f'{split}.supernodes')
    
    output_code = os.path.join(output_dir, f'{split}.code')
    output_type = os.path.join(output_dir, f'{split}.type')
    
    print(f"Reading from {token_seq_file}, {nodes_file}, {supernodes_file}")
    print(f"Writing to {output_code}, {output_type}")
    
    with open(token_seq_file, 'r') as f_seq, \
         open(nodes_file, 'r') as f_nodes, \
         open(supernodes_file, 'r') as f_super, \
         open(output_code, 'w') as f_code, \
         open(output_type, 'w') as f_type:
        
        lines_processed = 0
        for seq_line, nodes_line, super_line in tqdm(zip(f_seq, f_nodes, f_super), desc=f"Processing {split}"):
            try:
                # 解析token序列（ID列表）
                token_ids = json.loads(seq_line.strip())
                if not token_ids:
                    continue
                
                # 解析nodes（节点类型字符串数组）
                nodes = json.loads(nodes_line.strip())
                
                # 跳过空数据
                if nodes is None:
                    continue
                
                # 关键修复：使用token_ids作为基准，将ID转换为token字符串
                # 如果有vocab字典，使用字典转换；否则使用nodes数组（如果长度匹配）
                if vocab and id2token:
                    # 使用词典将token ID转换为字符串
                    tokens = []
                    for tid in token_ids:
                        if tid in id2token:
                            tokens.append(id2token[tid])
                        else:
                            tokens.append('<unk>')
                elif isinstance(nodes, list) and len(nodes) == len(token_ids):
                    # 如果nodes是列表且长度匹配，直接使用
                    tokens = nodes
                else:
                    # 长度不匹配或格式错误，跳过此样本
                    print(f"Warning: line {lines_processed} - token_ids length {len(token_ids)} != nodes length {len(nodes) if isinstance(nodes, list) else 'N/A'}, skipping")
                    continue
                
                # 解析supernodes（类型标注字典）
                supernodes = json.loads(super_line.strip())
                
                # 初始化类型序列（全部为O）- 长度必须与tokens一致
                types = ['O'] * len(tokens)
                
                # 先填充类型标注（在清理之前，使用原始索引）
                # supernodes格式: {"18": {"name": "T", "annotation": "str", ...}, ...}
                if supernodes:
                    for node_id, node_info in supernodes.items():
                        annotation = node_info.get('annotation')
                        
                        # 跳过null或None的标注
                        if annotation is None or annotation == 'null':
                            continue
                        
                        # 清理类型标注（去掉泛型参数）
                        if '[' in annotation:
                            annotation = annotation.split('[')[0]
                        
                        # node_id是字符串格式的索引（对应原始tokens数组）
                        try:
                            idx = int(node_id)
                            # 确保索引在有效范围内
                            if 0 <= idx < len(types):
                                types[idx] = annotation
                        except (ValueError, IndexError):
                            continue
                
                # 然后清理tokens和types：移除空字符串，替换所有空白字符
                cleaned_tokens = []
                cleaned_types = []
                for token, type_label in zip(tokens, types):
                    # 跳过空token
                    if not token or not token.strip():
                        continue
                    # 替换所有空白字符为下划线
                    token = ''.join('_' if c.isspace() else c for c in token)
                    # 如果清理后变成空字符串，跳过
                    if not token:
                        continue
                    cleaned_tokens.append(token)
                    cleaned_types.append(type_label)
                
                tokens = cleaned_tokens
                types = cleaned_types
                
                # 验证：确保 tokens 和 types 长度完全一致
                if len(tokens) != len(types):
                    print(f"Warning: line {lines_processed} - tokens length {len(tokens)} != types length {len(types)}, skipping")
                    continue
                
                # 格式化输出（添加<s>和</s>）
                # 关键：不使用 join，而是确保每个元素都正确分隔
                code_parts = ['<s>'] + tokens + ['</s>']
                type_parts = ['O'] + types + ['O']
                
                # 验证长度
                assert len(code_parts) == len(type_parts), \
                    f"Internal error at line {lines_processed}: {len(code_parts)} != {len(type_parts)}"
                
                # 生成输出行
                code_line = ' '.join(code_parts)
                type_line = ' '.join(type_parts)
                
                # 最终验证：确保split后长度一致
                code_tokens_count = len(code_line.split())
                type_tokens_count = len(type_line.split())
                if code_tokens_count != type_tokens_count:
                    print(f"ERROR: line {lines_processed} output mismatch - code={code_tokens_count}, type={type_tokens_count}")
                    print(f"  Original: tokens={len(tokens)}, types={len(types)}")
                    # 找出有问题的token
                    for i, (t, tp) in enumerate(zip(code_parts, type_parts)):
                        if ' ' in t or '\t' in t or '\n' in t:
                            print(f"  Problem token at {i}: '{t}' (has whitespace)")
                    continue
                
                f_code.write(code_line + '\n')
                f_type.write(type_line + '\n')
                
                lines_processed += 1
                
            except Exception as e:
                print(f"Error processing line {lines_processed}: {e}")
                continue
        
        print(f"Processed {lines_processed} examples for {split}")


def main():
    parser = argparse.ArgumentParser(description='Convert Typilus data to Transformer format')
    parser.add_argument('--typilus-dir', type=str, 
                       default='/mnt/data1/zhaojunzhang/typilus-data/typilus',
                       help='Typilus data directory')
    parser.add_argument('--output-dir', type=str,
                       default='/mnt/data1/zhaojunzhang/typilus-data/transformer',
                       help='Output directory for transformer data')
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'],
                       help='Data splits to convert')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # attributes目录 - 修正路径
    # Typilus数据可能在 typilus/attributes 或直接在 attributes
    if os.path.exists(os.path.join(args.typilus_dir, 'typilus', 'attributes')):
        attributes_dir = os.path.join(args.typilus_dir, 'typilus', 'attributes')
    elif os.path.exists(os.path.join(args.typilus_dir, 'attributes')):
        attributes_dir = os.path.join(args.typilus_dir, 'attributes')
    else:
        raise FileNotFoundError(f"Cannot find attributes directory in {args.typilus_dir}")
    
    print(f"Using attributes directory: {attributes_dir}")
    
    # 节点词典文件 - 尝试多个可能的位置
    dict_paths = [
        os.path.join(args.typilus_dir, 'typilus', 'type_inference', 'data-mmap', 'nodes.dict.json'),
        os.path.join(args.typilus_dir, 'type_inference', 'data-mmap', 'nodes.dict.json'),
        os.path.join(args.typilus_dir, 'nodes.dict.json'),
    ]
    
    dict_file = None
    for path in dict_paths:
        if os.path.exists(path):
            dict_file = path
            break
    
    if dict_file is None:
        print(f"Warning: Dictionary file not found in any of these locations:")
        for path in dict_paths:
            print(f"  - {path}")
        print("Will use node strings directly from attributes/nodes files")
    
    # 转换每个split
    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"Converting {split} split")
        print(f"{'='*60}")
        
        convert_attributes_to_transformer(
            attributes_dir=attributes_dir,
            dict_file=dict_file,
            output_dir=args.output_dir,
            split=split
        )
    
    print(f"\n{'='*60}")
    print("Conversion completed!")
    print(f"{'='*60}")
    print(f"\nOutput files in {args.output_dir}:")
    for split in args.splits:
        print(f"  - {split}.code")
        print(f"  - {split}.type")
    
    print("\nNext steps:")
    print("1. Copy dictionaries and SentencePiece model:")
    print(f"   cp {args.typilus_dir}/type_inference/data-mmap/nodes.dict.json {args.output_dir}/csnjs_8k_9995p_unigram_url.dict.txt")
    print(f"   # You may need to create a SentencePiece model or use identity tokenization")
    print("\n2. Update transformer config to point to this directory")


if __name__ == '__main__':
    main()
