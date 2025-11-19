#!/usr/bin/env python3
"""
创建一个虚拟的SentencePiece模型

虽然我们的数据不需要SentencePiece，但NaturalCC的代码会尝试加载它。
我们创建一个基本的模型文件以满足加载要求。
"""

import os
import argparse
import sentencepiece as spm


def create_dummy_sentencepiece_model(output_prefix, vocab_file=None):
    """创建一个虚拟的SentencePiece模型"""
    
    # 创建一个临时的训练文本文件
    temp_txt = output_prefix + '_temp.txt'
    
    # 如果有词典文件，从中提取token
    if vocab_file and os.path.exists(vocab_file):
        import json
        print(f"Reading vocabulary from {vocab_file}")
        tokens = []
        with open(vocab_file, 'r') as f:
            for line in f:
                try:
                    token, count = json.loads(line.strip())
                    tokens.append(token)
                except:
                    continue
        
        # 生成一些示例文本（使用vocabulary中的token）
        with open(temp_txt, 'w') as f:
            for i in range(0, len(tokens), 10):
                sample = ' '.join(tokens[i:i+10])
                f.write(sample + '\n')
    else:
        # 创建一个简单的示例文本
        print("Creating dummy training text")
        with open(temp_txt, 'w') as f:
            for i in range(100):
                f.write(f"dummy text line {i}\n")
    
    # 训练SentencePiece模型
    print(f"Training SentencePiece model: {output_prefix}.model")
    spm.SentencePieceTrainer.train(
        input=temp_txt,
        model_prefix=output_prefix,
        vocab_size=10000,
        model_type='unigram',
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )
    
    # 删除临时文件
    if os.path.exists(temp_txt):
        os.remove(temp_txt)
    
    print(f"✓ Created {output_prefix}.model and {output_prefix}.vocab")
    print(f"  Note: This is a dummy model for compatibility only")


def main():
    parser = argparse.ArgumentParser(description='Create dummy SentencePiece model')
    parser.add_argument('--output', type=str, required=True,
                       help='Output prefix (e.g., /path/to/model)')
    parser.add_argument('--vocab-file', type=str,
                       help='Optional: vocabulary file to use for training')
    
    args = parser.parse_args()
    
    create_dummy_sentencepiece_model(args.output, args.vocab_file)


if __name__ == '__main__':
    main()
