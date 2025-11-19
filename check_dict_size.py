#!/usr/bin/env python3
"""
检查Dictionary实际加载后的大小
"""
import sys
sys.path.insert(0, '/home/zhaojunzhang/workspace/type_pred/naturalcc')

from ncc.data.dictionary import Dictionary

dict_file = '/mnt/data1/zhaojunzhang/typilus-data/transformer/dict.txt'

print(f"Loading dictionary from {dict_file}")
dictionary = Dictionary()
dictionary.add_from_file(dict_file)

print(f"\nDictionary statistics:")
print(f"  len(dictionary) = {len(dictionary)}")
print(f"  len(dictionary.symbols) = {len(dictionary.symbols)}")
print(f"  dictionary.nspecial = {dictionary.nspecial}")
print(f"  dictionary.pad() = {dictionary.pad()}")
print(f"  dictionary.bos() = {dictionary.bos()}")
print(f"  dictionary.eos() = {dictionary.eos()}")
print(f"  dictionary.unk() = {dictionary.unk()}")

print(f"\nFirst 10 symbols:")
for i in range(min(10, len(dictionary))):
    print(f"  {i}: {dictionary[i]}")

print(f"\nLast 5 symbols:")
for i in range(max(0, len(dictionary)-5), len(dictionary)):
    print(f"  {i}: {dictionary[i]}")

# 检查文件中有多少行
import json
line_count = 0
with open(dict_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            line_count += 1

print(f"\nDict file has {line_count} lines")
print(f"Dictionary has {len(dictionary)} symbols")
print(f"Difference: {len(dictionary) - line_count} (should be {dictionary.nspecial} special tokens)")
