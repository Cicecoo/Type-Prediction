#!/usr/bin/env python3
"""
详细的测试评估脚本

功能：
1. 加载checkpoint进行预测
2. 计算各种指标（accuracy, precision, recall, F1）
3. 按类型统计准确率
4. 生成详细的评估报告
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm


def load_predictions(pred_file):
    """加载预测结果"""
    predictions = []
    with open(pred_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(line.split())
    return predictions


def load_references(ref_file):
    """加载真实标签"""
    references = []
    with open(ref_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                references.append(line.split())
    return references


def calculate_token_accuracy(predictions, references):
    """计算token级别的准确率"""
    total_tokens = 0
    correct_tokens = 0
    
    for pred, ref in zip(predictions, references):
        # 跳过<s>和</s>
        pred_clean = [p for p in pred if p not in ['<s>', '</s>', 'O']]
        ref_clean = [r for r in ref if r not in ['<s>', '</s>', 'O']]
        
        min_len = min(len(pred_clean), len(ref_clean))
        for i in range(min_len):
            total_tokens += 1
            if pred_clean[i] == ref_clean[i]:
                correct_tokens += 1
    
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    return accuracy, correct_tokens, total_tokens


def calculate_type_statistics(predictions, references):
    """统计各类型的准确率"""
    type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred, ref in zip(predictions, references):
        pred_clean = [p for p in pred if p not in ['<s>', '</s>']]
        ref_clean = [r for r in ref if r not in ['<s>', '</s>']]
        
        min_len = min(len(pred_clean), len(ref_clean))
        for i in range(min_len):
            ref_type = ref_clean[i]
            if ref_type != 'O':  # 只统计有类型标注的
                type_stats[ref_type]['total'] += 1
                if pred_clean[i] == ref_type:
                    type_stats[ref_type]['correct'] += 1
    
    return type_stats


def calculate_sequence_accuracy(predictions, references):
    """计算序列级别的准确率（完全匹配）"""
    total_seqs = len(predictions)
    correct_seqs = 0
    
    for pred, ref in zip(predictions, references):
        if pred == ref:
            correct_seqs += 1
    
    accuracy = correct_seqs / total_seqs if total_seqs > 0 else 0
    return accuracy, correct_seqs, total_seqs


def calculate_metrics(predictions, references):
    """计算precision, recall, F1"""
    tp = 0  # 预测为类型且正确
    fp = 0  # 预测为类型但错误
    fn = 0  # 应该预测为类型但预测为O
    
    for pred, ref in zip(predictions, references):
        pred_clean = [p for p in pred if p not in ['<s>', '</s>']]
        ref_clean = [r for r in ref if r not in ['<s>', '</s>']]
        
        min_len = min(len(pred_clean), len(ref_clean))
        for i in range(min_len):
            pred_type = pred_clean[i]
            ref_type = ref_clean[i]
            
            if ref_type != 'O':
                if pred_type == ref_type:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred_type != 'O':
                    fp += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def generate_report(results, output_file):
    """生成详细的评估报告"""
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Type Prediction Evaluation Report\n")
        f.write("="*80 + "\n\n")
        
        # 总体指标
        f.write("Overall Metrics:\n")
        f.write("-"*80 + "\n")
        f.write(f"Token Accuracy:    {results['token_accuracy']:.4f} "
                f"({results['correct_tokens']}/{results['total_tokens']})\n")
        f.write(f"Sequence Accuracy: {results['sequence_accuracy']:.4f} "
                f"({results['correct_sequences']}/{results['total_sequences']})\n")
        f.write(f"Precision:         {results['precision']:.4f}\n")
        f.write(f"Recall:            {results['recall']:.4f}\n")
        f.write(f"F1 Score:          {results['f1']:.4f}\n")
        f.write("\n")
        
        # 混淆矩阵统计
        f.write("Confusion Matrix Statistics:\n")
        f.write("-"*80 + "\n")
        f.write(f"True Positives:    {results['tp']}\n")
        f.write(f"False Positives:   {results['fp']}\n")
        f.write(f"False Negatives:   {results['fn']}\n")
        f.write("\n")
        
        # 按类型统计
        if results['type_stats']:
            f.write("Per-Type Statistics:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Type':<30} {'Correct':<10} {'Total':<10} {'Accuracy':<10}\n")
            f.write("-"*80 + "\n")
            
            # 按总数排序
            sorted_types = sorted(
                results['type_stats'].items(),
                key=lambda x: x[1]['total'],
                reverse=True
            )
            
            for type_name, stats in sorted_types[:50]:  # 显示前50个
                acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                f.write(f"{type_name:<30} {stats['correct']:<10} "
                       f"{stats['total']:<10} {acc:.4f}\n")
            
            if len(sorted_types) > 50:
                f.write(f"\n... and {len(sorted_types) - 50} more types\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate type prediction results')
    parser.add_argument('--pred-file', type=str, required=True,
                       help='Prediction file')
    parser.add_argument('--ref-file', type=str, required=True,
                       help='Reference (ground truth) file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("Loading predictions and references...")
    predictions = load_predictions(args.pred_file)
    references = load_references(args.ref_file)
    
    print(f"Loaded {len(predictions)} predictions and {len(references)} references")
    
    if len(predictions) != len(references):
        print(f"Warning: Number of predictions ({len(predictions)}) "
              f"!= number of references ({len(references)})")
    
    print("\nCalculating metrics...")
    
    # 计算各种指标
    token_acc, correct_tokens, total_tokens = calculate_token_accuracy(predictions, references)
    seq_acc, correct_seqs, total_seqs = calculate_sequence_accuracy(predictions, references)
    metrics = calculate_metrics(predictions, references)
    type_stats = calculate_type_statistics(predictions, references)
    
    # 汇总结果
    results = {
        'token_accuracy': token_acc,
        'correct_tokens': correct_tokens,
        'total_tokens': total_tokens,
        'sequence_accuracy': seq_acc,
        'correct_sequences': correct_seqs,
        'total_sequences': total_seqs,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'tp': metrics['tp'],
        'fp': metrics['fp'],
        'fn': metrics['fn'],
        'type_stats': type_stats,
    }
    
    # 打印结果
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    print(f"Token Accuracy:    {token_acc:.4f}")
    print(f"Sequence Accuracy: {seq_acc:.4f}")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"F1 Score:          {metrics['f1']:.4f}")
    print("="*80)
    
    # 生成报告
    os.makedirs(args.output_dir, exist_ok=True)
    report_file = os.path.join(args.output_dir, 'evaluation_report.txt')
    generate_report(results, report_file)
    
    # 保存JSON格式的结果
    json_file = os.path.join(args.output_dir, 'metrics.json')
    json_results = {k: v for k, v in results.items() if k != 'type_stats'}
    json_results['type_stats'] = {k: v for k, v in type_stats.items()}
    
    with open(json_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"✓ Metrics saved to: {json_file}")


if __name__ == '__main__':
    main()
