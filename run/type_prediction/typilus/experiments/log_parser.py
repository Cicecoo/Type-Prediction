#!/usr/bin/env python3
"""
训练日志解析工具
从训练日志中提取关键信息并保存为结构化数据
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime


class TrainingLogParser:
    """训练日志解析器"""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.data = {
            'metadata': {},
            'epochs': [],
            'training_steps': [],
            'validation_steps': [],
            'final_results': {}
        }
    
    def parse(self) -> Dict:
        """解析整个日志文件"""
        if not self.log_file.exists():
            print(f"日志文件不存在: {self.log_file}")
            return self.data
        
        print(f"解析日志文件: {self.log_file}")
        
        with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        self._extract_metadata(lines)
        self._extract_training_steps(lines)
        self._extract_epochs(lines)
        self._extract_final_results(lines)
        
        return self.data
    
    def _extract_metadata(self, lines: List[str]):
        """提取元数据（配置信息、GPU信息等）"""
        for line in lines[:200]:  # 只检查前200行
            # GPU信息
            if 'cuda' in line.lower() or 'gpu' in line.lower():
                if 'device' in line.lower():
                    self.data['metadata']['gpu_info'] = line.strip()
            
            # 学习率
            if 'learning_rate' in line.lower() or 'lr' in line:
                match = re.search(r'lr[:\s=]+([0-9.e-]+)', line, re.IGNORECASE)
                if match:
                    self.data['metadata']['learning_rate'] = float(match.group(1))
            
            # Batch size
            if 'batch' in line.lower() and 'size' in line.lower():
                match = re.search(r'batch[_\s]*size[:\s=]+(\d+)', line, re.IGNORECASE)
                if match:
                    self.data['metadata']['batch_size'] = int(match.group(1))
            
            # Epoch总数
            if 'epoch' in line.lower() and 'total' in line.lower():
                match = re.search(r'(\d+)', line)
                if match:
                    self.data['metadata']['total_epochs'] = int(match.group(1))
    
    def _extract_training_steps(self, lines: List[str]):
        """提取训练步骤信息"""
        for i, line in enumerate(lines):
            # 匹配训练loss
            # 格式示例: epoch 1 | loss 4.123 | ppl 61.45 | lr 0.0004
            if 'loss' in line.lower() and ('epoch' in line.lower() or 'step' in line.lower()):
                step_data = {}
                
                # 提取epoch
                epoch_match = re.search(r'epoch[:\s]+(\d+)', line, re.IGNORECASE)
                if epoch_match:
                    step_data['epoch'] = int(epoch_match.group(1))
                
                # 提取step
                step_match = re.search(r'step[:\s]+(\d+)', line, re.IGNORECASE)
                if step_match:
                    step_data['step'] = int(step_match.group(1))
                
                # 提取loss
                loss_match = re.search(r'loss[:\s=]+([0-9.]+)', line, re.IGNORECASE)
                if loss_match:
                    step_data['loss'] = float(loss_match.group(1))
                
                # 提取learning rate
                lr_match = re.search(r'lr[:\s=]+([0-9.e-]+)', line, re.IGNORECASE)
                if lr_match:
                    step_data['lr'] = float(lr_match.group(1))
                
                # 提取时间信息
                time_match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
                if time_match:
                    step_data['timestamp'] = time_match.group(1)
                
                if step_data:
                    self.data['training_steps'].append(step_data)
    
    def _extract_epochs(self, lines: List[str]):
        """提取每个epoch的汇总信息"""
        for i, line in enumerate(lines):
            # 匹配epoch结束信息
            if 'valid' in line.lower() and 'loss' in line.lower():
                epoch_data = {}
                
                # 查找附近的epoch信息
                for j in range(max(0, i-5), min(len(lines), i+5)):
                    epoch_match = re.search(r'epoch[:\s]+(\d+)', lines[j], re.IGNORECASE)
                    if epoch_match:
                        epoch_data['epoch'] = int(epoch_match.group(1))
                        break
                
                # 提取验证loss
                valid_loss_match = re.search(r'valid[_\s]*loss[:\s=]+([0-9.]+)', line, re.IGNORECASE)
                if valid_loss_match:
                    epoch_data['valid_loss'] = float(valid_loss_match.group(1))
                
                # 提取训练loss
                train_loss_match = re.search(r'train[_\s]*loss[:\s=]+([0-9.]+)', line, re.IGNORECASE)
                if train_loss_match:
                    epoch_data['train_loss'] = float(train_loss_match.group(1))
                
                # 提取准确率
                acc_match = re.search(r'acc[uracy]*[:\s=]+([0-9.]+)', line, re.IGNORECASE)
                if acc_match:
                    epoch_data['accuracy'] = float(acc_match.group(1))
                
                if epoch_data and 'epoch' in epoch_data:
                    # 避免重复
                    if not any(e.get('epoch') == epoch_data['epoch'] for e in self.data['epochs']):
                        self.data['epochs'].append(epoch_data)
    
    def _extract_final_results(self, lines: List[str]):
        """提取最终评估结果"""
        # 从文件末尾往前查找
        for line in reversed(lines[-200:]):
            # Top-1 accuracy
            if 'acc1' in line.lower():
                match = re.search(r'acc1[:\s=]+([0-9.]+)', line, re.IGNORECASE)
                if match:
                    self.data['final_results']['acc1'] = float(match.group(1))
            
            # Top-5 accuracy
            if 'acc5' in line.lower():
                match = re.search(r'acc5[:\s=]+([0-9.]+)', line, re.IGNORECASE)
                if match:
                    self.data['final_results']['acc5'] = float(match.group(1))
            
            # Average loss
            if 'avg_loss' in line.lower() or 'average loss' in line.lower():
                match = re.search(r'(?:avg_loss|average\s+loss)[:\s=]+([0-9.]+)', line, re.IGNORECASE)
                if match:
                    self.data['final_results']['avg_loss'] = float(match.group(1))
    
    def save_to_json(self, output_file: Path):
        """保存解析结果为JSON"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        print(f"✓ 解析结果已保存: {output_file}")
    
    def save_to_csv(self, output_dir: Path):
        """保存为CSV格式（用于Excel分析）"""
        output_dir.mkdir(exist_ok=True)
        
        # 保存训练步骤
        if self.data['training_steps']:
            df_steps = pd.DataFrame(self.data['training_steps'])
            csv_path = output_dir / 'training_steps.csv'
            df_steps.to_csv(csv_path, index=False)
            print(f"✓ 训练步骤已保存: {csv_path}")
        
        # 保存epoch汇总
        if self.data['epochs']:
            df_epochs = pd.DataFrame(self.data['epochs'])
            csv_path = output_dir / 'epochs_summary.csv'
            df_epochs.to_csv(csv_path, index=False)
            print(f"✓ Epoch汇总已保存: {csv_path}")
    
    def print_summary(self):
        """打印解析摘要"""
        print("\n" + "="*80)
        print("日志解析摘要")
        print("="*80)
        
        print(f"\n元数据:")
        for key, value in self.data['metadata'].items():
            print(f"  {key}: {value}")
        
        print(f"\n训练步骤数: {len(self.data['training_steps'])}")
        print(f"Epoch数: {len(self.data['epochs'])}")
        
        if self.data['epochs']:
            print(f"\nEpoch详情:")
            for epoch in self.data['epochs']:
                print(f"  Epoch {epoch.get('epoch', '?')}: "
                      f"train_loss={epoch.get('train_loss', 'N/A'):.4f}, "
                      f"valid_loss={epoch.get('valid_loss', 'N/A'):.4f}")
        
        if self.data['final_results']:
            print(f"\n最终结果:")
            for key, value in self.data['final_results'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        print("="*80 + "\n")


def parse_all_logs(experiments_root: Path):
    """解析所有实验的日志"""
    screen_dir = experiments_root.parent.parent.parent.parent / 'screen'
    
    if not screen_dir.exists():
        print(f"日志目录不存在: {screen_dir}")
        return
    
    log_files = list(screen_dir.glob('log_*.txt')) + list(screen_dir.glob('naturalcc_*.txt'))
    
    print(f"找到 {len(log_files)} 个日志文件\n")
    
    for log_file in log_files:
        parser = TrainingLogParser(log_file)
        data = parser.parse()
        
        # 确定实验名称
        exp_name = log_file.stem.replace('log_', '').replace('naturalcc_train_', '')
        exp_dir = experiments_root / exp_name
        
        if not exp_dir.exists():
            exp_dir = experiments_root / 'exp_batch_8' if 'base' in exp_name else experiments_root / exp_name
        
        if exp_dir.exists():
            # 保存解析结果
            parser.save_to_json(exp_dir / 'parsed_log.json')
            parser.save_to_csv(exp_dir)
            
            # 如果有最终结果，保存为results.json
            if data['final_results']:
                results = {
                    'name': exp_name,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    **data['final_results']
                }
                with open(exp_dir / 'results.json', 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"✓ 结果已保存: {exp_dir / 'results.json'}")
        
        parser.print_summary()


def main():
    import sys
    
    if len(sys.argv) > 1:
        # 解析指定的日志文件
        log_file = Path(sys.argv[1])
        parser = TrainingLogParser(log_file)
        data = parser.parse()
        parser.print_summary()
        
        if len(sys.argv) > 2:
            output_file = Path(sys.argv[2])
            parser.save_to_json(output_file)
    else:
        # 解析所有实验日志
        exp_root = Path(__file__).parent
        parse_all_logs(exp_root)


if __name__ == "__main__":
    main()
