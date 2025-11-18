#!/usr/bin/env python3
"""
Typilus 实验结果可视化工具
生成训练曲线、性能对比图等
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from typing import Dict, List
import pandas as pd

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# 实验根目录
EXP_ROOT = Path(__file__).parent


class ExperimentVisualizer:
    """实验结果可视化类"""
    
    def __init__(self, output_dir="visualizations"):
        self.output_dir = EXP_ROOT / output_dir
        self.output_dir.mkdir(exist_ok=True)
        
    def parse_training_log(self, log_file: Path) -> Dict:
        """解析训练日志文件，提取loss和其他指标"""
        epochs = []
        train_losses = []
        valid_losses = []
        learning_rates = []
        
        if not log_file.exists():
            return {
                'epochs': epochs,
                'train_losses': train_losses,
                'valid_losses': valid_losses,
                'learning_rates': learning_rates
            }
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # 解析训练loss
                if 'train_loss' in line.lower() or 'loss' in line.lower():
                    try:
                        # 尝试提取loss值
                        if 'loss' in line and '=' in line:
                            parts = line.split('loss')
                            for part in parts:
                                if '=' in part:
                                    value = part.split('=')[1].split()[0].strip(',')
                                    try:
                                        loss_val = float(value)
                                        if 0 < loss_val < 100:  # 合理的loss范围
                                            train_losses.append(loss_val)
                                    except:
                                        pass
                    except:
                        pass
                
                # 解析epoch
                if 'epoch' in line.lower():
                    try:
                        if 'epoch' in line:
                            parts = line.split('epoch')
                            for part in parts[1:]:
                                words = part.split()
                                for word in words[:3]:
                                    try:
                                        epoch_num = int(word.strip(':,'))
                                        if epoch_num not in epochs and 0 < epoch_num <= 100:
                                            epochs.append(epoch_num)
                                            break
                                    except:
                                        pass
                    except:
                        pass
        
        return {
            'epochs': sorted(list(set(epochs))) if epochs else list(range(1, len(train_losses)//50 + 1)),
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'learning_rates': learning_rates
        }
    
    def plot_training_curves(self, experiments: Dict[str, Dict], save_name="training_curves.png"):
        """绘制训练曲线对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        for exp_name, data in experiments.items():
            if data['train_losses']:
                # 平滑处理
                window = min(50, len(data['train_losses']) // 10)
                if window > 1:
                    smoothed = pd.Series(data['train_losses']).rolling(window=window, center=True).mean()
                else:
                    smoothed = data['train_losses']
                
                steps = list(range(len(smoothed)))
                axes[0].plot(steps, smoothed, label=exp_name, alpha=0.8, linewidth=2)
        
        axes[0].set_xlabel('Training Steps', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training Loss Curves', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 绘制最后100步的放大图
        for exp_name, data in experiments.items():
            if data['train_losses'] and len(data['train_losses']) > 100:
                last_n = min(200, len(data['train_losses']))
                axes[1].plot(range(last_n), data['train_losses'][-last_n:], 
                           label=exp_name, alpha=0.8, linewidth=2)
        
        axes[1].set_xlabel('Last N Steps', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Training Loss (Last 200 Steps)', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 训练曲线已保存: {save_path}")
        plt.close()
    
    def plot_performance_comparison(self, results: List[Dict], save_name="performance_comparison.png"):
        """绘制性能对比柱状图"""
        if not results:
            print("没有可用的实验结果")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        exp_names = [r['name'] for r in results]
        acc1 = [r.get('acc1', 0) for r in results]
        acc5 = [r.get('acc5', 0) for r in results]
        loss = [r.get('avg_loss', 0) for r in results]
        
        # Top-1 Accuracy
        bars1 = axes[0, 0].bar(exp_names, acc1, alpha=0.8, edgecolor='black')
        axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0, 0].set_title('Top-1 Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, value in zip(bars1, acc1):
            if value > 0:
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                              f'{value:.2f}%', ha='center', va='bottom', fontsize=10)
        
        # Top-5 Accuracy
        bars2 = axes[0, 1].bar(exp_names, acc5, alpha=0.8, color='orange', edgecolor='black')
        axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0, 1].set_title('Top-5 Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars2, acc5):
            if value > 0:
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                              f'{value:.2f}%', ha='center', va='bottom', fontsize=10)
        
        # Loss
        bars3 = axes[1, 0].bar(exp_names, loss, alpha=0.8, color='green', edgecolor='black')
        axes[1, 0].set_ylabel('Loss', fontsize=12)
        axes[1, 0].set_title('Average Loss Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars3, loss):
            if value > 0:
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                              f'{value:.4f}', ha='center', va='bottom', fontsize=10)
        
        # 相对于baseline的提升
        if results and results[0]['name'] == 'baseline':
            baseline_acc1 = results[0].get('acc1', 22.54)
            improvements = [(r.get('acc1', 0) - baseline_acc1) for r in results]
            colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in improvements]
            
            bars4 = axes[1, 1].bar(exp_names, improvements, alpha=0.8, color=colors, edgecolor='black')
            axes[1, 1].set_ylabel('Improvement (%)', fontsize=12)
            axes[1, 1].set_title('Improvement over Baseline (Top-1 Acc)', fontsize=14, fontweight='bold')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            for bar, value in zip(bars4, improvements):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                              f'{value:+.2f}%', ha='center', 
                              va='bottom' if value > 0 else 'top', fontsize=10)
        else:
            axes[1, 1].text(0.5, 0.5, 'No baseline data', ha='center', va='center',
                          transform=axes[1, 1].transAxes, fontsize=14)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 性能对比图已保存: {save_path}")
        plt.close()
    
    def plot_hyperparameter_impact(self, results: List[Dict], save_name="hyperparameter_impact.png"):
        """绘制超参数影响分析图"""
        if len(results) < 2:
            print("需要至少2个实验结果才能分析超参数影响")
            return
        
        # 提取超参数变化
        params_data = {
            'Learning Rate': [],
            'Hidden Size': [],
            'Num Layers': [],
            'Dropout': []
        }
        
        for r in results:
            hp = r.get('hyperparameters', {})
            params_data['Learning Rate'].append(hp.get('learning_rate', 4e-4))
            params_data['Hidden Size'].append(hp.get('encoder_hidden_size', 64))
            params_data['Num Layers'].append(hp.get('encoder_layers', 2))
            params_data['Dropout'].append(hp.get('encoder_dropout', 0.1))
        
        acc1_values = [r.get('acc1', 0) for r in results]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Learning Rate vs Accuracy
        if len(set(params_data['Learning Rate'])) > 1:
            axes[0, 0].scatter(params_data['Learning Rate'], acc1_values, s=100, alpha=0.6)
            axes[0, 0].set_xlabel('Learning Rate', fontsize=12)
            axes[0, 0].set_ylabel('Top-1 Accuracy (%)', fontsize=12)
            axes[0, 0].set_title('Learning Rate Impact', fontsize=14, fontweight='bold')
            axes[0, 0].set_xscale('log')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Hidden Size vs Accuracy
        if len(set(params_data['Hidden Size'])) > 1:
            axes[0, 1].scatter(params_data['Hidden Size'], acc1_values, s=100, alpha=0.6, color='orange')
            axes[0, 1].set_xlabel('Hidden Size', fontsize=12)
            axes[0, 1].set_ylabel('Top-1 Accuracy (%)', fontsize=12)
            axes[0, 1].set_title('Hidden Size Impact', fontsize=14, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Num Layers vs Accuracy
        if len(set(params_data['Num Layers'])) > 1:
            axes[1, 0].scatter(params_data['Num Layers'], acc1_values, s=100, alpha=0.6, color='green')
            axes[1, 0].set_xlabel('Number of Layers', fontsize=12)
            axes[1, 0].set_ylabel('Top-1 Accuracy (%)', fontsize=12)
            axes[1, 0].set_title('Network Depth Impact', fontsize=14, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Dropout vs Accuracy
        if len(set(params_data['Dropout'])) > 1:
            axes[1, 1].scatter(params_data['Dropout'], acc1_values, s=100, alpha=0.6, color='red')
            axes[1, 1].set_xlabel('Dropout Rate', fontsize=12)
            axes[1, 1].set_ylabel('Top-1 Accuracy (%)', fontsize=12)
            axes[1, 1].set_title('Dropout Impact', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 超参数影响图已保存: {save_path}")
        plt.close()
    
    def generate_summary_table(self, results: List[Dict], save_name="results_summary.csv"):
        """生成结果汇总表格"""
        if not results:
            print("没有可用的实验结果")
            return
        
        data = {
            'Experiment': [],
            'Top-1 Acc (%)': [],
            'Top-5 Acc (%)': [],
            'Loss': [],
            'LR': [],
            'Hidden': [],
            'Layers': [],
            'Dropout': [],
            'Status': []
        }
        
        for r in results:
            data['Experiment'].append(r.get('name', 'unknown'))
            data['Top-1 Acc (%)'].append(f"{r.get('acc1', 0):.2f}")
            data['Top-5 Acc (%)'].append(f"{r.get('acc5', 0):.2f}")
            data['Loss'].append(f"{r.get('avg_loss', 0):.4f}")
            
            hp = r.get('hyperparameters', {})
            data['LR'].append(f"{hp.get('learning_rate', 4e-4):.0e}")
            data['Hidden'].append(hp.get('encoder_hidden_size', 64))
            data['Layers'].append(hp.get('encoder_layers', 2))
            data['Dropout'].append(hp.get('encoder_dropout', 0.1))
            data['Status'].append(r.get('status', 'unknown'))
        
        df = pd.DataFrame(data)
        save_path = self.output_dir / save_name
        df.to_csv(save_path, index=False)
        print(f"✓ 结果汇总表已保存: {save_path}")
        
        # 同时保存为markdown格式（用于报告）
        md_path = self.output_dir / save_name.replace('.csv', '.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# 实验结果汇总\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n---\n生成时间: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"✓ Markdown表格已保存: {md_path}")
        
        return df


def main():
    """主函数：加载所有实验结果并生成可视化"""
    print("="*80)
    print("Typilus 实验结果可视化")
    print("="*80 + "\n")
    
    visualizer = ExperimentVisualizer()
    
    # 定义所有实验
    experiments = {
        'baseline': EXP_ROOT / 'exp_batch_8',
        'exp_lr_2e4': EXP_ROOT / 'exp_lr_2e4',
        'exp_lr_1e4': EXP_ROOT / 'exp_lr_1e4',
        'exp_dropout_02': EXP_ROOT / 'exp_dropout_02',
        'exp_hidden_128': EXP_ROOT / 'exp_hidden_128',
        'exp_layers_4': EXP_ROOT / 'exp_layers_4',
    }
    
    # 收集训练日志数据
    training_data = {}
    results = []
    
    for exp_name, exp_dir in experiments.items():
        # 解析训练日志
        log_file = EXP_ROOT.parent.parent.parent.parent / 'screen' / f'log_{exp_name}.txt'
        if not log_file.exists():
            log_file = EXP_ROOT.parent.parent.parent.parent / 'screen' / f'naturalcc_train_{exp_name}.txt'
        
        if log_file.exists():
            print(f"解析日志: {exp_name}")
            training_data[exp_name] = visualizer.parse_training_log(log_file)
        
        # 加载实验结果
        info_file = exp_dir / 'experiment_info.json'
        result_file = exp_dir / 'results.json'
        
        result = {'name': exp_name}
        
        if info_file.exists():
            with open(info_file) as f:
                info = json.load(f)
                result.update(info)
        
        if result_file.exists():
            with open(result_file) as f:
                res = json.load(f)
                result.update(res)
        
        # 如果没有results.json，检查results.txt
        result_txt = exp_dir / 'results.txt'
        if result_txt.exists() and 'acc1' not in result:
            try:
                with open(result_txt) as f:
                    content = f.read()
                    if 'acc1:' in content:
                        acc1 = float(content.split('acc1:')[1].split()[0])
                        result['acc1'] = acc1
                    if 'acc5:' in content:
                        acc5 = float(content.split('acc5:')[1].split()[0])
                        result['acc5'] = acc5
                    if 'avg_loss:' in content:
                        loss = float(content.split('avg_loss:')[1].split()[0])
                        result['avg_loss'] = loss
            except:
                pass
        
        if 'acc1' in result or 'hyperparameters' in result:
            results.append(result)
    
    print(f"\n找到 {len(results)} 个实验结果")
    print(f"找到 {len(training_data)} 个训练日志\n")
    
    # 生成可视化
    if training_data:
        print("生成训练曲线...")
        visualizer.plot_training_curves(training_data)
    
    if results:
        print("生成性能对比图...")
        visualizer.plot_performance_comparison(results)
        
        print("生成超参数影响图...")
        visualizer.plot_hyperparameter_impact(results)
        
        print("生成结果汇总表...")
        df = visualizer.generate_summary_table(results)
        
        print("\n" + "="*80)
        print("可视化完成！")
        print("="*80)
        print(f"\n所有图表已保存到: {visualizer.output_dir}")
        print("\n生成的文件:")
        for f in sorted(visualizer.output_dir.glob('*')):
            print(f"  - {f.name}")
    else:
        print("\n⚠ 警告: 未找到任何实验结果")
        print("请先运行实验并保存结果到 experiment_info.json 或 results.json")


if __name__ == "__main__":
    main()
