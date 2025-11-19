#!/usr/bin/env python3
"""
分析和可视化Transformer实验结果
生成对比报告，类似Typilus论文中的结果表格
"""

import os
import json
import yaml
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ExperimentAnalyzer:
    def __init__(self, exp_dir="experiments/transformer_series"):
        self.exp_dir = Path(exp_dir)
        self.experiments = self._load_experiments()
    
    def _load_experiments(self):
        """加载所有实验的元数据和结果"""
        experiments = []
        
        for exp_path in self.exp_dir.iterdir():
            if not exp_path.is_dir() or exp_path.name.startswith('.'):
                continue
            
            meta_file = exp_path / "meta.json"
            if not meta_file.exists():
                continue
            
            with open(meta_file) as f:
                meta = json.load(f)
            
            # 加载结果
            results_file = exp_path / "results" / "test_results.json"
            if results_file.exists():
                with open(results_file) as f:
                    results = json.load(f)
                meta['results'] = results
            
            # 加载配置
            config_file = exp_path / "config.yml"
            if config_file.exists():
                with open(config_file) as f:
                    config = yaml.safe_load(f)
                meta['config'] = config
            
            experiments.append(meta)
        
        return experiments
    
    def get_completed_experiments(self):
        """获取已完成的实验"""
        return [exp for exp in self.experiments if exp.get('status') == 'completed']
    
    def compare_by_group(self, metric='accuracy'):
        """按实验组对比结果"""
        groups = {
            'model_size': [],
            'layers': [],
            'lr': [],
            'dropout': [],
            'encoder': [],
            'batch_size': []
        }
        
        completed = self.get_completed_experiments()
        
        for exp in completed:
            name = exp['name']
            
            if 'exp_d_model_' in name:
                groups['model_size'].append(exp)
            elif 'exp_layers_' in name:
                groups['layers'].append(exp)
            elif 'exp_lr_' in name:
                groups['lr'].append(exp)
            elif 'exp_dropout_' in name:
                groups['dropout'].append(exp)
            elif 'exp_encoder_' in name:
                groups['encoder'].append(exp)
            elif 'exp_batch_' in name:
                groups['batch_size'].append(exp)
        
        return groups
    
    def generate_comparison_table(self, output_file='results_comparison.md'):
        """生成对比表格（Markdown格式）"""
        completed = self.get_completed_experiments()
        
        if not completed:
            print("No completed experiments found")
            return
        
        # 构建表格数据
        table_data = []
        
        for exp in completed:
            results = exp.get('results', {})
            config = exp.get('config', {})
            run_info = exp.get('run_info', {})
            
            row = {
                'Experiment': exp['name'],
                'Encoder': config.get('model', {}).get('encoder_type', '-'),
                'd_model': config.get('model', {}).get('d_model', '-'),
                'Layers': config.get('model', {}).get('n_encoder_layers', '-'),
                'LR': config.get('optimizer', {}).get('lr', '-'),
                'Dropout': config.get('model', {}).get('dropout', '-'),
                'Batch Size': config.get('training', {}).get('batch_size', '-'),
                'Accuracy': f"{results.get('accuracy', 0)*100:.2f}%",
                'Top-5 Acc': f"{results.get('top5_accuracy', 0)*100:.2f}%",
                'Loss': f"{results.get('loss', 0):.4f}",
                'Training Time': f"{run_info.get('elapsed_seconds', 0)/3600:.2f}h",
            }
            table_data.append(row)
        
        # 生成Markdown表格
        df = pd.DataFrame(table_data)
        
        with open(output_file, 'w') as f:
            f.write("# Transformer Type Prediction Results\n\n")
            f.write(f"**Total Experiments:** {len(completed)}\n\n")
            f.write("## Results Summary\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n")
            
            # 添加最佳结果
            f.write("## Best Results\n\n")
            
            if 'accuracy' in results:
                best_acc = max(completed, key=lambda x: x.get('results', {}).get('accuracy', 0))
                f.write(f"**Best Accuracy:** {best_acc['name']} - "
                       f"{best_acc['results']['accuracy']*100:.2f}%\n\n")
            
            if 'top5_accuracy' in results:
                best_top5 = max(completed, key=lambda x: x.get('results', {}).get('top5_accuracy', 0))
                f.write(f"**Best Top-5 Accuracy:** {best_top5['name']} - "
                       f"{best_top5['results']['top5_accuracy']*100:.2f}%\n\n")
        
        print(f"✓ Comparison table saved to {output_file}")
        return df
    
    def plot_accuracy_comparison(self, output_dir='visualizations'):
        """绘制准确率对比图"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        groups = self.compare_by_group()
        
        # 设置绘图风格
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 为每个实验组生成图表
        for group_name, exps in groups.items():
            if not exps:
                continue
            
            # 提取数据
            names = []
            accuracies = []
            
            for exp in exps:
                results = exp.get('results', {})
                if 'accuracy' in results:
                    names.append(exp['name'].replace(f'exp_{group_name}_', ''))
                    accuracies.append(results['accuracy'] * 100)
            
            if not names:
                continue
            
            # 绘图
            fig, ax = plt.subplots()
            bars = ax.bar(names, accuracies, color='skyblue', edgecolor='navy', alpha=0.7)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}%',
                       ha='center', va='bottom')
            
            ax.set_xlabel(group_name.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.set_title(f'Accuracy by {group_name.replace("_", " ").title()}', fontsize=14)
            ax.set_ylim([0, 100])
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            output_file = output_dir / f'accuracy_{group_name}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved plot: {output_file}")
    
    def plot_learning_curves(self, exp_names=None, output_file='learning_curves.png'):
        """绘制学习曲线"""
        if exp_names is None:
            # 默认绘制所有已完成的实验
            completed = self.get_completed_experiments()
            exp_names = [exp['name'] for exp in completed[:5]]  # 最多5个
        
        plt.figure(figsize=(12, 6))
        
        for exp_name in exp_names:
            # 读取训练日志
            log_file = self.exp_dir / exp_name / "logs" / "train.log"
            
            if not log_file.exists():
                continue
            
            # 解析日志提取loss和accuracy
            epochs = []
            losses = []
            
            with open(log_file) as f:
                for line in f:
                    # 这里需要根据实际的日志格式解析
                    # 示例：假设日志格式为 "epoch X | loss Y.YY"
                    if 'epoch' in line and 'loss' in line:
                        try:
                            parts = line.split('|')
                            epoch = int(parts[0].split()[-1])
                            loss = float(parts[1].split()[-1])
                            epochs.append(epoch)
                            losses.append(loss)
                        except:
                            continue
            
            if epochs and losses:
                plt.plot(epochs, losses, label=exp_name, marker='o', markersize=3)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Curves', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved learning curves: {output_file}")
    
    def generate_html_report(self, output_file='report.html'):
        """生成HTML格式的完整报告"""
        completed = self.get_completed_experiments()
        
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Transformer Type Prediction - Experiment Results</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        th {
            background-color: #4CAF50;
            color: white;
            padding: 12px;
            text-align: left;
        }
        td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .best {
            background-color: #d4edda;
            font-weight: bold;
        }
        .metric {
            display: inline-block;
            padding: 5px 10px;
            margin: 5px;
            background-color: #e3f2fd;
            border-radius: 4px;
        }
        .chart {
            margin: 20px 0;
            text-align: center;
        }
        .chart img {
            max-width: 100%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <h1>Transformer Type Prediction - Experiment Results</h1>
    <p><strong>Total Experiments:</strong> {total}</p>
    <p><strong>Completed:</strong> {completed}</p>
    <p><strong>Generated:</strong> {timestamp}</p>
""".format(
            total=len(self.experiments),
            completed=len(completed),
            timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # 添加结果表格
        if completed:
            html += "<h2>Results Summary</h2>\n"
            df = self.generate_comparison_table()
            html += df.to_html(index=False, classes='results-table')
        
        # 添加图表（如果存在）
        viz_dir = Path('visualizations')
        if viz_dir.exists():
            html += "<h2>Visualizations</h2>\n"
            for img in viz_dir.glob('*.png'):
                html += f'<div class="chart"><img src="{img}" alt="{img.stem}"></div>\n'
        
        html += """
</body>
</html>
"""
        
        with open(output_file, 'w') as f:
            f.write(html)
        
        print(f"✓ HTML report saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--exp-dir', type=str, default='experiments/transformer_series',
                       help='Experiments directory')
    parser.add_argument('--output', type=str, default='report.html',
                       help='Output report file')
    parser.add_argument('--format', choices=['html', 'markdown', 'both'], default='both',
                       help='Report format')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')
    
    args = parser.parse_args()
    
    analyzer = ExperimentAnalyzer(exp_dir=args.exp_dir)
    
    if args.format in ['markdown', 'both']:
        analyzer.generate_comparison_table('results_comparison.md')
    
    if args.plot:
        analyzer.plot_accuracy_comparison()
        analyzer.plot_learning_curves()
    
    if args.format in ['html', 'both']:
        analyzer.generate_html_report(args.output)
    
    print("\n✓ Analysis complete!")

if __name__ == '__main__':
    main()
