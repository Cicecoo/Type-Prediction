#!/usr/bin/env python3
"""
Typilus风格的Transformer类型预测实验套件

实验设计：
1. Baseline: 默认超参数
2. 变量实验：
   - 模型大小 (d_model: 256, 512, 1024)
   - 层数 (n_layers: 4, 6, 8)
   - 学习率 (lr: 1e-4, 5e-4, 1e-3)
   - Dropout (0.0, 0.1, 0.2)
   - Encoder类型 (transformer vs lstm)
3. 数据规模实验：
   - 训练集大小: 10%, 25%, 50%, 100%
4. 消融实验：
   - 不同的优化器 (adam, sgd)
   - 不同的batch size (16, 32, 64)
"""

import os
import json
import yaml
import argparse
from datetime import datetime
from pathlib import Path

class TransformerExperimentSuite:
    def __init__(self, base_dir="experiments/transformer_series"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 基础配置
        self.base_config = {
            'model': {
                'arch': 'typetransformer',
                'encoder_type': 'lstm',  # transformer or lstm
                'd_model': 512,
                'd_rep': 128,
                'n_head': 8,
                'n_encoder_layers': 6,
                'd_ff': 2048,
                'dropout': 0.1,
                'activation': 'relu',
            },
            'optimizer': {
                'name': 'fairseq_adam',
                'lr': 5e-4,
                'weight_decay': 0.0001,
                'adam_betas': '(0.9, 0.98)',
                'adam_eps': 1e-8,
            },
            'training': {
                'max_epoch': 50,
                'batch_size': 32,
                'update_freq': 1,
                'clip_norm': 1.0,
                'patience': 10,  # early stopping
            },
            'task': {
                'max_source_positions': 512,
                'max_target_positions': 512,
            }
        }
    
    def create_experiment(self, name, modifications, description=""):
        """创建单个实验配置"""
        exp_dir = self.base_dir / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 合并配置
        config = self._merge_config(self.base_config, modifications)
        
        # 保存配置
        config_path = exp_dir / "config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # 保存元信息
        meta = {
            'name': name,
            'description': description,
            'created': datetime.now().isoformat(),
            'modifications': modifications,
            'status': 'pending'
        }
        
        meta_path = exp_dir / "meta.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        # 创建子目录
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        (exp_dir / "results").mkdir(exist_ok=True)
        (exp_dir / "visualizations").mkdir(exist_ok=True)
        
        print(f"✓ Created experiment: {name}")
        return exp_dir
    
    def _merge_config(self, base, modifications):
        """深度合并配置"""
        import copy
        config = copy.deepcopy(base)
        
        for key, value in modifications.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        
        return config
    
    def generate_baseline(self):
        """生成baseline实验"""
        return self.create_experiment(
            name="exp_baseline",
            modifications={},
            description="Baseline configuration with default hyperparameters"
        )
    
    def generate_model_size_experiments(self):
        """模型大小系列实验"""
        experiments = []
        
        for d_model in [256, 512, 1024]:
            exp = self.create_experiment(
                name=f"exp_d_model_{d_model}",
                modifications={
                    'model': {'d_model': d_model}
                },
                description=f"Model dimension: {d_model}"
            )
            experiments.append(exp)
        
        return experiments
    
    def generate_layer_experiments(self):
        """层数系列实验"""
        experiments = []
        
        for n_layers in [4, 6, 8]:
            exp = self.create_experiment(
                name=f"exp_layers_{n_layers}",
                modifications={
                    'model': {'n_encoder_layers': n_layers}
                },
                description=f"Number of encoder layers: {n_layers}"
            )
            experiments.append(exp)
        
        return experiments
    
    def generate_lr_experiments(self):
        """学习率系列实验"""
        experiments = []
        
        for lr in [1e-4, 5e-4, 1e-3]:
            exp = self.create_experiment(
                name=f"exp_lr_{lr:.0e}",
                modifications={
                    'optimizer': {'lr': lr}
                },
                description=f"Learning rate: {lr}"
            )
            experiments.append(exp)
        
        return experiments
    
    def generate_dropout_experiments(self):
        """Dropout系列实验"""
        experiments = []
        
        for dropout in [0.0, 0.1, 0.2]:
            exp = self.create_experiment(
                name=f"exp_dropout_{dropout}",
                modifications={
                    'model': {'dropout': dropout}
                },
                description=f"Dropout rate: {dropout}"
            )
            experiments.append(exp)
        
        return experiments
    
    def generate_encoder_type_experiments(self):
        """Encoder类型对比实验"""
        experiments = []
        
        for encoder_type in ['transformer', 'lstm']:
            exp = self.create_experiment(
                name=f"exp_encoder_{encoder_type}",
                modifications={
                    'model': {'encoder_type': encoder_type}
                },
                description=f"Encoder type: {encoder_type}"
            )
            experiments.append(exp)
        
        return experiments
    
    def generate_batch_size_experiments(self):
        """Batch size系列实验"""
        experiments = []
        
        for batch_size in [16, 32, 64]:
            exp = self.create_experiment(
                name=f"exp_batch_{batch_size}",
                modifications={
                    'training': {'batch_size': batch_size}
                },
                description=f"Batch size: {batch_size}"
            )
            experiments.append(exp)
        
        return experiments
    
    def generate_all_experiments(self):
        """生成所有实验"""
        print("=" * 60)
        print("Generating Transformer Type Prediction Experiment Suite")
        print("=" * 60)
        
        all_experiments = []
        
        print("\n1. Baseline")
        all_experiments.append(self.generate_baseline())
        
        print("\n2. Model Size Experiments")
        all_experiments.extend(self.generate_model_size_experiments())
        
        print("\n3. Layer Number Experiments")
        all_experiments.extend(self.generate_layer_experiments())
        
        print("\n4. Learning Rate Experiments")
        all_experiments.extend(self.generate_lr_experiments())
        
        print("\n5. Dropout Experiments")
        all_experiments.extend(self.generate_dropout_experiments())
        
        print("\n6. Encoder Type Experiments")
        all_experiments.extend(self.generate_encoder_type_experiments())
        
        print("\n7. Batch Size Experiments")
        all_experiments.extend(self.generate_batch_size_experiments())
        
        # 生成实验索引
        self._generate_index(all_experiments)
        
        print(f"\n{'=' * 60}")
        print(f"Total experiments created: {len(all_experiments)}")
        print(f"Experiments directory: {self.base_dir}")
        print(f"{'=' * 60}")
        
        return all_experiments
    
    def _generate_index(self, experiments):
        """生成实验索引文件"""
        index = {
            'created': datetime.now().isoformat(),
            'total_experiments': len(experiments),
            'experiments': []
        }
        
        for exp_dir in experiments:
            meta_path = exp_dir / "meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                    index['experiments'].append({
                        'name': meta['name'],
                        'description': meta['description'],
                        'path': str(exp_dir.relative_to(self.base_dir))
                    })
        
        index_path = self.base_dir / "experiment_index.json"
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        
        # 生成README
        readme = self._generate_readme(index)
        readme_path = self.base_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme)
    
    def _generate_readme(self, index):
        """生成README文档"""
        readme = f"""# Transformer Type Prediction Experiments

**Created:** {index['created']}  
**Total Experiments:** {index['total_experiments']}

## Experiment Groups

### 1. Baseline
- `exp_baseline`: Default configuration

### 2. Model Size (d_model)
- `exp_d_model_256`: d_model=256
- `exp_d_model_512`: d_model=512 (baseline)
- `exp_d_model_1024`: d_model=1024

### 3. Number of Layers
- `exp_layers_4`: 4 encoder layers
- `exp_layers_6`: 6 encoder layers (baseline)
- `exp_layers_8`: 8 encoder layers

### 4. Learning Rate
- `exp_lr_1e-04`: lr=1e-4
- `exp_lr_5e-04`: lr=5e-4 (baseline)
- `exp_lr_1e-03`: lr=1e-3

### 5. Dropout Rate
- `exp_dropout_0.0`: no dropout
- `exp_dropout_0.1`: dropout=0.1 (baseline)
- `exp_dropout_0.2`: dropout=0.2

### 6. Encoder Type
- `exp_encoder_transformer`: Transformer encoder
- `exp_encoder_lstm`: LSTM encoder (baseline)

### 7. Batch Size
- `exp_batch_16`: batch_size=16
- `exp_batch_32`: batch_size=32 (baseline)
- `exp_batch_64`: batch_size=64

## Running Experiments

### Single Experiment
```bash
python run_experiment.py --exp-name exp_baseline
```

### Batch Experiments
```bash
python run_batch_experiments.py --group model_size
```

### All Experiments
```bash
python run_batch_experiments.py --all
```

## Results Analysis

After experiments complete:

```bash
python analyze_results.py --output report.html
```

## Directory Structure

```
experiments/transformer_series/
├── exp_baseline/
│   ├── config.yml           # Experiment configuration
│   ├── meta.json           # Metadata
│   ├── checkpoints/        # Model checkpoints
│   ├── logs/              # Training logs
│   ├── results/           # Evaluation results
│   └── visualizations/    # Plots and figures
├── exp_d_model_256/
├── ...
└── experiment_index.json   # Experiment catalog
```

## Metrics Tracked

- **Accuracy**: Token-level type prediction accuracy
- **Top-5 Accuracy**: Top-5 prediction accuracy
- **Loss**: Training and validation loss
- **Precision/Recall/F1**: Per-type metrics
- **Training Time**: Wall-clock time per epoch
- **Memory Usage**: Peak GPU memory

## Comparison with Typilus

Compare results with Typilus baseline:

```bash
python compare_with_typilus.py --typilus-results path/to/typilus/results
```
"""
        return readme

def main():
    parser = argparse.ArgumentParser(description='Generate Transformer experiment suite')
    parser.add_argument('--base-dir', type=str, default='experiments/transformer_series',
                       help='Base directory for experiments')
    parser.add_argument('--groups', nargs='+', 
                       choices=['baseline', 'model_size', 'layers', 'lr', 'dropout', 
                               'encoder', 'batch_size', 'all'],
                       default=['all'],
                       help='Experiment groups to generate')
    
    args = parser.parse_args()
    
    suite = TransformerExperimentSuite(base_dir=args.base_dir)
    
    if 'all' in args.groups:
        suite.generate_all_experiments()
    else:
        if 'baseline' in args.groups:
            suite.generate_baseline()
        if 'model_size' in args.groups:
            suite.generate_model_size_experiments()
        if 'layers' in args.groups:
            suite.generate_layer_experiments()
        if 'lr' in args.groups:
            suite.generate_lr_experiments()
        if 'dropout' in args.groups:
            suite.generate_dropout_experiments()
        if 'encoder' in args.groups:
            suite.generate_encoder_type_experiments()
        if 'batch_size' in args.groups:
            suite.generate_batch_size_experiments()

if __name__ == '__main__':
    main()
