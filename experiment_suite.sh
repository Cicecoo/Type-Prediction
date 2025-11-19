#!/bin/bash
# Transformer类型预测实验套件 - 快速启动脚本

echo "=========================================="
echo "Transformer Type Prediction Experiment Suite"
echo "=========================================="
echo

# 检查是否在服务器上
if [ ! -d "/mnt/data1/zhaojunzhang/typilus-data" ]; then
    echo "❌ Error: Data directory not found"
    echo "Please run this script on the server"
    exit 1
fi

# 激活环境
source /mnt/data1/zhaojunzhang/packages/anaconda3/etc/profile.d/conda.sh
conda activate naturalcc

echo "✓ Environment activated: naturalcc"
echo

# 显示菜单
echo "Please select an action:"
echo "1. Generate all experiment configurations"
echo "2. Run baseline experiment"
echo "3. Run model size experiments"
echo "4. Run learning rate experiments"
echo "5. Run all experiments (serial)"
echo "6. Run all experiments (parallel, multi-GPU)"
echo "7. Analyze results and generate report"
echo "8. View experiment status"
echo "9. Exit"
echo

read -p "Enter your choice [1-9]: " choice

case $choice in
    1)
        echo "Generating experiment configurations..."
        python generate_experiment_suite.py
        echo "✓ Done! Check experiments/transformer_series/"
        ;;
    
    2)
        echo "Running baseline experiment..."
        python run_batch_experiments.py --group baseline --gpus 0
        ;;
    
    3)
        echo "Running model size experiments..."
        python run_batch_experiments.py --group model_size --gpus 0
        ;;
    
    4)
        echo "Running learning rate experiments..."
        python run_batch_experiments.py --group lr --gpus 0
        ;;
    
    5)
        echo "Running all experiments (serial mode)..."
        echo "This will take a long time. Continue? (y/n)"
        read confirm
        if [ "$confirm" = "y" ]; then
            python run_batch_experiments.py --all --mode serial --gpus 0
        fi
        ;;
    
    6)
        echo "Running all experiments (parallel mode)..."
        echo "Enter GPU IDs (e.g., '0 1 2 3'): "
        read gpu_ids
        echo "This will take a long time. Continue? (y/n)"
        read confirm
        if [ "$confirm" = "y" ]; then
            python run_batch_experiments.py --all --mode parallel --gpus $gpu_ids
        fi
        ;;
    
    7)
        echo "Analyzing results..."
        python analyze_results.py --plot --format both
        echo "✓ Done! Check report.html and results_comparison.md"
        ;;
    
    8)
        echo "Experiment Status:"
        echo "=================="
        if [ -f "experiments/transformer_series/experiment_index.json" ]; then
            python -c "
import json
from pathlib import Path

index_file = Path('experiments/transformer_series/experiment_index.json')
with open(index_file) as f:
    index = json.load(f)

print(f\"Total experiments: {index['total_experiments']}\")
print()

# Count by status
status_count = {}
for exp_dir in Path('experiments/transformer_series').iterdir():
    if not exp_dir.is_dir() or exp_dir.name.startswith('.'):
        continue
    meta_file = exp_dir / 'meta.json'
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
            status = meta.get('status', 'unknown')
            status_count[status] = status_count.get(status, 0) + 1

print('Status Summary:')
for status, count in sorted(status_count.items()):
    print(f'  {status}: {count}')
"
        else
            echo "No experiments found. Please generate configurations first."
        fi
        ;;
    
    9)
        echo "Goodbye!"
        exit 0
        ;;
    
    *)
        echo "Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo
echo "=========================================="
echo "Operation completed!"
echo "=========================================="
