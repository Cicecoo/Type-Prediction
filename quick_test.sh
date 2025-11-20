#!/bin/bash
# 快速验证实验 - 最小配置，2个epoch快速测试

echo "=========================================="
echo "Quick Test Experiment"
echo "=========================================="
echo "Config: LSTM, 1 layer, 128 dim, 2 epochs"
echo "Purpose: Verify training pipeline works"
echo "=========================================="
echo ""

cd /home/zhaojunzhang/workspace/type_pred/naturalcc
conda activate naturalcc

python run_transformer_experiment.py \
  --exp-name quick_test \
  --base-dir /mnt/data1/zhaojunzhang/experiments/quick_test \
  --data-dir /mnt/data1/zhaojunzhang/typilus-data/transformer \
  --encoder-type lstm \
  --encoder-layers 1 \
  --encoder-embed-dim 128 \
  --dropout 0.1 \
  --lr 0.001 \
  --batch-size 16 \
  --max-epoch 2 \
  --warmup-updates 100

echo ""
echo "=========================================="
if [ $? -eq 0 ]; then
    echo "✓ Quick test completed successfully!"
    echo "✓ Training pipeline is working"
else
    echo "✗ Quick test failed"
    echo "✗ Check logs for errors"
fi
echo "=========================================="
