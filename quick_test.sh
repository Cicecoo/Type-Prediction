#!/bin/bash
# 快速验证实验 - 小数据集，2个epoch快速测试

echo "=========================================="
echo "Quick Test Experiment"
echo "=========================================="
echo "Steps:"
echo "  1. Sample small dataset (1000 samples)"
echo "  2. Train (2 epochs, LSTM 1 layer 128 dim)"
echo "  3. Validate"
echo "  4. Test"
echo "Purpose: Verify complete pipeline works"
echo "=========================================="
echo ""

cd /home/zhaojunzhang/workspace/type_pred/naturalcc
conda activate naturalcc

# 运行快速测试（包含数据采样、训练、验证、测试）
python run_quick_test.py \
  --exp-name quick_test \
  --base-dir /mnt/data1/zhaojunzhang/experiments/quick_test \
  --data-dir /mnt/data1/zhaojunzhang/typilus-data/transformer \
  --small-data-dir /mnt/data1/zhaojunzhang/typilus-data/transformer_small \
  --n-samples 1000

echo ""
echo "=========================================="
if [ $? -eq 0 ]; then
    echo "✓ Quick test completed successfully!"
    echo "✓ Complete pipeline is working"
    echo ""
    echo "Results:"
    echo "  - Checkpoints: /mnt/data1/zhaojunzhang/experiments/quick_test/quick_test/checkpoints/"
    echo "  - Train log: /mnt/data1/zhaojunzhang/experiments/quick_test/quick_test/logs/train.log"
    echo "  - Eval log: /mnt/data1/zhaojunzhang/experiments/quick_test/quick_test/logs/eval.log"
    echo "  - Results: /mnt/data1/zhaojunzhang/experiments/quick_test/quick_test/results/"
else
    echo "✗ Quick test failed"
    echo "✗ Check logs: /mnt/data1/zhaojunzhang/experiments/quick_test/quick_test/logs/"
fi
echo "=========================================="
