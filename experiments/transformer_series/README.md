# Transformer Type Prediction Experiments

**Created:** 2025-11-20T00:10:39.561863  
**Total Experiments:** 18

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
