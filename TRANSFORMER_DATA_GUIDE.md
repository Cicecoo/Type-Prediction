# Transformer数据准备指南

## 问题说明

Typilus和Transformer使用不同的数据格式：
- **Typilus**: 图结构（nodes + edges）
- **Transformer**: 序列结构（code tokens + type labels）

## 解决方案

### 方法1：转换Typilus数据（推荐）

使用提供的转换脚本将typilus数据转为transformer格式：

```bash
# 在服务器上运行
python convert_typilus_to_transformer.py \
    --typilus-dir /mnt/data1/zhaojunzhang/typilus-data/typilus \
    --output-dir /mnt/data1/zhaojunzhang/typilus-data/transformer \
    --splits train valid test
```

这将生成：
```
/mnt/data1/zhaojunzhang/typilus-data/transformer/
├── train.code    # 代码序列
├── train.type    # 类型标注
├── valid.code
├── valid.type
├── test.code
└── test.type
```

### 方法2：创建配置文件指向Python数据

创建新的transformer配置文件 `run/type_prediction/transformer/config/python_train.yml`，参考`javascript_lstm_train.yml`，修改以下路径：

```yaml
dataset:
  srcdict: /mnt/data1/zhaojunzhang/typilus-data/transformer/nodes.dict.txt
  src_sp: /mnt/data1/zhaojunzhang/typilus-data/transformer/python.model  # 或使用identity tokenizer
  tgtdict: /mnt/data1/zhaojunzhang/typilus-data/transformer/types.dict.txt

task:
  data: /mnt/data1/zhaojunzhang/typilus-data/transformer
  source_lang: code
  target_lang: type
```

## 数据格式说明

### Transformer期望的格式

**train.code** (每行一个样本):
```
<s> def function_name ( arg1 : int ) -> str : return result </s>
```

**train.type** (与code对齐):
```
O O O O O O int O O O str O O O O
```

其中：
- `O` = 无类型标注
- 其他 = 实际类型（如 `int`, `str`, `List`, etc.）

### Typilus现有格式

**token-sequence**:
```json
[17,20,22,24,26,27,30,33,35,37,...]
```

**nodes**:
```json
{"0": ["def"], "1": ["function_name"], "2": ["("], ...}
```

**supernodes**:
```json
{"5": {"annotation": "int", "type": "parameter"}, ...}
```

## 注意事项

### 1. 词典问题

Transformer需要：
- `csnjs_8k_9995p_unigram_url.dict.txt` (源代码词典)
- `target.dict.txt` (类型词典)

你可以：
```bash
# 从typilus词典生成
cp typilus/type_inference/data-mmap/nodes.dict.json transformer/srcdict.txt
cp typilus/type_inference/data-mmap/supernodes.annotation.dict.json transformer/tgtdict.txt
```

### 2. SentencePiece模型

Transformer配置中需要`src_sp`（SentencePiece模型）。选项：
1. 训练一个（较复杂）
2. 使用identity tokenization（修改代码跳过subword切分）
3. 使用dummy模型

### 3. 数据量问题

如果转换后数据太大，考虑：
- 只转换一部分数据做快速实验
- 修改转换脚本添加采样参数

## 快速开始

```bash
# 1. 转换数据（在服务器上）
cd /home/zhaojunzhang/workspace/type_pred/naturalcc
python convert_typilus_to_transformer.py

# 2. 准备词典和SentencePiece（简化版）
cd /mnt/data1/zhaojunzhang/typilus-data/transformer
cp ../typilus/type_inference/data-mmap/nodes.dict.json ./source.dict.txt
cp ../typilus/type_inference/data-mmap/supernodes.annotation.dict.json ./target.dict.txt

# 3. 创建配置文件
cd ~/workspace/type_pred/naturalcc/run/type_prediction/transformer/config
cp javascript_lstm_train.yml python_train.yml
# 修改python_train.yml中的路径

# 4. 训练
python run/type_prediction/transformer/train.py --language python_train
```

## 对比实验建议

为了公平对比Typilus、LSTM、Transformer：

1. **使用相同数据集**: 都用Python数据（typilus的数据）
2. **使用相同的划分**: train/valid/test保持一致
3. **相同评估指标**: Acc@1, Acc@5

这样才能真正比较模型架构的差异！
