#!/usr/bin/env python3
"""
Transformer类型预测实验管理脚本

功能：
1. 自动创建实验目录结构
2. 生成训练配置文件
3. 执行训练
4. 训练后自动测试
5. 记录实验信息和结果
"""

import os
import sys
import yaml
import json
import argparse
import subprocess
import shutil
from datetime import datetime
from pathlib import Path


class TransformerExperiment:
    """Transformer实验管理器"""
    
    def __init__(self, exp_name, base_dir, data_dir, config_template=None):
        """
        Args:
            exp_name: 实验名称
            base_dir: 实验基础目录
            data_dir: 数据目录
            config_template: 配置模板路径
        """
        self.exp_name = exp_name
        self.base_dir = Path(base_dir)
        self.data_dir = Path(data_dir)
        self.exp_dir = self.base_dir / exp_name
        
        # 实验子目录
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.log_dir = self.exp_dir / 'logs'
        self.result_dir = self.exp_dir / 'results'
        self.config_path = self.exp_dir / 'config.yml'
        self.info_path = self.exp_dir / 'info.txt'
        
        # 配置模板
        if config_template and os.path.exists(config_template):
            with open(config_template, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
    
    def _get_default_config(self):
        """获取默认配置（完整版本，包含所有必需字段）"""
        return {
            'criterion': 'type_predicition_cross_entropy',
            'optimizer': 'adam_simple',
            'lr_scheduler': 'polynomial_decay',
            'tokenizer': None,
            'bpe': None,
            
            'common': {
                'no_progress_bar': 0,
                'log_interval': 100,
                'log_format': 'simple',
                'tensorboard_logdir': str(self.log_dir / 'tensorboard'),
                'seed': 1,
                'cpu': 0,
                'fp16': 0,
                'memory_efficient_fp16': 0,
                'fp16_no_flatten_grads': 0,
                'fp16_init_scale': 128,
                'fp16_scale_window': None,
                'fp16_scale_tolerance': 0.0,
                'min_loss_scale': 1e-4,
                'threshold_loss_scale': None,
                'user_dir': None,
                'empty_cache_freq': 0,
                'all_gather_list_size': 16384,
                'task': 'type_prediction',
            },
            
            'dataset': {
                'num_workers': 0,
                'skip_invalid_size_inputs_valid_test': 1,
                'max_tokens': None,
                'max_sentences': 16,
                'required_batch_size_multiple': 8,
                'dataset_impl': 'raw',
                'train_subset': 'train',
                'valid_subset': 'valid',
                'validate_interval': 1,
                'fixed_validation_seed': None,
                'disable_validation': 0,
                'max_tokens_valid': None,
                'max_sentences_valid': 16,
                'curriculum': 100,
                'test_subset': 'test',
                'num_shards': 1,
                'shard_id': 0,
                'joined_dictionary': 1,
                'srcdict': str(self.data_dir / 'dict.txt'),
                # 'src_sp': 不再需要，代码已修改为跳过SentencePiece加载
                'tgtdict': str(self.data_dir / 'dict.txt'),
            },
            
            'distributed_training': {
                'distributed_world_size': 1,
                'distributed_rank': 0,
                'distributed_backend': 'nccl',
                'distributed_init_method': None,
                'distributed_port': -1,
                'device_id': 0,
                'distributed_no_spawn': 0,
                'ddp_backend': 'c10d',
                'bucket_cap_mb': 25,
                'fix_batches_to_gpus': None,
                'find_unused_parameters': 0,
                'fast_stat_sync': 0,
                'broadcast_buffers': 0,
                'global_sync_iter': 50,
                'warmup_iterations': 500,
            },
            
            'task': {
                'data': str(self.data_dir),
                'sample_break_mode': 'complete',
                'tokens_per_sample': 1024,
                'mask_prob': 0.15,
                'leave_unmasked_prob': 0.1,
                'random_token_prob': 0.1,
                'freq_weighted_replacement': 0,
                'mask_whole_words': 0,
                'pooler_activation_fn': 'tanh',
                'source_lang': 'code',
                'target_lang': 'type',
                'load_alignments': 0,
                'left_pad_source': 1,
                'left_pad_target': 0,
                'max_source_positions': 2048,
                'max_target_positions': 2048,
                'upsample_primary': 1,
                'truncate_source': 1,
                'eval_accuracy': 1,
            },
            
            'model': {
                'arch': 'typetransformer',
                'pooler_dropout': 0.0,
                'activation_fn': 'gelu',
                'dropout': 0.1,
                'attention_dropout': 0.1,
                'activation_dropout': 0.0,
                'relu_dropout': 0.0,
                'encoder_type': 'lstm',
                'encoder_positional_embeddings': 1,
                'encoder_embed_path': 0,
                'encoder_embed_dim': 512,
                'encoder_ffn_embed_dim': 2048,
                'encoder_layers': 2,
                'encoder_attention_heads': 8,
                'encoder_normalize_before': 0,
                'encoder_learned_pos': 0,
                'decoder_embed_path': '',
                'decoder_embed_dim': 0,
                'decoder_ffn_embed_dim': 0,
                'decoder_layers': 0,
                'decoder_attention_heads': 0,
                'decoder_learned_pos': 0,
                'decoder_normalize_before': 0,
                'share_decoder_input_output_embed': 0,
                'share_all_embeddings': 0,
                'no_token_positional_embeddings': 0,
                'adaptive_softmax_cutoff': 0,
                'adaptive_softmax_dropout': 0.0,
                'no_cross_attention': 0,
                'cross_self_attention': 0,
                'layer_wise_attention': 0,
                'encoder_layerdrop': 0.0,
                'decoder_layerdrop': 0.0,
                'encoder_layers_to_keep': None,
                'decoder_layers_to_keep': None,
                'layernorm_embedding': 0,
                'no_scale_embedding': 0,
                'encoder_max_relative_len': 0,
                'max_source_positions': 2048,
                'max_target_positions': 2048,
            },
            
            'optimization': {
                'max_epoch': 50,
                'max_update': 0,
                'clip_norm': 25,
                'sentence_avg': None,
                'update_freq': [1],
                'lr': [0.0001],
                'min_lr': -1,
                'use_bmuf': 0,
                'force_anneal': None,
                'warmup_updates': 1000,
                'end_learning_rate': 0.0,
                'power': 1.0,
                'total_num_update': 50000,
                'adam': {
                    'adam_betas': '(0.9, 0.999)',
                    'adam_eps': 1e-6,
                    'weight_decay': 0.01,
                    'use_old_adam': 1,
                },
                'adagrad': {
                    'weight_decay': 0.0,
                },
                'binary_cross_entropy': {
                    'infonce': 0,
                    'loss-weights': '',
                    'log-keys': '',
                },
            },
            
            'checkpoint': {
                'save_dir': str(self.checkpoint_dir),
                'restore_file': 'checkpoint_last.pt',
                'reset_dataloader': None,
                'reset_lr_scheduler': None,
                'reset_meters': None,
                'reset_optimizer': None,
                'optimizer_overrides': '{}',
                'save_interval': 1,
                'save_interval_updates': 0,
                'keep_interval_updates': 0,
                'keep_last_epochs': 5,
                'keep_best_checkpoints': 3,
                'no_save': 0,
                'no_epoch_checkpoints': 0,
                'no_last_checkpoints': 0,
                'no_save_optimizer_state': None,
                'best_checkpoint_metric': 'accuracy',
                'maximize_best_checkpoint_metric': 1,
                'patience': 10,
            },
            
            'eval': {
                'path': str(self.checkpoint_dir / 'checkpoint_best.pt'),
                'remove_bpe': None,
                'quiet': 0,
                'results_path': None,
                'model_overrides': '{}',
            }
        }
    
    def setup_experiment(self):
        """创建实验目录结构"""
        print(f"\n{'='*60}")
        print(f"Setting up experiment: {self.exp_name}")
        print(f"{'='*60}")
        
        # 创建目录
        for directory in [self.exp_dir, self.checkpoint_dir, self.log_dir, self.result_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        
        # 创建tensorboard日志目录
        tb_dir = self.log_dir / 'tensorboard'
        tb_dir.mkdir(exist_ok=True)
        
        return True
    
    def update_config(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if '.' in key:
                # 支持嵌套键，如 'model.encoder_layers'
                parts = key.split('.')
                config = self.config
                for part in parts[:-1]:
                    if part not in config:
                        config[part] = {}
                    config = config[part]
                config[parts[-1]] = value
            else:
                self.config[key] = value
        
        # 更新路径
        self.config['common']['tensorboard_logdir'] = str(self.log_dir / 'tensorboard')
        self.config['checkpoint']['save_dir'] = str(self.checkpoint_dir)
        self.config['eval']['path'] = str(self.checkpoint_dir / 'checkpoint_best.pt')
    
    def save_config(self):
        """保存配置文件"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Saved config to: {self.config_path}")
        
        # 同时保存到transformer的config目录（让train.py能找到）
        transformer_config_dir = Path('run/type_prediction/transformer/config')
        transformer_config_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用实验名称作为配置名称
        transformer_config_path = transformer_config_dir / f'{self.exp_name}.yml'
        with open(transformer_config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        print(f"✓ Also saved to: {transformer_config_path}")
    
    def save_experiment_info(self, extra_info=None):
        """保存实验信息"""
        info = {
            'experiment_name': self.exp_name,
            'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_directory': str(self.data_dir),
            'config': {
                'learning_rate': self.config['optimization']['lr'][0],
                'batch_size': self.config['dataset']['max_sentences'],
                'max_epochs': self.config['optimization']['max_epoch'],
                'encoder_type': self.config['model']['encoder_type'],
                'encoder_layers': self.config['model']['encoder_layers'],
                'encoder_embed_dim': self.config['model']['encoder_embed_dim'],
                'dropout': self.config['model']['dropout'],
            }
        }
        
        if extra_info:
            info.update(extra_info)
        
        with open(self.info_path, 'w') as f:
            f.write(f"Experiment: {self.exp_name}\n")
            f.write(f"{'='*60}\n\n")
            for key, value in info.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        print(f"✓ Saved experiment info to: {self.info_path}")
    
    def train(self):
        """执行训练"""
        print(f"\n{'='*60}")
        print("Starting training...")
        print(f"{'='*60}\n")
        
        # 构建训练命令 - transformer的train.py使用--language参数
        log_file = self.log_dir / 'train.log'
        
        # transformer的train.py会从config/目录加载{language}.yml
        # 我们在save_config时已经保存了一份到那里
        train_script = 'run/type_prediction/transformer/train.py'
        
        cmd = f"python {train_script} --language {self.exp_name} 2>&1 | tee {log_file}"
        
        print(f"Command: {cmd}\n")
        print(f"Note: Config file will be loaded from run/type_prediction/transformer/config/{self.exp_name}.yml\n")
        
        # 执行训练
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=os.getcwd(),
                check=True
            )
            print(f"\n✓ Training completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Training failed with error code {e.returncode}")
            print(f"\nPlease check:")
            print(f"1. NaturalCC is properly installed")
            print(f"2. You are in the naturalcc directory")
            print(f"3. The config file exists: run/type_prediction/transformer/config/{self.exp_name}.yml")
            print(f"4. Training script exists: {train_script}")
            return False
    
    def evaluate(self):
        """训练后评估"""
        print(f"\n{'='*60}")
        print("Evaluating model on test set...")
        print(f"{'='*60}\n")
        
        # 检查最佳checkpoint是否存在
        best_ckpt = self.checkpoint_dir / 'checkpoint_best.pt'
        if not best_ckpt.exists():
            print(f"Warning: Best checkpoint not found, using last checkpoint")
            best_ckpt = self.checkpoint_dir / 'checkpoint_last.pt'
        
        if not best_ckpt.exists():
            print(f"✗ No checkpoint found for evaluation")
            return False
        
        # 构建评估命令
        log_file = self.log_dir / 'eval.log'
        cmd = f"ncc-eval --configs {self.config_path} 2>&1 | tee {log_file}"
        cmd_fallback = f"python -m ncc_cli.eval --configs {self.config_path} 2>&1 | tee {log_file}"
        
        print(f"Command: {cmd}\n")
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=os.getcwd(),
                check=True
            )
            print(f"\n✓ Evaluation completed!")
            
            # 解析评估结果
            self._parse_eval_results(log_file)
            return True
            
        except subprocess.CalledProcessError as e:
            # 尝试fallback命令
            print(f"ncc-eval not found, trying python -m ncc_cli.eval...")
            try:
                result = subprocess.run(
                    cmd_fallback,
                    shell=True,
                    cwd=os.getcwd(),
                    check=True
                )
                print(f"\n✓ Evaluation completed!")
                self._parse_eval_results(log_file)
                return True
            except subprocess.CalledProcessError as e2:
                print(f"\n✗ Evaluation failed with error code {e2.returncode}")
                return False
    
    def _parse_eval_results(self, log_file):
        """解析评估结果并保存"""
        if not os.path.exists(log_file):
            return
        
        results = {}
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                # 提取准确率等指标
                if 'accuracy' in line.lower():
                    results['accuracy'] = line
                elif 'precision' in line.lower():
                    results['precision'] = line
                elif 'recall' in line.lower():
                    results['recall'] = line
                elif 'f1' in line.lower():
                    results['f1'] = line
        
        # 保存结果
        result_file = self.result_dir / 'test_results.txt'
        with open(result_file, 'w') as f:
            f.write(f"Test Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*60}\n\n")
            for key, value in results.items():
                f.write(f"{value}\n")
        
        print(f"✓ Results saved to: {result_file}")
    
    def run_full_experiment(self):
        """运行完整实验流程"""
        # 1. 创建目录
        if not self.setup_experiment():
            return False
        
        # 2. 保存配置
        self.save_config()
        
        # 3. 保存实验信息
        self.save_experiment_info()
        
        # 4. 训练
        if not self.train():
            return False
        
        # 5. 评估
        self.evaluate()
        
        print(f"\n{'='*60}")
        print(f"Experiment completed: {self.exp_name}")
        print(f"Results saved in: {self.exp_dir}")
        print(f"{'='*60}\n")
        
        return True


def main():
    parser = argparse.ArgumentParser(description='Run Transformer type prediction experiment')
    
    parser.add_argument('--exp-name', type=str, required=True,
                       help='Experiment name')
    parser.add_argument('--base-dir', type=str, 
                       default='/mnt/data1/zhaojunzhang/experiments/transformer',
                       help='Base directory for experiments')
    parser.add_argument('--data-dir', type=str,
                       default='/mnt/data1/zhaojunzhang/typilus-data/transformer',
                       help='Data directory')
    parser.add_argument('--config-template', type=str,
                       help='Config template file')
    
    # 模型超参数
    parser.add_argument('--encoder-type', type=str, default='lstm',
                       choices=['lstm', 'transformer'],
                       help='Encoder type')
    parser.add_argument('--encoder-layers', type=int, default=2,
                       help='Number of encoder layers')
    parser.add_argument('--encoder-embed-dim', type=int, default=512,
                       help='Encoder embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # 训练超参数
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--max-epoch', type=int, default=50,
                       help='Maximum epochs')
    parser.add_argument('--warmup-updates', type=int, default=1000,
                       help='Warmup updates')
    
    # 控制选项
    parser.add_argument('--skip-train', action='store_true',
                       help='Skip training (only evaluate)')
    parser.add_argument('--skip-eval', action='store_true',
                       help='Skip evaluation')
    
    args = parser.parse_args()
    
    # 创建实验
    exp = TransformerExperiment(
        exp_name=args.exp_name,
        base_dir=args.base_dir,
        data_dir=args.data_dir,
        config_template=args.config_template
    )
    
    # 更新配置
    exp.update_config(
        **{
            'model.encoder_type': args.encoder_type,
            'model.encoder_layers': args.encoder_layers,
            'model.encoder_embed_dim': args.encoder_embed_dim,
            'model.dropout': args.dropout,
            'model.attention_dropout': args.dropout,
            'optimization.lr': [args.lr],
            'optimization.max_epoch': args.max_epoch,
            'optimization.warmup_updates': args.warmup_updates,
            'dataset.max_sentences': args.batch_size,
            'dataset.max_sentences_valid': args.batch_size,
        }
    )
    
    # 设置实验
    exp.setup_experiment()
    exp.save_config()
    exp.save_experiment_info()
    
    # 训练
    if not args.skip_train:
        if not exp.train():
            sys.exit(1)
    
    # 评估
    if not args.skip_eval:
        exp.evaluate()
    
    print(f"\n{'='*60}")
    print(f"✓ Experiment completed: {args.exp_name}")
    print(f"✓ Results in: {exp.exp_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
