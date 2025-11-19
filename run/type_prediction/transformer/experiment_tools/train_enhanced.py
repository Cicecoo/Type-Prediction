#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transformer 增强版训练脚本 - 添加详细日志和实验追踪
与 Typilus 实验保持一致的日志格式
"""

import math
import os
import sys
import random
import json
from pathlib import Path

import numpy as np
import torch

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ncc import LOGGER
from ncc import tasks
from ncc.data import iterators
from ncc.trainers.ncc_trainers import Trainer
from ncc.utils import checkpoint_utils, distributed_utils, utils
from ncc.utils.file_ops.yaml_io import load_yaml
from ncc.utils.file_utils import remove_files
from ncc.utils.logging import meters, metrics, progress_bar


# ==================== 训练日志记录器 ====================

class TrainingLogger:
    """详细的训练日志记录器 - 与 Typilus 一致"""
    
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[TrainingLogger] 日志目录: {self.log_dir}")
        
        # 简化指标（用于绘图）
        self.history = {
            'epochs': [], 'train_loss': [], 'valid_loss': [],
            'train_ppl': [], 'valid_ppl': [], 'learning_rate': []
        }
        
        # 详细指标（包含所有原始数据）
        self.detailed_history = []
        
        self.metrics_file = self.log_dir / 'metrics.json'
        self.detailed_file = self.log_dir / 'detailed_metrics.txt'
        self.raw_log_file = self.log_dir / 'training_output.log'
        
        # 打开原始输出日志文件
        try:
            self.raw_log_handle = open(self.raw_log_file, 'w', encoding='utf-8', buffering=1)
            print(f"[TrainingLogger] 原始输出保存至: {self.raw_log_file}")
        except Exception as e:
            print(f"[TrainingLogger] 无法打开日志文件: {e}")
            self.raw_log_handle = None
    
    def log_raw_output(self, message):
        """记录原始训练输出"""
        if self.raw_log_handle:
            try:
                self.raw_log_handle.write(str(message) + '\n')
                self.raw_log_handle.flush()
            except:
                pass
    
    def log(self, epoch, train_stats, valid_stats, lr):
        """记录一个epoch的训练统计"""
        # 简化指标
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_stats.get('loss', 0))
        self.history['valid_loss'].append(valid_stats.get('loss', 0) if valid_stats else 0)
        self.history['train_ppl'].append(train_stats.get('ppl', 0))
        self.history['valid_ppl'].append(valid_stats.get('ppl', 0) if valid_stats else 0)
        self.history['learning_rate'].append(lr)
        
        # 详细指标
        epoch_detail = {
            'epoch': epoch,
            'learning_rate': lr,
            'train': dict(train_stats),
            'valid': dict(valid_stats) if valid_stats else {}
        }
        self.detailed_history.append(epoch_detail)
        
        # 保存简化指标（JSON）
        try:
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2)
            print(f"[TrainingLogger] Epoch {epoch} 指标已保存")
        except Exception as e:
            print(f"[TrainingLogger] 保存metrics.json失败: {e}")
        
        # 保存详细指标（文本格式）
        try:
            with open(self.detailed_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Epoch {epoch} | LR: {lr:.6f}\n")
                f.write(f"{'-'*80}\n")
                f.write("TRAIN:\n")
                for k, v in sorted(train_stats.items()):
                    f.write(f"  {k:30s}: {v}\n")
                if valid_stats:
                    f.write("\nVALID:\n")
                    for k, v in sorted(valid_stats.items()):
                        f.write(f"  {k:30s}: {v}\n")
                f.flush()
        except Exception as e:
            print(f"[TrainingLogger] 保存详细日志失败: {e}")
    
    def close(self):
        """关闭日志文件"""
        if self.raw_log_handle:
            self.raw_log_handle.close()
    
    def plot(self):
        """生成训练曲线"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            if not self.history['epochs']:
                return
            
            plots_dir = self.log_dir / 'plots'
            plots_dir.mkdir(exist_ok=True)
            
            epochs = self.history['epochs']
            
            # 创建合并图
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Loss
            axes[0, 0].plot(epochs, self.history['train_loss'], 'o-', label='Train')
            if any(self.history['valid_loss']):
                axes[0, 0].plot(epochs, self.history['valid_loss'], 's-', label='Valid')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
            
            # PPL
            if any(self.history['train_ppl']):
                axes[0, 1].plot(epochs, self.history['train_ppl'], 'o-', label='Train')
            if any(self.history['valid_ppl']):
                axes[0, 1].plot(epochs, self.history['valid_ppl'], 's-', label='Valid')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Perplexity')
            axes[0, 1].set_title('Perplexity')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
            
            # Learning Rate
            axes[1, 0].plot(epochs, self.history['learning_rate'], 'o-', color='green')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].grid(alpha=0.3)
            
            # Gap (overfitting indicator)
            if any(self.history['valid_loss']):
                gaps = [t - v for t, v in zip(self.history['train_loss'], self.history['valid_loss'])]
                axes[1, 1].plot(epochs, gaps, 'o-', color='red')
                axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Train - Valid Loss')
                axes[1, 1].set_title('Train-Valid Gap (Overfitting Indicator)')
                axes[1, 1].grid(alpha=0.3)
            
            plt.tight_layout()
            plot_file = plots_dir / 'training_curves.png'
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"[TrainingLogger] 训练曲线已保存: {plot_file}")
        except Exception as e:
            print(f"[TrainingLogger] 绘图失败: {e}")


# ==================== 训练函数 ====================

@metrics.aggregate('train')
def train(args, trainer, task, epoch_itr, logger=None):
    """Train the model for one epoch."""
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args['distributed_training']['fix_batches_to_gpus'],
        shuffle=(epoch_itr.next_epoch_idx > args['dataset']['curriculum']),
    )
    update_freq = (
        args['optimization']['update_freq'][epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(args['optimization']['update_freq'])
        else args['optimization']['update_freq'][-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args['common']['log_format'],
        log_interval=args['common']['log_interval'],
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            args['common']['tensorboard_logdir'] if distributed_utils.is_master(args) else None
        ),
        default_log_format=('tqdm' if not args['common']['no_progress_bar'] else 'simple'),
    )

    task.begin_epoch(epoch_itr.epoch, trainer.get_model())

    valid_subsets = args['dataset']['valid_subset'].split(',')
    max_update = args['optimization']['max_update'] or math.inf
    
    for samples in progress:
        with metrics.aggregate('train_inner'):
            log_output = trainer.train_step(samples)
            if log_output is None:
                continue

        # 记录原始输出
        if logger:
            logger.log_raw_output(f"Epoch {epoch_itr.epoch} | Update {trainer.get_num_updates()}")

        num_updates = trainer.get_num_updates()
        if num_updates % args['common']['log_interval'] == 0:
            stats = get_training_stats(metrics.get_smoothed_values('train_inner'))
            progress.log(stats, tag='train_inner', step=num_updates)
            
            # 记录到日志
            if logger:
                logger.log_raw_output(f"  train_inner: {stats}")
            
            metrics.reset_meters('train_inner')

        if (
            not args['dataset']['disable_validation']
            and args['checkpoint']['save_interval_updates'] > 0
            and num_updates % args['checkpoint']['save_interval_updates'] == 0
            and num_updates > 0
        ):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets, logger)
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    stats = get_training_stats(metrics.get_smoothed_values('train'))
    progress.print(stats, tag='train')
    metrics.reset_meters('train')
    
    return stats


def validate(args, trainer, task, epoch_itr, subsets, logger=None):
    """Evaluate the model on the validation set(s)."""
    if args['dataset']['fixed_validation_seed'] is not None:
        utils.set_torch_seed(args['dataset']['fixed_validation_seed'])

    valid_losses = []
    for subset in subsets:
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args['dataset']['max_tokens_valid'],
            max_sentences=args['dataset']['max_sentences_valid'],
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args['dataset']['skip_invalid_size_inputs_valid_test'],
            required_batch_size_multiple=args['dataset']['required_batch_size_multiple'],
            seed=args['common']['seed'],
            num_shards=args['distributed_training']['distributed_world_size'],
            shard_id=args['distributed_training']['distributed_rank'],
            num_workers=args['dataset']['num_workers'],
        ).next_epoch_itr(shuffle=False)
        
        progress = progress_bar.progress_bar(
            itr,
            log_format=args['common']['log_format'],
            log_interval=args['common']['log_interval'],
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                args['common']['tensorboard_logdir'] if distributed_utils.is_master(args) else None
            ),
            default_log_format=('tqdm' if not args['common']['no_progress_bar'] else 'simple'),
        )

        with metrics.aggregate(new_root=True) as agg:
            for sample in progress:
                trainer.valid_step(sample)

        stats = get_valid_stats(args, trainer, agg.get_smoothed_values())
        progress.print(stats, tag=subset, step=trainer.get_num_updates())
        
        # 记录验证输出
        if logger:
            logger.log_raw_output(f"Valid {subset}: {stats}")

        valid_losses.append(stats[args['checkpoint']['best_checkpoint_metric']])

    return valid_losses


def get_valid_stats(args, trainer, stats):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args['checkpoint']['best_checkpoint_metric'])
        best_function = max if args['checkpoint']['maximize_best_checkpoint_metric'] else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[args['checkpoint']['best_checkpoint_metric']],
        )
    return stats


def get_training_stats(stats):
    if 'nll_loss' in stats and 'ppl' not in stats:
        stats['ppl'] = utils.get_perplexity(stats['nll_loss'])
    return stats


def should_stop_early(args, valid_loss):
    if valid_loss is None:
        return False
    if args['checkpoint']['patience'] <= 0:
        return False

    def is_better(a, b):
        return a > b if args['checkpoint']['maximize_best_checkpoint_metric'] else a < b

    prev_best = getattr(should_stop_early, 'best', None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= args['checkpoint']['patience']:
            LOGGER.info(f'Early stop: validation hasn\'t improved for {args["checkpoint"]["patience"]} epochs')
        return should_stop_early.num_runs >= args['checkpoint']['patience']


def single_main(args, init_distributed=False):
    """Main training function with enhanced logging."""
    assert args['dataset']['max_tokens'] is not None or args['dataset']['max_sentences'] is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'
    metrics.reset()

    # 初始化日志记录器
    logger = TrainingLogger(args['checkpoint']['save_dir'])
    logger.log_raw_output("=" * 80)
    logger.log_raw_output("Transformer Type Prediction Training")
    logger.log_raw_output("=" * 80)

    # Initialize CUDA
    if torch.cuda.is_available() and not args['common']['cpu']:
        torch.cuda.set_device(args['distributed_training']['device_id'])
    np.random.seed(args['common']['seed'])
    torch.manual_seed(args['common']['seed'])
    if init_distributed:
        args['distributed_training']['distributed_rank'] = distributed_utils.distributed_init(args)

    # Verify checkpoint directory
    if distributed_utils.is_master(args):
        save_dir = args['checkpoint']['save_dir']
        checkpoint_utils.verify_checkpoint_directory(save_dir)
        remove_files(save_dir, 'pt')

    LOGGER.info(args)
    logger.log_raw_output(f"Config: {json.dumps(args, indent=2, default=str)}")

    # Setup task
    task = tasks.setup_task(args)
    logger.log_raw_output(f"Task: {task.__class__.__name__}")

    # Load validation dataset
    for valid_sub_split in args['dataset']['valid_subset'].split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    LOGGER.info(model)
    LOGGER.info(f'Model {args["model"]["arch"]}, criterion {criterion.__class__.__name__}')
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(f'Model params: {num_params:,} (trainable: {num_trainable:,})')
    logger.log_raw_output(f"Model parameters: {num_params:,} (trainable: {num_trainable:,})")

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    LOGGER.info(f'Training on {args["distributed_training"]["distributed_world_size"]} GPUs')
    LOGGER.info(f'Max tokens per GPU: {args["dataset"]["max_tokens"]}, '
                f'Max sentences: {args["dataset"]["max_sentences"]}')

    # Load checkpoint
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer, combine=False)

    # Training loop
    max_epoch = args['optimization']['max_epoch'] or math.inf
    max_update = args['optimization']['max_update'] or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()
    valid_subsets = args['dataset']['valid_subset'].split(',')
    
    try:
        while (
            lr > args['optimization']['min_lr']
            and epoch_itr.next_epoch_idx <= max_epoch
            and trainer.get_num_updates() < max_update
        ):
            # Train for one epoch
            logger.log_raw_output(f"\n{'='*80}\nEpoch {epoch_itr.epoch}\n{'='*80}")
            train_stats = train(args, trainer, task, epoch_itr, logger)

            # Validate
            if not args['dataset']['disable_validation'] and \
               epoch_itr.epoch % args['dataset']['validate_interval'] == 0:
                valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets, logger)
                valid_stats = get_valid_stats(args, trainer, metrics.get_smoothed_values('valid'))
            else:
                valid_losses = [None]
                valid_stats = {}

            # Log metrics
            lr = trainer.get_lr()
            logger.log(epoch_itr.epoch, train_stats, valid_stats, lr)

            # Save checkpoint
            if epoch_itr.epoch % args['checkpoint']['save_interval'] == 0:
                checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

            # Early stopping
            if should_stop_early(args, valid_losses[0]):
                logger.log_raw_output(f"Early stopping triggered at epoch {epoch_itr.epoch}")
                break

            # Next epoch
            epoch_itr = trainer.get_train_iterator(
                epoch_itr.next_epoch_idx,
                combine=False,
                load_dataset=(os.pathsep in args['task']['data']),
            )

    finally:
        train_meter.stop()
        LOGGER.info(f'Done training in {train_meter.sum:.1f} seconds')
        logger.log_raw_output(f"\nTraining completed in {train_meter.sum:.1f} seconds")
        
        # Generate plots
        logger.plot()
        logger.close()


def cli_main():
    """Command-line interface."""
    import argparse
    parser = argparse.ArgumentParser(description="Train Transformer for type prediction")
    parser.add_argument(
        "--config", "-c", required=True, type=str,
        help="Path to config YAML file"
    )
    args = parser.parse_args()
    
    config_file = Path(args.config).resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    LOGGER.info(f'Loading config from {config_file}')
    args_dict = load_yaml(str(config_file))
    
    if args_dict['distributed_training']['distributed_init_method'] is None:
        distributed_utils.infer_init_method(args_dict)

    if args_dict['distributed_training']['distributed_init_method'] is not None:
        # Distributed training
        if torch.cuda.device_count() > 1 and not args_dict['distributed_training']['distributed_no_spawn']:
            start_rank = args_dict['distributed_training']['distributed_rank']
            args_dict['distributed_training']['distributed_rank'] = None
            torch.multiprocessing.spawn(
                fn=lambda i, a, sr: distributed_main(i, a, sr),
                args=(args_dict, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args_dict['distributed_training']['device_id'], args_dict)
    elif args_dict['distributed_training']['distributed_world_size'] > 1:
        # Fallback for multiple GPUs
        assert args_dict['distributed_training']['distributed_world_size'] <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args_dict['distributed_training']['distributed_init_method'] = f'tcp://localhost:{port}'
        args_dict['distributed_training']['distributed_rank'] = None
        torch.multiprocessing.spawn(
            fn=lambda i, a: distributed_main(i, a, 0),
            args=(args_dict,),
            nprocs=args_dict['distributed_training']['distributed_world_size'],
        )
    else:
        LOGGER.info('Single GPU training...')
        single_main(args_dict)


def distributed_main(device_id, args, start_rank=0):
    args['distributed_training']['device_id'] = device_id
    if args['distributed_training']['distributed_rank'] is None:
        args['distributed_training']['distributed_rank'] = start_rank + device_id
    single_main(args, init_distributed=True)


if __name__ == '__main__':
    cli_main()
