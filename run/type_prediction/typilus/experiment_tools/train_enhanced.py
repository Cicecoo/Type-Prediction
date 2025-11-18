#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版训练脚本 - 添加自动日志和可视化
在原始train.py基础上增加训练记录功能
"""

import math
import os
import sys
import random
import json
from pathlib import Path

import numpy as np
import torch

# 添加父目录到路径以导入原始训练模块
sys.path.insert(0, str(Path(__file__).parent))

from ncc import LOGGER
from ncc import tasks
from ncc.data import iterators
from ncc.trainers.ncc_trainers import Trainer
from ncc.utils import checkpoint_utils, distributed_utils, utils
from ncc.utils.file_ops.yaml_io import load_yaml
from ncc.utils.file_utils import remove_files
from ncc.utils.logging import meters, metrics, progress_bar
import torch.multiprocessing


# ==================== 训练日志记录器 ====================

class TrainingLogger:
    """简化的训练日志记录器"""
    
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = {
            'epochs': [], 'train_loss': [], 'valid_loss': [],
            'train_ppl': [], 'valid_ppl': [], 'learning_rate': []
        }
        self.metrics_file = self.log_dir / 'metrics.json'
    
    def log(self, epoch, train_stats, valid_stats, lr):
        """记录一个epoch的数据"""
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_stats.get('loss', 0))
        self.history['valid_loss'].append(valid_stats.get('loss', 0) if valid_stats else 0)
        self.history['train_ppl'].append(train_stats.get('ppl', 0))
        self.history['valid_ppl'].append(valid_stats.get('ppl', 0) if valid_stats else 0)
        self.history['learning_rate'].append(lr)
        
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2)
    
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
            
            # 合并图
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Loss
            axes[0, 0].plot(epochs, self.history['train_loss'], 'o-', label='Train')
            if any(self.history['valid_loss']):
                axes[0, 0].plot(epochs, self.history['valid_loss'], 's-', label='Valid')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)
            
            # PPL
            if any(self.history['train_ppl']):
                axes[0, 1].plot(epochs, self.history['train_ppl'], 'o-', label='Train')
            if any(self.history['valid_ppl']):
                axes[0, 1].plot(epochs, self.history['valid_ppl'], 's-', label='Valid')
            axes[0, 1].set_ylabel('Perplexity')
            axes[0, 1].set_title('Perplexity')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)
            
            # 学习率
            axes[1, 0].plot(epochs, self.history['learning_rate'], 'o-', color='green')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(alpha=0.3)
            
            # 统计信息
            axes[1, 1].axis('off')
            stats_text = f"训练统计\n\n"
            stats_text += f"总轮数: {len(epochs)}\n"
            if self.history['train_loss']:
                stats_text += f"最佳训练Loss: {min(self.history['train_loss']):.4f}\n"
            valid_losses = [v for v in self.history['valid_loss'] if v > 0]
            if valid_losses:
                stats_text += f"最佳验证Loss: {min(valid_losses):.4f}\n"
            axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'training.png', dpi=120, bbox_inches='tight')
            plt.close()
            
            LOGGER.info(f'训练曲线已保存: {plots_dir / "training.png"}')
            
        except ImportError:
            LOGGER.warning('matplotlib未安装，跳过图表生成')
        except Exception as e:
            LOGGER.warning(f'生成图表错误: {e}')


training_logger = None


# ==================== 原始训练函数（复制） ====================

@metrics.aggregate('train')
def train(args, trainer, task, epoch_itr):
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
    num_updates = 0
    for samples in progress:
        with metrics.aggregate('train_inner'):
            log_output = trainer.train_step(samples)
            if log_output is None:
                continue

        num_updates = trainer.get_num_updates()
        if num_updates % args['common']['log_interval'] == 0:
            stats = get_training_stats(metrics.get_smoothed_values('train_inner'))
            progress.log(stats, tag='train_inner', step=num_updates)
            metrics.reset_meters('train_inner')

        if (not args['dataset']['disable_validation']
            and args['checkpoint']['save_interval_updates'] > 0
            and num_updates % args['checkpoint']['save_interval_updates'] == 0
            and num_updates > 0):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    stats = get_training_stats(metrics.get_smoothed_values('train'))
    progress.print(stats, tag='train', step=num_updates)
    metrics.reset_meters('train')
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    if args['dataset']['fixed_validation_seed'] is not None:
        utils.set_torch_seed(args['dataset']['fixed_validation_seed'])

    valid_losses = []
    valid_stats_list = []
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

        valid_losses.append(stats[args['checkpoint']['best_checkpoint_metric']])
        valid_stats_list.append(stats)

    return valid_losses, valid_stats_list[0] if valid_stats_list else None


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
    stats['wall'] = round(metrics.get_meter('default', 'wall').elapsed_time, 0)
    return stats


def should_stop_early(args, valid_loss):
    if valid_loss is None or args['checkpoint']['patience'] <= 0:
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
        return should_stop_early.num_runs >= args['checkpoint']['patience']


def single_main(args, init_distributed=False):
    global training_logger
    
    assert args['dataset']['max_tokens'] is not None or args['dataset']['max_sentences'] is not None
    metrics.reset()

    if torch.cuda.is_available() and not args['common']['cpu']:
        torch.cuda.set_device(args['distributed_training']['device_id'])
    random.seed(args['common']['seed'])
    np.random.seed(args['common']['seed'])
    torch.manual_seed(args['common']['seed'])
    torch.cuda.manual_seed(args['common']['seed'])
    if init_distributed:
        args['distributed_training']['distributed_rank'] = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        save_dir = args['checkpoint']['save_dir']
        checkpoint_utils.verify_checkpoint_directory(save_dir)
        
        # 初始化日志记录器
        log_dir = Path(save_dir).parent / 'logs'
        training_logger = TrainingLogger(log_dir)
        LOGGER.info(f'训练日志将保存到: {log_dir}')

    task = tasks.setup_task(args)
    task.load_dataset(args['dataset']['valid_subset'], combine=False, epoch=1)

    model = task.build_model(args)
    criterion = task.build_criterion(args)
    LOGGER.info(model)
    LOGGER.info('模型: {}, 损失函数: {}'.format(args['model']['arch'], criterion.__class__.__name__))
    LOGGER.info('模型参数: {} (可训练: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    trainer = Trainer(args, task, model, criterion)
    LOGGER.info('使用 {} 个GPU训练'.format(args['distributed_training']['distributed_world_size']))

    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer, combine=False)

    max_epoch = args['optimization']['max_epoch'] or math.inf
    max_update = args['optimization']['max_update'] or math.inf
    lr = trainer.get_lr()
    train_meter = meters.StopwatchMeter()
    train_meter.start()
    valid_subsets = args['dataset']['valid_subset'].split(',')
    
    while (lr > args['optimization']['min_lr']
           and epoch_itr.next_epoch_idx <= max_epoch
           and trainer.get_num_updates() < max_update):
        
        train_stats = train(args, trainer, task, epoch_itr)

        if not args['dataset']['disable_validation'] and epoch_itr.epoch % args['dataset']['validate_interval'] == 0:
            valid_losses, valid_stats = validate(args, trainer, task, epoch_itr, valid_subsets)
        else:
            valid_losses = [None]
            valid_stats = None

        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])
        
        # 记录日志
        if training_logger and distributed_utils.is_master(args):
            training_logger.log(epoch_itr.epoch, train_stats, valid_stats, lr)
            training_logger.plot()

        if epoch_itr.epoch % args['checkpoint']['save_interval'] == 0:
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if should_stop_early(args, valid_losses[0]):
            LOGGER.info('早停: 验证性能已{}轮未提升'.format(args['checkpoint']['patience']))
            break

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            combine=False,
            load_dataset=(os.pathsep in args['task']['data']),
        )

    train_meter.stop()
    LOGGER.info('训练完成，用时 {:.1f} 秒'.format(train_meter.sum))
    
    if training_logger and distributed_utils.is_master(args):
        training_logger.plot()
        LOGGER.info(f'所有日志已保存到: {training_logger.log_dir}')


def distributed_main(i, args, start_rank=0):
    args['distributed_training']['device_id'] = i
    if args['distributed_training']['distributed_rank'] is None:
        args['distributed_training']['distributed_rank'] = start_rank + i
    single_main(args, init_distributed=True)


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(description="Typilus训练（增强版）")
    parser.add_argument("--yaml_file", "-f", type=str, default='config/typilus',
                       help="配置文件路径（可以是绝对路径或相对路径，无.yml后缀）")
    args = parser.parse_args()
    
    # 如果传入的是绝对路径，直接使用；否则从脚本目录解析
    if os.path.isabs(args.yaml_file):
        yaml_file = '{}.yml'.format(args.yaml_file)
    else:
        yaml_file = os.path.join(os.path.dirname(__file__), '{}.yml'.format(args.yaml_file))
    
    LOGGER.info('加载配置: {}'.format(yaml_file))
    args = load_yaml(yaml_file)

    if args['distributed_training']['distributed_init_method'] is None:
        distributed_utils.infer_init_method(args)

    if args['distributed_training']['distributed_init_method'] is not None:
        if torch.cuda.device_count() > 1 and not args['distributed_training']['distributed_no_spawn']:
            start_rank = args['distributed_training']['distributed_rank']
            args['distributed_training']['distributed_rank'] = None
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args['distributed_training']['device_id'], args)
    elif args['distributed_training']['distributed_world_size'] > 1:
        assert args['distributed_training']['distributed_world_size'] <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args['distributed_training']['distributed_init_method'] = 'tcp://localhost:{port}'.format(port=port)
        args['distributed_training']['distributed_rank'] = None
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args,),
            nprocs=args['distributed_training']['distributed_world_size'],
        )
    else:
        LOGGER.info('单GPU训练...')
        single_main(args)


if __name__ == '__main__':
    cli_main()
