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
import torch.nn.functional as F


# ==================== 自动测试函数 ====================

def run_test_after_training(args, task, trainer):
    """训练完成后自动运行测试"""
    from ncc import LOGGER
    from ncc.utils import checkpoint_utils, utils
    from ncc.utils.logging import progress_bar
    
    # 设置测试配置
    test_subset = args['dataset'].get('test_subset', 'test')
    checkpoint_dir = Path(args['checkpoint']['save_dir'])
    best_checkpoint = checkpoint_dir / 'checkpoint_best.pt'
    
    if not best_checkpoint.exists():
        LOGGER.warning(f'找不到最佳checkpoint: {best_checkpoint}')
        return
    
    LOGGER.info(f'加载最佳checkpoint: {best_checkpoint}')
    LOGGER.info(f'测试集: {test_subset}')
    
    # 加载测试数据
    task.load_dataset(test_subset)
    
    # 加载最佳模型
    models, _model_args = checkpoint_utils.load_model_ensemble(
        [str(best_checkpoint)],
        task=task,
    )
    
    model = models[0]
    use_cuda = torch.cuda.is_available() and not args['common']['cpu']
    if use_cuda:
        model.cuda()
    model.eval()
    
    # 获取数据迭代器
    itr = task.get_batch_iterator(
        dataset=task.dataset(test_subset),
        max_tokens=args['dataset']['max_tokens'],
        max_sentences=args['dataset']['max_sentences'],
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            model.max_positions(),
        ),
        ignore_invalid_inputs=args['dataset']['skip_invalid_size_inputs_valid_test'],
        required_batch_size_multiple=args['dataset']['required_batch_size_multiple'],
        seed=args['common']['seed'],
        num_shards=1,
        shard_id=0,
        num_workers=0,
    ).next_epoch_itr(shuffle=False)
    
    progress = progress_bar.progress_bar(
        itr,
        log_format='simple',
        log_interval=100,
        prefix=f"测试 '{test_subset}' 集",
        default_log_format='simple',
    )
    
    # 准备评估
    tgt_dict = task.target_dictionary(key='supernodes.annotation.type')
    no_type_idx = tgt_dict.indices.get('O', -100)
    any_type_idx = tgt_dict.indices.get('$any$', None)
    
    def accuracy(output, target, topk=(1,), ignore_idx=[]):
        with torch.no_grad():
            maxk = max(topk)
            target_vocab_size = output.size(-1)
            keep_idx = torch.tensor(
                [i for i in range(target_vocab_size) if i not in ignore_idx],
                device=output.device,
            ).long()
            _, pred = output[..., keep_idx].topk(maxk, -1, True, True)
            pred = keep_idx[pred]
            correct = pred.eq(target.unsqueeze(-1).expand_as(pred)).long()
            mask = torch.ones_like(target).long()
            for idx in ignore_idx:
                mask = mask.long() & (~target.eq(idx)).long()
            deno = mask.sum().item()
            correct = correct * mask.unsqueeze(-1)
            res = []
            for k in topk:
                res.append(correct[..., :k].reshape(-1).float().sum().item())
            return res, deno
    
    # 运行测试
    num1, num5, num_labels_total = 0, 0, 0
    num1_any, num5_any, num_labels_any_total = 0, 0, 0
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for sample in progress:
            if use_cuda:
                sample = utils.move_to_cuda(sample)
            if 'net_input' not in sample:
                continue
            
            net_output = model(**sample['net_input'])
            logits = net_output[0] if isinstance(net_output, (tuple, list)) else net_output
            
            if 'target_ids' in sample:
                logits = logits.index_select(0, sample['target_ids'])
            labels = sample['target'].view(-1)
            
            loss = F.cross_entropy(logits, labels, ignore_index=no_type_idx)
            total_loss += loss.item()
            
            logits_metric = logits.unsqueeze(0)
            labels_metric = labels.unsqueeze(0)
            
            (corr1_any, corr5_any), num_labels_any = accuracy(
                logits_metric.cpu(), labels_metric.cpu(), topk=(1, 5), ignore_idx=(no_type_idx,)
            )
            num1_any += corr1_any
            num5_any += corr5_any
            num_labels_any_total += num_labels_any
            
            (corr1, corr5), num_labels = accuracy(
                logits_metric.cpu(), labels_metric.cpu(), topk=(1, 5),
                ignore_idx=tuple(idx for idx in (no_type_idx, any_type_idx) if idx is not None),
            )
            num1 += corr1
            num5 += corr5
            num_labels_total += num_labels
            count += 1
    
    # 计算结果
    avg_loss = float(total_loss) / count
    acc1 = float(num1) / num_labels_total * 100
    acc5 = float(num5) / num_labels_total * 100
    acc1_any = float(num1_any) / num_labels_any_total * 100
    acc5_any = float(num5_any) / num_labels_any_total * 100
    
    LOGGER.info('\n' + '='*80)
    LOGGER.info('测试结果:')
    LOGGER.info('-'*80)
    LOGGER.info(f'平均Loss:      {avg_loss:.4f}')
    LOGGER.info(f'Acc@1:         {acc1:.2f}%')
    LOGGER.info(f'Acc@5:         {acc5:.2f}%')
    LOGGER.info(f'Acc@1 (含any): {acc1_any:.2f}%')
    LOGGER.info(f'Acc@5 (含any): {acc5_any:.2f}%')
    LOGGER.info('='*80)
    
    # 保存结果
    res_file = checkpoint_dir / 'res.txt'
    with open(res_file, 'w') as f:
        f.write(f'avg_loss: {avg_loss}\n')
        f.write(f'acc1: {acc1}\n')
        f.write(f'acc5: {acc5}\n')
        f.write(f'acc1_any: {acc1_any}\n')
        f.write(f'acc5_any: {acc5_any}\n')
    
    LOGGER.info(f'测试结果已保存: {res_file}')
    
    # 更新日志
    log_dir = checkpoint_dir.parent / 'logs'
    if log_dir.exists():
        # 更新详细日志
        detailed_log = log_dir / 'detailed_metrics.txt'
        with open(detailed_log, 'a', encoding='utf-8') as f:
            f.write("\n\n")
            f.write("="*80 + "\n")
            f.write("TEST RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"avg_loss                      : {avg_loss:.4f}\n")
            f.write(f"acc1                          : {acc1:.2f}%\n")
            f.write(f"acc5                          : {acc5:.2f}%\n")
            f.write(f"acc1_any                      : {acc1_any:.2f}%\n")
            f.write(f"acc5_any                      : {acc5_any:.2f}%\n")
        
        # 更新 metrics.json
        metrics_file = log_dir / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            
            metrics_data['test_results'] = {
                'avg_loss': avg_loss,
                'acc1': acc1,
                'acc5': acc5,
                'acc1_any': acc1_any,
                'acc5_any': acc5_any,
            }
            
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2)
        
        LOGGER.info(f'训练日志已更新: {log_dir}')


# ==================== 训练日志记录器 ====================

class TrainingLogger:
    """详细的训练日志记录器"""
    
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir).resolve()  # 使用绝对路径
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
            print(f"[TrainingLogger] 原始输出将保存到: {self.raw_log_file}")
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
        """记录一个epoch的训练统计（包含所有可用数据）"""
        # 简化指标
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_stats.get('loss', 0))
        self.history['valid_loss'].append(valid_stats.get('loss', 0) if valid_stats else 0)
        self.history['train_ppl'].append(train_stats.get('ppl', 0))
        self.history['valid_ppl'].append(valid_stats.get('ppl', 0) if valid_stats else 0)
        self.history['learning_rate'].append(lr)
        
        # 详细指标（保存所有字段）
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
            print(f"[TrainingLogger] Epoch {epoch} 指标已保存: {self.metrics_file}")
        except Exception as e:
            print(f"[TrainingLogger] 保存metrics.json失败: {e}")
        
        # 保存详细指标（文本格式，易于查看）
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
            
            # 记录训练过程输出
            if hasattr(trainer, '_logger') and trainer._logger:
                trainer._logger.log_raw_output(f"Step {num_updates}: {stats}")
            
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

        # 记录验证输出
        if hasattr(trainer, '_logger') and trainer._logger:
            trainer._logger.log_raw_output(f"Validation [{subset}]: {stats}")

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
    
    # 将logger附加到trainer以便在训练循环中访问
    if distributed_utils.is_master(args):
        trainer._logger = training_logger
    else:
        trainer._logger = None

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
        training_logger.close()
        training_logger.plot()
        LOGGER.info(f'所有日志已保存到: {training_logger.log_dir}')
        
        # 自动运行测试
        LOGGER.info('\n' + '='*80)
        LOGGER.info('开始测试...')
        LOGGER.info('='*80)
        
        try:
            run_test_after_training(args, task, trainer)
        except Exception as e:
            LOGGER.warning(f'测试过程出错: {e}')
            import traceback
            traceback.print_exc()


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
