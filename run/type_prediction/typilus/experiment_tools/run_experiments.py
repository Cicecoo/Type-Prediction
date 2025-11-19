#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Typilus 参数调优实验工具 - 一体化脚本
支持自动化实验、日志记录、可视化和结果分析
"""

import os
import sys
import json
import yaml
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# ==================== 实验配置 ====================

EXPERIMENTS = [
    {
        "name": "baseline",
        "desc": "基线实验",
        "params": {}  # 使用默认配置
    },
    {
        "name": "exp_lr_1e-3",
        "desc": "学习率1e-3",
        "params": {"optimization": {"lrs": [1e-3]}}
    },
    {
        "name": "exp_lr_1e-4",
        "desc": "学习率1e-4",
        "params": {"optimization": {"lrs": [1e-4]}}
    },
    {
        "name": "exp_batch_64",
        "desc": "批量大小64",
        "params": {"dataset": {"max_sentences": 64}}
    },
    {
        "name": "exp_hidden_128",
        "desc": "隐藏层128",
        "params": {
            "model": {
                "encoder_embed_dim": 128,
                "encoder_hidden_size": 128,
                "edge_in": 128,
                "edge_out": 128
            }
        }
    },
    {
        "name": "exp_best",
        "desc": "推荐配置",
        "params": {
            "optimization": {"lrs": [5e-4]},
            "dataset": {"max_sentences": 64},
            "model": {
                "encoder_embed_dim": 128,
                "encoder_hidden_size": 128,
                "edge_in": 128,
                "edge_out": 128,
                "encoder_layers": 4
            }
        }
    }
]

# ==================== 工具函数 ====================

def deep_update(base_dict, update_dict):
    """递归更新字典"""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def get_default_config():
    """获取默认配置（不依赖外部文件）"""
    return {
        'criterion': 'typilus',
        'optimizer': 'torch_adam',
        'lr_scheduler': 'fixed',
        'tokenizer': None,
        'bpe': None,
        
        'common': {
            'no_progress_bar': 0,
            'log_interval': 50,
            'log_format': 'simple',
            'tensorboard_logdir': '',
            'memory_efficient_fp16': 1,
            'fp16_no_flatten_grads': 1,
            'fp16_init_scale': 128,
            'fp16_scale_window': None,
            'fp16_scale_tolerance': 0.0,
            'min_loss_scale': 1e-4,
            'threshold_loss_scale': None,
            'empty_cache_freq': 0,
            'task': 'typilus',
            'seed': 1,
            'cpu': 0,
            'fp16': 0,
            'fp16_opt_level': '01',
            'server_ip': '',
            'server_port': '',
            'bf16': 0,
        },
        
        'dataset': {
            'num_workers': 0,
            'skip_invalid_size_inputs_valid_test': 1,
            'max_tokens': None,
            'max_sentences': 32,
            'required_batch_size_multiple': 8,
            'dataset_impl': 'mmap',
            'train_subset': 'train',
            'valid_subset': 'valid',
            'validate_interval': 1,
            'fixed_validation_seed': None,
            'disable_validation': 0,
            'max_tokens_valid': None,
            'max_sentences_valid': 256,
            'curriculum': 0,
            'gen_subset': 'test',
            'num_shards': 1,
            'shard_id': 0,
            'test_subset': 'test',
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
            'local_rank': -1,
            'block_momentum': 0.875,
            'block_lr': 1,
            'use_nbm': 0,
            'average_sync': 0,
        },
        
        'task': {
            'data': '~/workspace/type_pred/typilus/type_inference/data-mmap',
            'source_langs': ['nodes', 'edges'],
            'target_langs': ['supernodes.annotation'],
            'load_alignments': 0,
            'left_pad_source': 0,
            'left_pad_target': 0,
            'max_source_positions': 512,
            'max_target_positions': 30,
            'upsample_primary': 1,
            'truncate_source': 1,
            'truncate_target': 1,
            'append_eos_to_target': 1,
            'eval_bleu': 1,
            'eval_bleu_detok': 'space',
            'eval_bleu_detok_args': None,
            'eval_tokenized_bleu': 0,
            'eval_bleu_remove_bpe': None,
            'eval_bleu_args': None,
            'eval_bleu_print_samples': 0,
        },
        
        'model': {
            'arch': 'typilus',
            'encoder_embed_dim': 64,
            'max_subtoken_len': 5,
            'encoder_hidden_size': 64,
            'encoder_layers': 2,
            'edge_types': 8,
            'edge_in': 64,
            'edge_out': 64,
            'edge_backward': 1,
            'timesteps': [7, 1],
            'encoder_dropout': 0.1,
            'decoder_dropout': 0.1,
        },
        
        'optimization': {
            'max_epoch': 0,
            'max_update': 0,
            'clip_norm': 25,
            'update_freq': [1],
            'lrs': [4e-4],
            'min_lr': -1,
            'use_bmuf': 1,
            'force_anneal': 0,
            'warmup_updates': 0,
            'lr_shrink': 0.98,
            'margin': 2,
            'sentence_avg': 1,
            'adam': {
                'adam_betas': '(0.9, 0.999)',
                'adam_eps': 1e-8,
                'weight_decay': 0.0,
                'use_old_adam': 0,
            },
            'weight_decay': 0.0,
            'adam_epsilon': 1e-8,
            'max_grad_norm': 1.0,
            'num_train_epochs': 5,
            'max_steps': -1,
            'warmup_steps': 0,
            'gradient_accumulation_steps': 1,
        },
        
        'checkpoint': {
            'restore_file': 'checkpoint_last.pt',
            'reset_dataloader': None,
            'reset_lr_scheduler': None,
            'reset_meters': None,
            'reset_optimizer': None,
            'optimizer_overrides': '{}',
            'save_interval': 1,
            'save_interval_updates': 0,
            'keep_interval_updates': 0,
            'keep_last_epochs': -1,
            'keep_best_checkpoints': -1,
            'no_save': 0,
            'no_epoch_checkpoints': 1,
            'no_last_checkpoints': 0,
            'no_save_optimizer_state': None,
            'best_checkpoint_metric': 'loss',
            'maximize_best_checkpoint_metric': 1,
            'patience': 10,
            'save_dir': '~/workspace/type_pred/naturalcc/run/type_prediction/typilus/checkpoints/base',
            'should_continue': 0,
            'model_name_or_path': None,
            'cache_dir': None,
            'logging_steps': 500,
            'save_steps': 2000,
            'save_total_limit': 2,
            'overwrite_output_dir': 0,
            'overwrite_cache': 0,
        },
        
        'eval': {
            'path': '~/workspace/type_pred/naturalcc/run/type_prediction/typilus/checkpoints/base/checkpoint_best.pt',
            'result_path': None,
            'remove_bpe': None,
            'quiet': 0,
            'model_overrides': '{}',
            'max_sentences': 2048,
            'beam': 1,
            'nbest': 1,
            'max_len_a': 0,
            'max_len_b': 30,
            'min_len': 1,
            'match_source_len': 0,
            'no_early_stop': 0,
            'unnormalized': 0,
            'no_beamable_mm': 0,
            'lenpen': 1,
            'unkpen': 0,
            'replace_unk': None,
            'sacrebleu': 0,
            'score_reference': 0,
            'prefix_size': 0,
            'no_repeat_ngram_size': 0,
            'sampling': 0,
            'sampling_topk': -1,
            'sampling_topp': -1,
            'temperature': 1.0,
            'diverse_beam_groups': -1,
            'diverse_beam_strength': 0.5,
            'diversity_rate': -1.0,
            'print_alignment': 0,
            'print_step': 0,
            'iter_decode_eos_penalty': 0.0,
            'iter_decode_max_iter': 10,
            'iter_decode_force_max_iter': 0,
            'iter_decode_with_beam': 1,
            'iter_decode_with_external_reranker': 0,
            'retain_iter_history': 0,
            'decoding_format': None,
            'nltk_bleu': 1,
            'rouge': 1,
        },
    }


def create_config(exp_config, output_path):
    """创建实验配置（不需要base_config文件）"""
    # 从默认配置开始
    config = get_default_config()
    
    # 应用实验特定的参数
    if exp_config.get("params"):
        config = deep_update(config, exp_config["params"])
    
    # 设置实验目录
    exp_dir = Path(output_path).parent
    config["checkpoint"]["save_dir"] = str(exp_dir / "checkpoints")
    config["common"]["tensorboard_logdir"] = str(exp_dir / "tensorboard")
    
    # 保存配置
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return config


def run_single_experiment(exp_config, train_script):
    """运行单个实验"""
    name = exp_config["name"]
    print(f"\n{'='*60}")
    print(f"实验: {name} - {exp_config['desc']}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    exp_dir = Path("~/workspace/type_pred/naturalcc/run/type_prediction/typilus/experiments").expanduser() / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = exp_dir / "config.yml"
    create_config(exp_config, str(config_path))
    
    # 保存实验信息
    with open(exp_dir / "info.txt", 'w', encoding='utf-8') as f:
        f.write(f"实验: {name}\n描述: {exp_config['desc']}\n")
        f.write(f"开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 运行训练（使用增强版脚本）
    log_file = exp_dir / "training.log"
    # 传递绝对路径（无.yml后缀）
    config_path_no_ext = str(config_path).replace('.yml', '')
    cmd = [sys.executable, train_script, "--yaml_file", config_path_no_ext]
    
    start_time = time.time()
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()
            process.wait()
        
        success = process.returncode == 0
        status = "成功" if success else f"失败({process.returncode})"
        print(f"\n{'✓' if success else '✗'} {name} {status}")
        
    except Exception as e:
        success = False
        status = f"异常: {str(e)}"
        print(f"\n✗ {name} {status}")
    
    # 更新信息
    duration = time.time() - start_time
    with open(exp_dir / "info.txt", 'a', encoding='utf-8') as f:
        f.write(f"结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"耗时: {duration/3600:.2f}小时\n状态: {status}\n")
    
    return success


def visualize_experiment(exp_dir):
    """为单个实验生成可视化"""
    metrics_file = Path(exp_dir) / "logs" / "metrics.json"
    if not metrics_file.exists():
        return
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        with open(metrics_file) as f:
            data = json.load(f)
        
        if not data.get('epochs'):
            return
        
        plots_dir = Path(exp_dir) / "logs" / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        epochs = data['epochs']
        
        # Loss图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(epochs, data['train_loss'], 'o-', label='Train')
        if any(data['valid_loss']):
            ax1.plot(epochs, data['valid_loss'], 's-', label='Valid')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # PPL图
        if any(data.get('train_ppl', [])):
            ax2.plot(epochs, data['train_ppl'], 'o-', label='Train')
        if any(data.get('valid_ppl', [])):
            ax2.plot(epochs, data['valid_ppl'], 's-', label='Valid')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Perplexity')
        ax2.set_title('Perplexity')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'training.png', dpi=120, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        pass
    except Exception as e:
        print(f"可视化错误: {e}")


def analyze_all_experiments(exp_base_dir):
    """分析所有实验结果"""
    exp_base_dir = Path(exp_base_dir).expanduser()
    if not exp_base_dir.exists():
        print("实验目录不存在")
        return
    
    results = []
    for exp_dir in exp_base_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        metrics_file = exp_dir / "logs" / "metrics.json"
        if not metrics_file.exists():
            continue
        
        with open(metrics_file) as f:
            data = json.load(f)
        
        if data.get('valid_loss'):
            valid_losses = [v for v in data['valid_loss'] if v > 0]
            best_loss = min(valid_losses) if valid_losses else 0
        else:
            best_loss = 0
        
        results.append({
            'name': exp_dir.name,
            'epochs': len(data.get('epochs', [])),
            'best_loss': best_loss,
            'final_loss': data['train_loss'][-1] if data.get('train_loss') else 0
        })
    
    if not results:
        print("没有找到实验结果")
        return
    
    # 打印对比表
    print("\n" + "="*80)
    print("实验结果对比")
    print("="*80)
    print(f"{'实验名称':<25} {'轮数':<8} {'最终Loss':<12} {'最佳验证Loss':<15}")
    print("-"*80)
    
    for r in sorted(results, key=lambda x: x['best_loss'] if x['best_loss'] > 0 else 999):
        loss_str = f"{r['best_loss']:.4f}" if r['best_loss'] > 0 else "N/A"
        print(f"{r['name']:<25} {r['epochs']:<8} {r['final_loss']:<12.4f} {loss_str:<15}")
    
    print("="*80)
    
    # 生成对比图
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 验证Loss对比
        for exp_dir in exp_base_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            metrics_file = exp_dir / "logs" / "metrics.json"
            if not metrics_file.exists():
                continue
            
            with open(metrics_file) as f:
                data = json.load(f)
            
            if data.get('epochs') and data.get('valid_loss'):
                epochs = data['epochs']
                valid_loss = [v if v > 0 else None for v in data['valid_loss']]
                axes[0].plot(epochs, valid_loss, 'o-', label=exp_dir.name, linewidth=2)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Validation Loss')
        axes[0].set_title('验证Loss对比')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # 最佳Loss柱状图
        names = [r['name'] for r in results if r['best_loss'] > 0]
        losses = [r['best_loss'] for r in results if r['best_loss'] > 0]
        axes[1].bar(range(len(names)), losses)
        axes[1].set_xticks(range(len(names)))
        axes[1].set_xticklabels(names, rotation=45, ha='right')
        axes[1].set_ylabel('Best Validation Loss')
        axes[1].set_title('最佳验证Loss对比')
        axes[1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(exp_base_dir / 'comparison.png', dpi=120, bbox_inches='tight')
        print(f"\n对比图已保存: {exp_base_dir / 'comparison.png'}")
        
    except ImportError:
        print("\n提示: 安装matplotlib可生成对比图 (pip install matplotlib)")
    except Exception as e:
        print(f"\n生成对比图错误: {e}")
    
    # 生成简单报告
    report_file = exp_base_dir / "report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 实验结果报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| 实验名称 | 轮数 | 最终Loss | 最佳验证Loss |\n")
        f.write("|---------|------|---------|-------------|\n")
        for r in results:
            loss_str = f"{r['best_loss']:.4f}" if r['best_loss'] > 0 else "N/A"
            f.write(f"| {r['name']} | {r['epochs']} | {r['final_loss']:.4f} | {loss_str} |\n")
    
    print(f"报告已保存: {report_file}\n")


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(description='Typilus 参数调优实验工具')
    parser.add_argument('--analyze', action='store_true', help='仅分析结果')
    parser.add_argument('--run-only', action='store_true', help='仅运行实验')
    parser.add_argument('--exp-dir', default='~/workspace/type_pred/naturalcc/run/type_prediction/typilus/experiments', help='实验目录')
    args = parser.parse_args()
    
    # 找到训练脚本（不再需要base_config）
    script_file = Path(__file__).resolve()
    experiment_tools_dir = script_file.parent
    typilus_dir = experiment_tools_dir.parent
    
    # 优先使用增强版脚本
    train_script = experiment_tools_dir / "train_enhanced.py"
    if not train_script.exists():
        train_script = typilus_dir / "train.py"
    
    if not train_script.exists():
        print(f"错误: 找不到训练脚本")
        print(f"尝试路径: {experiment_tools_dir / 'train_enhanced.py'}")
        print(f"备用路径: {typilus_dir / 'train.py'}")
        return
    
    if args.analyze:
        print("分析实验结果...\n")
        analyze_all_experiments(args.exp_dir)
        return
    
    if not args.run_only:
        print("="*60)
        print("Typilus 参数调优实验系统")
        print("="*60)
        print(f"\n共 {len(EXPERIMENTS)} 个实验:\n")
        for i, exp in enumerate(EXPERIMENTS, 1):
            print(f"{i}. {exp['name']}: {exp['desc']}")
        print("\n" + "="*60)
        input("\n按 Enter 开始实验...")
    
    # 运行实验
    results = []
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n进度: {i}/{len(EXPERIMENTS)}")
        success = run_single_experiment(exp, str(train_script))
        results.append((exp['name'], success))
        
        if not success:
            response = input("\n实验失败，继续? (y/n): ")
            if response.lower() != 'y':
                break
        
        if i < len(EXPERIMENTS):
            print("\n等待3秒...")
            time.sleep(3)
    
    # 打印总结
    print("\n" + "="*60)
    print("实验完成")
    print("="*60)
    for name, success in results:
        print(f"{'✓' if success else '✗'} {name}")
    print("="*60)
    
    # 自动分析（如果不是只运行模式）
    if not args.run_only:
        print("\n正在分析结果...")
        analyze_all_experiments(args.exp_dir)


if __name__ == "__main__":
    main()
