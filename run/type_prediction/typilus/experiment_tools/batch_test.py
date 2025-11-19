#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为 Typilus 实验批量运行测试
自动测试所有未测试的实验，并更新结果
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path
import json


def find_untested_experiments(experiments_dir):
    """查找所有未测试的实验"""
    experiments_dir = Path(experiments_dir)
    untested = []
    
    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        checkpoint_dir = exp_dir / 'checkpoints'
        test_result_file = checkpoint_dir / 'res.txt'
        checkpoint_best = checkpoint_dir / 'checkpoint_best.pt'
        
        # 检查是否有checkpoint但没有测试结果
        if checkpoint_best.exists() and not test_result_file.exists():
            untested.append({
                'name': exp_name,
                'exp_dir': str(exp_dir),
                'checkpoint_dir': str(checkpoint_dir),
                'checkpoint_path': str(checkpoint_best),
                'config_file': str(exp_dir / 'config.yml')
            })
    
    return untested


def create_test_config(exp_info, base_config_path):
    """为实验创建测试配置"""
    # 读取实验的训练配置
    if Path(exp_info['config_file']).exists():
        with open(exp_info['config_file'], 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        # 如果没有配置文件，使用基础配置
        with open(base_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # 修改为测试配置
    config['dataset']['test_subset'] = 'test'
    config['eval']['path'] = exp_info['checkpoint_path']
    config['checkpoint']['save_dir'] = exp_info['checkpoint_dir']
    
    # 创建临时测试配置文件
    # 注意: type_predict.py 会自动添加 .yml 后缀，所以这里不要加
    temp_config_path = Path(exp_info['checkpoint_dir']) / 'test_config.yml'
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return str(temp_config_path)


def run_test(exp_info, test_script_path):
    """运行单个实验的测试"""
    print("\n" + "="*80)
    print(f"测试实验: {exp_info['name']}")
    print("="*80)
    print(f"Checkpoint: {exp_info['checkpoint_path']}")
    print(f"输出目录: {exp_info['checkpoint_dir']}\n")
    
    # 创建测试配置
    base_config = Path(__file__).parent / 'config_base.yml'
    test_config = create_test_config(exp_info, base_config)
    
    # 构建测试命令
    # type_predict.py 会自动添加 .yml 后缀，所以传入时要去掉
    test_config_no_ext = str(test_config).replace('.yml', '')
    cmd = [
        sys.executable,
        str(test_script_path),
        '-f', test_config_no_ext
    ]
    
    print(f"命令: {' '.join(cmd)}\n")
    
    try:
        # 运行测试
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=str(Path(test_script_path).parent)
        )
        
        print("测试完成!")
        print(f"结果已保存: {exp_info['checkpoint_dir']}/res.txt")
        
        # 解析并显示结果
        res_file = Path(exp_info['checkpoint_dir']) / 'res.txt'
        if res_file.exists():
            print("\n测试结果:")
            with open(res_file, 'r') as f:
                content = f.read()
                print(content)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"测试失败: {e}")
        print(f"错误输出:\n{e.stderr}")
        return False
    except Exception as e:
        print(f"运行测试时出错: {e}")
        return False


def update_training_metrics(exp_info):
    """在训练日志中添加测试结果"""
    res_file = Path(exp_info['checkpoint_dir']) / 'res.txt'
    logs_dir = Path(exp_info['exp_dir']) / 'logs'
    
    if not res_file.exists() or not logs_dir.exists():
        return
    
    # 读取测试结果
    test_results = {}
    with open(res_file, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                try:
                    test_results[key.strip()] = float(value.strip())
                except:
                    pass
    
    # 更新详细日志
    detailed_log = logs_dir / 'detailed_metrics.txt'
    if detailed_log.exists():
        with open(detailed_log, 'a', encoding='utf-8') as f:
            f.write("\n\n")
            f.write("="*80 + "\n")
            f.write("TEST RESULTS\n")
            f.write("="*80 + "\n")
            for key, value in test_results.items():
                f.write(f"{key:30s}: {value}\n")
    
    # 更新 metrics.json
    metrics_file = logs_dir / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        metrics['test_results'] = test_results
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
    
    print(f"✓ 训练日志已更新: {logs_dir}")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='批量测试 Typilus 实验')
    parser.add_argument('--exp_dir', '-e', type=str,
                       default='../experiments',
                       help='实验目录路径')
    parser.add_argument('--exp', type=str,
                       help='只测试指定实验')
    parser.add_argument('--dry_run', action='store_true',
                       help='只列出需要测试的实验，不实际运行')
    
    args = parser.parse_args()
    
    # 确定路径 - 使用绝对路径避免问题
    script_dir = Path(__file__).resolve().parent
    typilus_dir = script_dir.parent
    
    exp_dir = typilus_dir / 'experiments'
    if not exp_dir.exists():
        exp_dir = Path(args.exp_dir).resolve()
    
    test_script = typilus_dir / 'type_predict.py'
    
    # 调试信息
    print(f"脚本目录: {script_dir}")
    print(f"Typilus目录: {typilus_dir}")
    print(f"测试脚本路径: {test_script}")
    print(f"测试脚本是否存在: {test_script.exists()}\n")
    
    if not test_script.exists():
        print(f"错误: 找不到测试脚本 {test_script}")
        print(f"\n可能的原因:")
        print(f"  1. 路径计算错误")
        print(f"  2. type_predict.py 文件不存在")
        print(f"\n请检查以下目录:")
        print(f"  {typilus_dir}")
        if typilus_dir.exists():
            print(f"\n该目录下的文件:")
            for f in sorted(typilus_dir.iterdir())[:10]:
                print(f"    - {f.name}")
        return
    
    print(f"实验目录: {exp_dir}")
    print(f"测试脚本: {test_script}\n")
    
    # 查找未测试的实验
    untested = find_untested_experiments(exp_dir)
    
    if not untested:
        print("所有实验都已完成测试!")
        return
    
    print(f"找到 {len(untested)} 个未测试的实验:")
    for exp in untested:
        print(f"  - {exp['name']}")
    print()
    
    # 如果指定了特定实验
    if args.exp:
        untested = [e for e in untested if e['name'] == args.exp]
        if not untested:
            print(f"错误: 找不到实验 '{args.exp}' 或该实验已测试")
            return
    
    if args.dry_run:
        print("(dry run 模式，不会实际运行测试)")
        return
    
    # 运行测试
    success_count = 0
    failed_count = 0
    
    for exp_info in untested:
        success = run_test(exp_info, test_script)
        
        if success:
            success_count += 1
            # 更新训练日志
            update_training_metrics(exp_info)
        else:
            failed_count += 1
        
        print()
    
    # 总结
    print("\n" + "="*80)
    print("测试完成汇总")
    print("="*80)
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"总计: {len(untested)}")
    
    if success_count > 0:
        print(f"\n建议运行以下命令查看更新后的分析:")
        print(f"  python analyze_results.py")


if __name__ == '__main__':
    main()
