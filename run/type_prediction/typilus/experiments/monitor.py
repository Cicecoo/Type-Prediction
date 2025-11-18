#!/usr/bin/env python3
"""
实时监控脚本
监控正在运行的实验，显示实时进度
"""

import time
import subprocess
from pathlib import Path
from datetime import datetime
import re


class ExperimentMonitor:
    """实验监控器"""
    
    def __init__(self, log_file: Path, refresh_interval: int = 5):
        self.log_file = log_file
        self.refresh_interval = refresh_interval
        self.last_position = 0
        self.current_epoch = 0
        self.current_loss = 0.0
        self.steps = 0
        
    def check_gpu_status(self):
        """检查GPU使用情况（可选）"""
        # 在受限服务器上可能无法执行nvidia-smi，因此简化处理
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                                       '--format=csv,noheader,nounits'],
                                      capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return "GPU信息不可用（需要nvidia-smi权限）"
    
    def read_new_lines(self):
        """读取日志文件的新内容"""
        if not self.log_file.exists():
            return []
        
        try:
            with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(self.last_position)
                new_lines = f.readlines()
                self.last_position = f.tell()
                return new_lines
        except:
            return []
    
    def parse_line(self, line: str):
        """解析日志行，提取关键信息"""
        # 提取epoch
        epoch_match = re.search(r'epoch[:\s]+(\d+)', line, re.IGNORECASE)
        if epoch_match:
            self.current_epoch = int(epoch_match.group(1))
        
        # 提取loss
        loss_match = re.search(r'loss[:\s=]+([0-9.]+)', line, re.IGNORECASE)
        if loss_match:
            self.current_loss = float(loss_match.group(1))
            self.steps += 1
    
    def display_status(self):
        """显示当前状态"""
        # 清屏（跨平台）
        print("\033[2J\033[H", end='')
        
        print("="*80)
        print(f"实验监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        print(f"\n日志文件: {self.log_file}")
        print(f"刷新间隔: {self.refresh_interval}秒")
        print("\n" + "-"*80)
        
        # 训练状态
        print(f"\n📊 训练状态:")
        print(f"  当前Epoch: {self.current_epoch}")
        print(f"  训练步数: {self.steps}")
        print(f"  当前Loss: {self.current_loss:.4f}")
        
        # GPU状态
        print(f"\n🖥️  GPU状态:")
        gpu_status = self.check_gpu_status()
        for line in gpu_status.split('\n'):
            print(f"  {line}")
        
        # 文件大小
        if self.log_file.exists():
            size_mb = self.log_file.stat().st_size / (1024 * 1024)
            print(f"\n📝 日志文件大小: {size_mb:.2f} MB")
        
        print("\n" + "-"*80)
        print("按 Ctrl+C 停止监控")
        print("="*80)
    
    def monitor(self):
        """开始监控"""
        print(f"开始监控: {self.log_file}")
        print(f"等待日志文件...")
        
        # 等待日志文件出现
        while not self.log_file.exists():
            time.sleep(1)
        
        print("检测到日志文件，开始监控...")
        
        try:
            while True:
                new_lines = self.read_new_lines()
                
                for line in new_lines:
                    self.parse_line(line)
                    
                    # 检查是否完成
                    if 'training complete' in line.lower() or 'finished' in line.lower():
                        print("\n\n🎉 训练完成!")
                        return
                
                self.display_status()
                time.sleep(self.refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\n监控已停止")


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python monitor.py <实验名称> [刷新间隔(秒)]")
        print("\n示例:")
        print("  python monitor.py exp_lr_2e4")
        print("  python monitor.py exp_lr_2e4 10")
        return
    
    exp_name = sys.argv[1]
    refresh = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    # 查找日志文件
    log_file = Path(__file__).parent.parent.parent.parent.parent / 'screen' / f'log_{exp_name}.txt'
    
    if not log_file.exists():
        log_file = Path(__file__).parent.parent.parent.parent.parent / 'screen' / f'naturalcc_train_{exp_name}.txt'
    
    if not log_file.exists():
        # 尝试其他可能的路径
        possible_paths = [
            Path(f'./screen/log_{exp_name}.txt'),
            Path(f'./screen/naturalcc_train_{exp_name}.txt'),
            Path(f'../screen/log_{exp_name}.txt'),
        ]
        
        for path in possible_paths:
            if path.exists():
                log_file = path
                break
        else:
            print(f"错误: 找不到日志文件 log_{exp_name}.txt")
            print(f"尝试过的路径: {log_file}")
            return
    
    monitor = ExperimentMonitor(log_file, refresh)
    monitor.monitor()


if __name__ == "__main__":
    main()
