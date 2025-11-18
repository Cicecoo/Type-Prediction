# Typilus 实验工具 - 快速启动脚本

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Typilus 参数调优实验工具" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "选择操作:" -ForegroundColor Yellow
Write-Host "1. 运行所有实验（自动化）"
Write-Host "2. 分析已有结果"
Write-Host "3. 查看帮助"
Write-Host "0. 退出`n"

$choice = Read-Host "请输入 [0-3]"

switch ($choice) {
    1 {
        Write-Host "`n启动自动化实验..." -ForegroundColor Green
        python run_experiments.py
    }
    2 {
        Write-Host "`n分析结果..." -ForegroundColor Green
        python run_experiments.py --analyze
    }
    3 {
        Get-Content README.md
    }
    0 {
        Write-Host "退出" -ForegroundColor Cyan
    }
    default {
        Write-Host "无效选项" -ForegroundColor Red
    }
}
