# Status check for BingX L2 collector.
$projectRoot = "C:\Users\J\OneDrive\CLAUDE_CODE_FIN\bingx_rl_trading_bot"
$proc = Get-WmiObject Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -and $_.CommandLine.Contains('bingx_l2_collector.py') }

if (-not $proc) {
    Write-Host "Collector NOT running"
    exit 1
}

Write-Host "Collector RUNNING (PID $($proc.ProcessId))"
$cpu = Get-Counter "\Process(python*)\% Processor Time" -ErrorAction SilentlyContinue
Write-Host "Started: $($proc.CreationDate)"

# Recent log tail
$logPath = "$projectRoot\scripts\data_pipeline\storage\run.log"
if (Test-Path $logPath) {
    Write-Host "`n--- Recent log (last 10 lines) ---"
    Get-Content $logPath -Tail 10
}

# Storage size
$storage = "$projectRoot\scripts\data_pipeline\storage"
$today = (Get-Date -Format "yyyyMMdd")
$depthFile = "$storage\btc_depth_$today.parquet"
$tradeFile = "$storage\btc_trades_$today.parquet"
Write-Host "`n--- Storage today ($today) ---"
if (Test-Path $depthFile) {
    $sizeMB = [math]::Round((Get-Item $depthFile).Length / 1MB, 2)
    Write-Host "  Depth:  $sizeMB MB"
} else { Write-Host "  Depth:  not yet written" }
if (Test-Path $tradeFile) {
    $sizeMB = [math]::Round((Get-Item $tradeFile).Length / 1MB, 2)
    Write-Host "  Trades: $sizeMB MB"
} else { Write-Host "  Trades: not yet written" }
