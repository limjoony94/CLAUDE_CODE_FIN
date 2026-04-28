# Start BingX L2 collector as background process (Windows).
# Usage: powershell -ExecutionPolicy Bypass -File scripts/data_pipeline/start_collector.ps1
# Stop:  powershell -ExecutionPolicy Bypass -File scripts/data_pipeline/stop_collector.ps1

$projectRoot = "C:\Users\J\OneDrive\CLAUDE_CODE_FIN\bingx_rl_trading_bot"
$scriptPath  = "$projectRoot\scripts\data_pipeline\bingx_l2_collector.py"

# Check if already running
$existing = Get-WmiObject Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -and $_.CommandLine.Contains('bingx_l2_collector.py') }

if ($existing) {
    Write-Host "Collector already running (PID $($existing.ProcessId))"
    exit 0
}

# Start hidden process, capture stdout/stderr to log
Start-Process -FilePath 'python' `
    -ArgumentList "`"$scriptPath`"" `
    -WindowStyle Hidden `
    -WorkingDirectory $projectRoot

Start-Sleep -Seconds 2

$started = Get-WmiObject Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -and $_.CommandLine.Contains('bingx_l2_collector.py') }

if ($started) {
    Write-Host "Collector started (PID $($started.ProcessId))"
    Write-Host "Logs: $projectRoot\scripts\data_pipeline\storage\run.log"
} else {
    Write-Host "ERROR: collector failed to start"
    exit 1
}
