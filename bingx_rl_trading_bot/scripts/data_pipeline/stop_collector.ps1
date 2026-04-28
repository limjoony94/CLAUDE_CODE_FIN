# Stop BingX L2 collector.
$found = Get-WmiObject Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -and $_.CommandLine.Contains('bingx_l2_collector.py') }

if (-not $found) {
    Write-Host "Collector not running"
    exit 0
}

foreach ($proc in $found) {
    Write-Host "Stopping PID $($proc.ProcessId)"
    Stop-Process -Id $proc.ProcessId -Force
}
