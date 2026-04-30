# R26 Watchdog — Register/Unregister with Windows Task Scheduler (schtasks-based)
# Uses native schtasks.exe for compatibility.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File r26_watchdog_register.ps1
#   powershell -ExecutionPolicy Bypass -File r26_watchdog_register.ps1 -Action Unregister
#   powershell -ExecutionPolicy Bypass -File r26_watchdog_register.ps1 -Action Status

param(
    [ValidateSet('Register', 'Unregister', 'Status')]
    [string]$Action = 'Register'
)

$ErrorActionPreference = 'Continue'
$TaskName = 'R26GridBotWatchdog'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$watchdogScript = Join-Path $scriptDir 'r26_watchdog.ps1'
$silentLauncher = Join-Path $scriptDir 'r26_watchdog_silent.vbs'

if ($Action -eq 'Status') {
    & schtasks /Query /TN $TaskName /V /FO LIST 2>&1 | Out-Host
    exit 0
}

if ($Action -eq 'Unregister') {
    & schtasks /Delete /TN $TaskName /F 2>&1 | Out-Host
    exit 0
}

# Register: every 5 min using schtasks via SILENT VBS launcher (no window popup)
# wscript.exe runs .vbs without console window, .vbs launches PowerShell with vbHide
$cmd = "wscript.exe `"$silentLauncher`""

Write-Host "Registering scheduled task '$TaskName' (silent VBS wrapper)..."
Write-Host "  VBS launcher: $silentLauncher"
Write-Host "  Watchdog logic: $watchdogScript"
Write-Host "  Interval: every 5 min"
Write-Host "  Command: $cmd"
Write-Host ""

# Delete first if exists
& schtasks /Delete /TN $TaskName /F 2>&1 | Out-Null

# Create
$result = & schtasks /Create /TN $TaskName /TR $cmd /SC MINUTE /MO 5 /F 2>&1
Write-Host $result

Write-Host ""
Write-Host "Verify: powershell -File r26_watchdog_register.ps1 -Action Status"
Write-Host "Logs: logs/r26_watchdog.log"
