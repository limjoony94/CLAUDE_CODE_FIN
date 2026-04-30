# R26 Grid Bot Watchdog
# ----------------------
# Checks if r26_grid_bot.py is running. If not, starts it.
# Should be scheduled by Windows Task Scheduler every 5 min.
#
# The bot's own lock file (PID-based) prevents duplicate launches.
#
# Usage:
#   Manual: powershell -File scripts/ops/r26_watchdog.ps1
#   Scheduled: see scripts/ops/r26_watchdog_register.ps1

$ErrorActionPreference = 'Continue'

# Determine project root (parent of parent of this script)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)

$botScript = Join-Path $projectRoot 'scripts\production\r26_grid_bot.py'
$logFile = Join-Path $projectRoot 'logs\r26_watchdog.log'

# Ensure logs dir exists
$logDir = Split-Path -Parent $logFile
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

function Write-WatchdogLog {
    param([string]$Message, [string]$Level = 'INFO')
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $line = "$ts [$Level] $Message"
    Write-Output $line
    try {
        Add-Content -Path $logFile -Value $line -ErrorAction SilentlyContinue
    } catch { }
}

# Find existing bot process (matches CommandLine containing r26_grid_bot)
$botProcess = Get-WmiObject Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like '*r26_grid_bot*' }

if ($botProcess) {
    Write-WatchdogLog "Bot alive (PID: $($botProcess.ProcessId))"
    exit 0
}

# Bot not found — launch it
Write-WatchdogLog "Bot NOT running. Launching..." 'WARN'

# Verify bot script exists
if (-not (Test-Path $botScript)) {
    Write-WatchdogLog "Bot script not found at: $botScript" 'ERROR'
    exit 1
}

# Launch in background
try {
    $proc = Start-Process -FilePath 'python' `
        -ArgumentList $botScript `
        -WindowStyle Hidden `
        -WorkingDirectory $projectRoot `
        -PassThru
    Write-WatchdogLog "Bot launched (PID: $($proc.Id))"
    exit 0
} catch {
    Write-WatchdogLog "Launch failed: $_" 'ERROR'
    exit 1
}
