@echo off
REM Error and Warning Monitor - Shows only errors, warnings, and critical events

color 0C
title Error and Warning Monitor

:refresh
cls

echo ================================================================================
echo ERROR AND WARNING MONITOR
echo ================================================================================
echo.
echo Updated: %date% %time%
echo.

cd /d "C:\Users\J\OneDrive\CLAUDE_CODE_FIN\bingx_rl_trading_bot"

REM Find the most recent log file
for /f "delims=" %%i in ('dir /b /o-d logs\phase4_dynamic_testnet_trading_*.log 2^>nul') do (
    set logfile=logs\%%i
    goto :logfound
)
set logfile=NOT_FOUND
:logfound

if "%logfile%"=="NOT_FOUND" (
    echo [ERROR] Log file not found. Bot may not be running.
    echo.
    pause
    exit /b
)

echo Log File: %logfile%
echo.

echo ================================================================================
echo CRITICAL ERRORS (Last 20)
echo ================================================================================
echo.

powershell -Command "$errors = Get-Content '%logfile%' | Select-String 'ERROR|Exception|Failed|Traceback' | Select-Object -Last 20; if ($errors) { $errors } else { Write-Host '[OK] No critical errors found' }"

echo.
echo ================================================================================
echo WARNINGS (Last 20)
echo ================================================================================
echo.

powershell -Command "$warnings = Get-Content '%logfile%' | Select-String 'WARNING|WARN' | Select-Object -Last 20; if ($warnings) { $warnings } else { Write-Host '[OK] No warnings found' }"

echo.
echo ================================================================================
echo CONNECTION ISSUES
echo ================================================================================
echo.

powershell -Command "$connIssues = Get-Content '%logfile%' | Select-String 'connection|timeout|ConnectionError|HTTP error' | Select-Object -Last 10; if ($connIssues) { $connIssues } else { Write-Host '[OK] No connection issues detected' }"

echo.
echo ================================================================================
echo DATA ISSUES
echo ================================================================================
echo.

powershell -Command "$dataIssues = Get-Content '%logfile%' | Select-String 'Insufficient.*data|Missing data|Invalid data|NaN|null' | Select-Object -Last 10; if ($dataIssues) { $dataIssues } else { Write-Host '[OK] No data issues detected' }"

echo.
echo ================================================================================
echo TRADE EXECUTION ERRORS
echo ================================================================================
echo.

powershell -Command "$tradeErrors = Get-Content '%logfile%' | Select-String 'remains OPEN.*exception|Trade.*failed|Order.*failed' | Select-Object -Last 10; if ($tradeErrors) { $tradeErrors } else { Write-Host '[OK] No trade execution errors' }"

echo.
echo ================================================================================
echo MODEL LOADING ISSUES
echo ================================================================================
echo.

powershell -Command "$modelIssues = Get-Content '%logfile%' | Select-String 'Model.*error|Failed to load|pickle.*error' | Select-Object -Last 5; if ($modelIssues) { $modelIssues } else { Write-Host '[OK] All models loaded successfully' }"

echo.
echo ================================================================================
echo ERROR STATISTICS
echo ================================================================================
echo.

powershell -Command "$errorCount = (Get-Content '%logfile%' | Select-String 'ERROR').Count; $warningCount = (Get-Content '%logfile%' | Select-String 'WARNING').Count; $exceptionCount = (Get-Content '%logfile%' | Select-String 'Exception').Count; Write-Host \"Total Errors: $errorCount\"; Write-Host \"Total Warnings: $warningCount\"; Write-Host \"Total Exceptions: $exceptionCount\""

echo.
echo ================================================================================
echo RECENT CRITICAL EVENTS (All Severities)
echo ================================================================================
echo.

powershell -Command "Get-Content '%logfile%' | Select-String 'ERROR|WARNING|CRITICAL|Exception' | Select-Object -Last 15"

echo.
echo ================================================================================
echo HEALTH CHECK
echo ================================================================================
echo.

powershell -Command "$lastError = Get-Content '%logfile%' | Select-String 'ERROR' | Select-Object -Last 1; if ($lastError) { $match = $lastError -match '(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'; if ($match) { Write-Host \"Last Error Time: $($Matches[1])\" } else { Write-Host 'Last Error: Recent' } } else { Write-Host 'Status: [32mNo errors detected[0m' }"

powershell -Command "$lastWarning = Get-Content '%logfile%' | Select-String 'WARNING' | Select-Object -Last 1; if ($lastWarning) { $match = $lastWarning -match '(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'; if ($match) { Write-Host \"Last Warning Time: $($Matches[1])\" } else { Write-Host 'Last Warning: Recent' } } else { Write-Host 'Status: [32mNo warnings detected[0m' }"

echo.
echo ================================================================================
echo OPTIONS
echo ================================================================================
echo.
echo [R] Refresh
echo [C] Clear Screen and Continue Monitoring
echo [Q] Quit
echo.

set /p choice="Enter your choice: "

if /i "%choice%"=="R" goto refresh
if /i "%choice%"=="C" goto refresh
if /i "%choice%"=="Q" exit

goto refresh
