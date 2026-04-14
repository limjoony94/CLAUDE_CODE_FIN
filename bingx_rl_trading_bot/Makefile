.PHONY: help start stop monitor status test clean backup-state restore-state

# Default target
help:
	@echo "C1 Breakout v2.6 — Available Commands"
	@echo ""
	@echo "Production:"
	@echo "  make start            Start C1 Breakout bot (Windows background)"
	@echo "  make stop             Stop C1 Breakout bot"
	@echo "  make monitor          Monitor bot logs (real-time)"
	@echo "  make status           Check bot status"
	@echo ""
	@echo "Development:"
	@echo "  make test             Verify module imports"
	@echo "  make clean            Clean temporary files"
	@echo ""
	@echo "Safety:"
	@echo "  make backup-state     Backup bot state"
	@echo "  make restore-state    List available backups"
	@echo ""

# Production Commands (Windows)
start:
	@echo "Starting C1 Breakout bot..."
	powershell -Command "Start-Process -FilePath 'python' -ArgumentList 'scripts/production/c1_breakout_bot.py' -WindowStyle Hidden -WorkingDirectory '$(CURDIR)'"
	@echo "Bot started. Use 'make monitor' to view logs."

stop:
	@echo "Stopping C1 Breakout bot..."
	powershell -Command "Get-WmiObject Win32_Process -Filter \"Name='python.exe' AND CommandLine LIKE '%%c1_breakout%%'\" | ForEach-Object { Stop-Process -Id $$_.ProcessId -Force }"
	@echo "Bot stopped."

monitor:
	@echo "Monitoring C1 Breakout bot (Ctrl+C to exit)..."
	@tail -f logs/c1_breakout.log

status:
	@echo "Checking C1 Breakout bot status..."
	@powershell -Command "Get-WmiObject Win32_Process -Filter \"Name='python.exe' AND CommandLine LIKE '%%c1_breakout%%'\" | Select-Object ProcessId, CreationDate" || echo "Bot not running"
	@echo ""
	@echo "Latest state:"
	@python -c "import json; d=json.load(open('results/c1_breakout_state.json')); print(f\"Positions: {len(d.get('positions',[]))}, Trades: {len(d.get('trade_history',[]))}, Updated: {d.get('updated','?')}\")" 2>/dev/null || echo "No state file"

# Development
test:
	@echo "Verifying C1 Breakout module imports..."
	@python -c "from scripts.production.c1_breakout import bot, signals, indicators, config; print('All modules import OK')"
	@python -c "from scripts.production.c1_breakout.config import load_config; c = load_config(); print(f'Config loaded: {len(c)} sections')"

# Cleanup
clean:
	@echo "Cleaning temporary files..."
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache htmlcov .coverage
	@echo "Cleanup complete."

# Safety
backup-state:
	@echo "Backing up C1 Breakout state..."
	@cp results/c1_breakout_state.json "results/c1_breakout_state_backup_$$(date +%Y%m%d_%H%M%S).json"
	@echo "State backed up."

restore-state:
	@echo "Available backups:"
	@ls -1 results/c1_breakout_state_backup_* 2>/dev/null || echo "No backups found"
