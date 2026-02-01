.PHONY: help start-pattern stop-pattern monitor clean test install backtest

# Default target
help:
	@echo "BingX RL Trading Bot - Available Commands"
	@echo ""
	@echo "Production:"
	@echo "  make start-pattern    Start Pattern 5m bot (background)"
	@echo "  make stop-pattern     Stop Pattern 5m bot"
	@echo "  make monitor          Monitor bot logs (real-time)"
	@echo "  make status           Check bot status"
	@echo ""
	@echo "Development:"
	@echo "  make install          Install dependencies"
	@echo "  make test             Run test suite"
	@echo "  make backtest         Run v1.23.0 validation backtest"
	@echo "  make clean            Clean temporary files"
	@echo ""
	@echo "Data:"
	@echo "  make download-data    Download latest market data"
	@echo "  make analyze-patterns Analyze pattern performance"
	@echo ""

# Production Commands
start-pattern:
	@echo "Starting Pattern 5m bot..."
	@nohup python3 scripts/production/pattern_5m_bot.py > /dev/null 2>&1 &
	@echo "Bot started. Use 'make monitor' to view logs."

stop-pattern:
	@echo "Stopping Pattern 5m bot..."
	@pkill -f "pattern_5m_bot.py" || echo "No bot process found"
	@rm -f results/pattern_5m_bot.lock
	@echo "Bot stopped."

monitor:
	@echo "Monitoring Pattern 5m bot (Ctrl+C to exit)..."
	@tail -f logs/pattern_5m_bot_$$(date +%Y%m%d).log

status:
	@echo "Checking bot status..."
	@ps aux | grep "pattern_5m_bot.py" | grep -v grep || echo "Bot not running"
	@if [ -f "results/pattern_5m_bot.lock" ]; then \
		echo "Lock file present: results/pattern_5m_bot.lock"; \
	fi

# Development Commands
install:
	@echo "Installing dependencies..."
	@pip install -r requirements.txt
	@echo "Installation complete."

test:
	@echo "Running test suite..."
	@pytest tests/ -v
	@echo "Tests complete."

test-coverage:
	@echo "Running tests with coverage..."
	@pytest --cov=scripts/production/pattern_5m --cov-report=html --cov-report=term tests/
	@echo "Coverage report generated in htmlcov/"

backtest:
	@echo "Running v1.23.0 validation backtest..."
	@python3 scripts/analysis/v1230_validation_backtest.py

# Data Commands
download-data:
	@echo "Downloading latest market data..."
	@python3 scripts/analysis/download_extended_data.py
	@echo "Data download complete."

analyze-patterns:
	@echo "Analyzing pattern performance..."
	@python3 scripts/analysis/pattern_correlation_analysis.py

# Cleanup
clean:
	@echo "Cleaning temporary files..."
	@rm -rf __pycache__
	@rm -rf scripts/__pycache__
	@rm -rf scripts/production/__pycache__
	@rm -rf scripts/production/pattern_5m/__pycache__
	@rm -rf .pytest_cache
	@rm -rf htmlcov
	@rm -rf .coverage
	@rm -f results/*.lock
	@echo "Cleanup complete."

# Research Commands
discover-patterns:
	@echo "Running pattern discovery..."
	@python3 scripts/analysis/pattern_discovery_optimized.py

validate-patterns:
	@echo "Running pattern validation..."
	@python3 scripts/analysis/production_pattern_validation.py

# Utilities
format:
	@echo "Formatting code with black..."
	@black scripts/ tests/
	@echo "Code formatted."

lint:
	@echo "Linting code..."
	@flake8 scripts/ tests/ --max-line-length=120
	@echo "Linting complete."

# Safety Commands
backup-state:
	@echo "Backing up bot state..."
	@cp results/pattern_5m_bot_state.json results/pattern_5m_bot_state.json.backup_$$(date +%Y%m%d_%H%M%S)
	@echo "State backed up."

restore-state:
	@echo "Available backups:"
	@ls -1 results/pattern_5m_bot_state.json.backup_* 2>/dev/null || echo "No backups found"
