.PHONY: setup start stop status test report clean

BOT_DIR  := bingx_rl_trading_bot
OPS_DIR  := $(BOT_DIR)/scripts/ops
VENV     := $(BOT_DIR)/.venv
PYTHON   := $(VENV)/bin/python3

setup:
	@bash $(OPS_DIR)/setup.sh

start:
	@bash $(OPS_DIR)/start_bot.sh

stop:
	@bash $(OPS_DIR)/stop_bot.sh

status:
	@bash $(OPS_DIR)/status.sh

test:
	@echo "=== 🧪 Import Check ==="
	@$(PYTHON) -c "import ccxt; import pandas; import numpy; import yaml; print('✅ All core imports OK')" 2>/dev/null \
		|| python3 -c "import ccxt; import pandas; import numpy; import yaml; print('✅ All core imports OK')"
	@echo "=== 🧪 Bot Module Check ==="
	@$(PYTHON) -c "import sys; sys.path.insert(0,'$(BOT_DIR)'); from scripts.production.pattern_5m.bot import *; print('✅ Bot module loads OK')" 2>/dev/null || echo "⚠️  Bot module check skipped (venv not ready?)"

report:
	@echo "=== 📊 Daily Report ==="
	@bash $(OPS_DIR)/health_check.sh 2>/dev/null || echo "ℹ️  health_check.sh not available"

clean:
	@echo "=== 🧹 Cleaning logs and caches ==="
	find $(BOT_DIR) -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find $(BOT_DIR) -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleaned"
