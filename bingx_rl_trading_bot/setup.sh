#!/bin/bash
# BingX RL Trading Bot - Setup Script
# Prepares environment for Pattern 5m trading bot

set -e  # Exit on error

echo "=========================================="
echo "BingX RL Trading Bot Setup"
echo "=========================================="
echo ""

# Check Python version
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python $PYTHON_VERSION detected"
echo ""

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created at .venv/"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Install dependencies
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
    echo "✓ Dependencies installed"
    echo ""
    echo "Installed packages:"
    .venv/bin/pip list | grep -E "ccxt|pandas|numpy|pyyaml|pytest"
else
    echo "Error: requirements.txt not found"
    exit 1
fi
echo ""

# Create .env from example
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✓ Created .env from template"
        echo ""
        echo "⚠️  CRITICAL: Edit .env with your BingX API credentials!"
        echo "   Required variables:"
        echo "   - BINGX_API_KEY"
        echo "   - BINGX_SECRET_KEY"
        echo "   - TRADING_MODE (testnet or mainnet)"
    else
        echo "Error: .env.example not found"
        exit 1
    fi
else
    echo "✓ .env already exists"
fi
echo ""

# Create necessary directories
echo "Creating directory structure..."
mkdir -p logs
mkdir -p results
mkdir -p data/historical
mkdir -p data/features
mkdir -p archive
echo "✓ Created directories:"
echo "  - logs/ (runtime logs)"
echo "  - results/ (bot state, metrics)"
echo "  - data/historical/ (raw market data)"
echo "  - data/features/ (processed indicators)"
echo "  - archive/ (deprecated code)"
echo ""

# Verify config files
echo "Verifying configuration files..."
if [ -f "config/pattern_5m_config.yaml" ]; then
    echo "✓ Pattern 5m config found"
else
    echo "⚠️  Warning: config/pattern_5m_config.yaml not found"
fi

if [ -f "config/api_keys.yaml" ]; then
    echo "✓ API keys config found"
else
    echo "⚠️  Note: config/api_keys.yaml not found (optional if using .env)"
fi
echo ""

# Run tests to verify installation
echo "Running test suite..."
if [ -f ".venv/bin/pytest" ]; then
    .venv/bin/pytest scripts/production/pattern_5m/tests/ -v --tb=short
    echo ""
    echo "✓ All tests passed"
else
    echo "⚠️  pytest not found - skipping tests"
fi
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Bot Configuration:"
echo "  Version: Pattern 5m v1.23.0"
echo "  Patterns: 10 (5 LONG + 5 SHORT)"
echo "  Win Rate: 73.9% (270-day backtest)"
echo "  Leverage: 3x"
echo ""
echo "Next Steps:"
echo "1. Activate virtual environment: source .venv/bin/activate"
echo "2. Edit .env with your BingX API credentials"
echo "3. Verify config/pattern_5m_config.yaml settings"
echo "4. Start bot: make start-pattern (or run START_PATTERN_5M.bat)"
echo "5. Monitor logs: make monitor (or run MONITOR_PATTERN_5M.bat)"
echo ""
echo "Available Commands (within venv):"
echo "  make help          - Show all available commands"
echo "  make test          - Run test suite"
echo "  make backtest      - Run validation backtest"
echo "  make start-pattern - Start Pattern 5m bot"
echo "  make stop-pattern  - Stop bot"
echo "  make monitor       - View real-time logs"
echo ""
echo "Documentation: See README.md for detailed guide"
echo ""
