#!/bin/bash
# CLAUDE_CODE_FIN - Workspace Setup Script
# Sets up Python environment and trading bot infrastructure

set -e  # Exit on error

echo "=========================================="
echo "CLAUDE_CODE_FIN Workspace Setup"
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

# Navigate to bot directory
cd bingx_rl_trading_bot

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
if [ -f "requirements/dev.txt" ]; then
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements/dev.txt
    echo "✓ Dependencies installed (development + runtime)"
else
    echo "Warning: requirements/dev.txt not found"
fi
echo ""

# Create .env from example
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✓ Created .env from template"
        echo "⚠️  IMPORTANT: Edit .env with your BingX API keys!"
    else
        echo "Warning: .env.example not found"
    fi
else
    echo "✓ .env already exists"
fi
echo ""

# Create necessary directories
mkdir -p logs results data/historical archive
echo "✓ Created directory structure:"
echo "  - logs/ (bot runtime logs)"
echo "  - results/ (bot state and metrics)"
echo "  - data/historical/ (market data)"
echo "  - archive/ (deprecated code)"
echo ""

# Create config from example
if [ ! -f "config/api_keys.yaml" ]; then
    echo "⚠️  Note: config/api_keys.yaml not found"
    echo "   Create it from config/api_keys.yaml.example if needed"
fi
echo ""

# Return to root
cd ..

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next Steps:"
echo "1. Activate virtual environment: source bingx_rl_trading_bot/.venv/bin/activate"
echo "2. Edit bingx_rl_trading_bot/.env with your API keys"
echo "3. Configure config/api_keys.yaml if using YAML config"
echo "4. Run: make start-pattern (to start Pattern 5m bot)"
echo "5. Run: make monitor (to view logs)"
echo ""
echo "For more commands: make help"
echo ""
