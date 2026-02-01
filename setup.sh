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

# Install dependencies
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "Warning: requirements.txt not found"
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
echo "1. Edit bingx_rl_trading_bot/.env with your API keys"
echo "2. Configure config/api_keys.yaml if using YAML config"
echo "3. Run: make start-pattern (to start Pattern 5m bot)"
echo "4. Run: make monitor (to view logs)"
echo ""
echo "For more commands: make help"
echo ""
