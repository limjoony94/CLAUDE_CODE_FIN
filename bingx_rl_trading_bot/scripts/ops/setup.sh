#!/bin/bash
# setup.sh — venv 생성 + 의존성 설치
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

echo "=== 🔧 Setting up Python environment ==="
echo "Project: $PROJECT_ROOT"

# Create venv if not exists
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "✅ venv created at $VENV_DIR"
else
    echo "ℹ️  venv already exists at $VENV_DIR"
fi

# Activate and install
echo "📥 Installing dependencies..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install -r "$PROJECT_ROOT/requirements.txt"

echo ""
echo "✅ Setup complete!"
echo "💡 Activate: source $VENV_DIR/bin/activate"
