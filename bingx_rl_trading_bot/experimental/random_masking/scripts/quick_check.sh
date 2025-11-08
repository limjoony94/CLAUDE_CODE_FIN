#!/bin/bash
# Quick Environment Check Script
# Usage: bash scripts/quick_check.sh

echo "========================================"
echo "RANDOM MASKING EXPERIMENT - QUICK CHECK"
echo "========================================"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check functions
check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

echo ""
echo "=== 1. GPU Check ==="
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
        check_pass "GPU Available: $GPU_NAME ($GPU_MEM)"
    else
        check_fail "nvidia-smi failed to execute"
    fi
else
    check_fail "nvidia-smi not found (GPU required for experiments)"
fi

echo ""
echo "=== 2. Disk Space Check ==="
AVAILABLE_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -ge 5 ]; then
    check_pass "Disk space: ${AVAILABLE_GB}GB available (need 5GB minimum)"
else
    check_fail "Disk space: ${AVAILABLE_GB}GB available (need 5GB minimum)"
fi

echo ""
echo "=== 3. Python Environment ==="
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    check_pass "Python: $PYTHON_VERSION"
else
    check_fail "Python not found"
fi

echo ""
echo "=== 4. Required Packages ==="
PACKAGES=("torch" "numpy" "pandas" "ccxt" "loguru" "pyyaml")
for pkg in "${PACKAGES[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        VERSION=$(python -c "import $pkg; print($pkg.__version__)" 2>/dev/null)
        check_pass "$pkg: $VERSION"
    else
        check_fail "$pkg: not installed"
    fi
done

echo ""
echo "=== 5. Configuration Files ==="
CONFIGS=("configs/baseline_config.yaml" "configs/proposed_config.yaml" "configs/variant_infill_heavy.yaml" "configs/variant_forecast_heavy.yaml")
for config in "${CONFIGS[@]}"; do
    if [ -f "$config" ]; then
        check_pass "$config exists"
    else
        check_fail "$config not found"
    fi
done

echo ""
echo "=== 6. Execution Scripts ==="
SCRIPTS=("main.py" "experiments/run_ablation.py" "experiments/statistical_test.py")
for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        check_pass "$script exists"
    else
        check_fail "$script not found"
    fi
done

echo ""
echo "=== 7. Directory Structure ==="
DIRS=("data/raw" "results/experiments" "logs" "runs")
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        check_pass "$dir/ exists"
    else
        check_warn "$dir/ not found (will be created automatically)"
    fi
done

echo ""
echo "=== 8. Data Status ==="
if [ -f "data/raw/crypto_candles.parquet" ]; then
    FILE_SIZE=$(du -h data/raw/crypto_candles.parquet | awk '{print $1}')
    check_pass "Data collected: crypto_candles.parquet ($FILE_SIZE)"
else
    check_warn "Data not collected yet (run data collection first)"
fi

echo ""
echo "========================================"
echo "CHECK COMPLETE"
echo "========================================"

# Final recommendation
echo ""
if [ "$AVAILABLE_GB" -ge 5 ] && command -v nvidia-smi &> /dev/null && [ -f "configs/baseline_config.yaml" ]; then
    echo -e "${GREEN}✓ System is READY for experiments!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Collect data: python -m random_masking.data.collector --symbols BTCUSDT ETHUSDT --start 2022-01-01 --end 2024-09-30 --interval 1m"
    echo "  2. Train baseline: python main.py --config configs/baseline_config.yaml"
    echo "  3. Run ablation: python experiments/run_ablation.py"
else
    echo -e "${RED}⚠ System has missing requirements${NC}"
    echo ""
    echo "Fix the issues above before starting experiments"
fi
echo ""
