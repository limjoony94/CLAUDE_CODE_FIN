#!/usr/bin/env python3
"""
Production vs Backtest Configuration Comparison
프로덕션 로그와 백테스트 스크립트의 설정 및 신호 일치도 검증
"""

import re
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import pytz

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_FILE = PROJECT_ROOT / "logs" / "opportunity_gating_bot_4x_20251017.log"
PRODUCTION_BOT = PROJECT_ROOT / "scripts" / "production" / "opportunity_gating_bot_4x.py"
BACKTEST_SCRIPT = PROJECT_ROOT / "scripts" / "experiments" / "full_backtest_opportunity_gating_4x.py"

print("=" * 80)
print("PRODUCTION vs BACKTEST CONFIGURATION COMPARISON")
print("프로덕션 로그와 백테스트 설정 일치도 검증")
print("=" * 80)

# ============================================================================
# 1. Extract Production Bot Configuration
# ============================================================================
print("\n" + "=" * 80)
print("1️⃣ PRODUCTION BOT 설정 (opportunity_gating_bot_4x.py)")
print("=" * 80)

with open(PRODUCTION_BOT, 'r', encoding='utf-8') as f:
    prod_code = f.read()

# Extract thresholds
prod_config = {}
threshold_patterns = {
    'LONG_THRESHOLD': r'LONG_THRESHOLD\s*=\s*([\d.]+)',
    'SHORT_THRESHOLD': r'SHORT_THRESHOLD\s*=\s*([\d.]+)',
    'ML_EXIT_THRESHOLD_LONG': r'ML_EXIT_THRESHOLD_LONG\s*=\s*([\d.]+)',
    'ML_EXIT_THRESHOLD_SHORT': r'ML_EXIT_THRESHOLD_SHORT\s*=\s*([\d.]+)',
    'EMERGENCY_STOP_LOSS': r'EMERGENCY_STOP_LOSS\s*=\s*([\d.]+)',
    'EMERGENCY_MAX_HOLD_TIME': r'EMERGENCY_MAX_HOLD_TIME\s*=\s*(\d+)',
    'OPPORTUNITY_COST_GATE': r'OPPORTUNITY_COST_GATE\s*=\s*([\d.]+)',
}

for name, pattern in threshold_patterns.items():
    match = re.search(pattern, prod_code)
    if match:
        prod_config[name] = float(match.group(1))

print("\n📊 Entry & Exit Thresholds:")
print(f"  LONG Entry Threshold:        {prod_config.get('LONG_THRESHOLD', 'NOT FOUND')}")
print(f"  SHORT Entry Threshold:       {prod_config.get('SHORT_THRESHOLD', 'NOT FOUND')}")
print(f"  LONG ML Exit Threshold:      {prod_config.get('ML_EXIT_THRESHOLD_LONG', 'NOT FOUND')}")
print(f"  SHORT ML Exit Threshold:     {prod_config.get('ML_EXIT_THRESHOLD_SHORT', 'NOT FOUND')}")

print("\n🛡️ Risk Management:")
print(f"  Stop Loss (% of balance):    {prod_config.get('EMERGENCY_STOP_LOSS', 'NOT FOUND') * 100:.1f}%")
print(f"  Max Hold Time (candles):     {int(prod_config.get('EMERGENCY_MAX_HOLD_TIME', 0))}")

print("\n🚪 Opportunity Gating:")
print(f"  Gate Threshold:              {prod_config.get('OPPORTUNITY_COST_GATE', 'NOT FOUND')}")

# Extract model paths
model_patterns = {
    'LONG Entry Model': r'xgboost_long_entry_enhanced_(\d{8}_\d{6})\.pkl',
    'SHORT Entry Model': r'xgboost_short_entry_enhanced_(\d{8}_\d{6})\.pkl',
    'LONG Exit Model': r'xgboost_long_exit_oppgating_improved_(\d{8}_\d{6})\.pkl',
    'SHORT Exit Model': r'xgboost_short_exit_oppgating_improved_(\d{8}_\d{6})\.pkl',
}

print("\n🤖 ML Models:")
for name, pattern in model_patterns.items():
    match = re.search(pattern, prod_code)
    if match:
        print(f"  {name}: {match.group(1)}")

# ============================================================================
# 2. Extract Backtest Script Configuration
# ============================================================================
print("\n" + "=" * 80)
print("2️⃣ BACKTEST SCRIPT 설정 (full_backtest_opportunity_gating_4x.py)")
print("=" * 80)

with open(BACKTEST_SCRIPT, 'r', encoding='utf-8') as f:
    backtest_code = f.read()

backtest_config = {}
for name, pattern in threshold_patterns.items():
    match = re.search(pattern, backtest_code)
    if match:
        backtest_config[name] = float(match.group(1))

print("\n📊 Entry & Exit Thresholds:")
print(f"  LONG Entry Threshold:        {backtest_config.get('LONG_THRESHOLD', 'NOT FOUND')}")
print(f"  SHORT Entry Threshold:       {backtest_config.get('SHORT_THRESHOLD', 'NOT FOUND')}")
print(f"  LONG ML Exit Threshold:      {backtest_config.get('ML_EXIT_THRESHOLD_LONG', 'NOT FOUND')}")
print(f"  SHORT ML Exit Threshold:     {backtest_config.get('ML_EXIT_THRESHOLD_SHORT', 'NOT FOUND')}")

print("\n🛡️ Risk Management:")
print(f"  Stop Loss (% of balance):    {backtest_config.get('EMERGENCY_STOP_LOSS', 'NOT FOUND') * 100:.1f}%")
print(f"  Max Hold Time (candles):     {int(backtest_config.get('EMERGENCY_MAX_HOLD_TIME', 0))}")

print("\n🚪 Opportunity Gating:")
print(f"  Gate Threshold:              {backtest_config.get('OPPORTUNITY_COST_GATE', 'NOT FOUND')}")

print("\n🤖 ML Models:")
for name, pattern in model_patterns.items():
    match = re.search(pattern, backtest_code)
    if match:
        print(f"  {name}: {match.group(1)}")

# ============================================================================
# 3. Configuration Comparison
# ============================================================================
print("\n" + "=" * 80)
print("3️⃣ CONFIGURATION COMPARISON (프로덕션 vs 백테스트)")
print("=" * 80)

all_params = set(prod_config.keys()) | set(backtest_config.keys())
mismatches = []

for param in sorted(all_params):
    prod_val = prod_config.get(param, 'NOT FOUND')
    back_val = backtest_config.get(param, 'NOT FOUND')

    if prod_val == back_val:
        status = "✅ 일치"
    else:
        status = "❌ 불일치"
        mismatches.append(param)

    print(f"\n{param}:")
    print(f"  Production:  {prod_val}")
    print(f"  Backtest:    {back_val}")
    print(f"  Status:      {status}")

# ============================================================================
# 4. Extract Recent Log Signals (Last 2 Hours)
# ============================================================================
print("\n" + "=" * 80)
print("4️⃣ RECENT LOG SIGNALS (최근 2시간)")
print("=" * 80)

# Parse log file
now = datetime.now(pytz.UTC)
two_hours_ago = now - timedelta(hours=2)

print(f"\n기간: {two_hours_ago.strftime('%Y-%m-%d %H:%M')} ~ {now.strftime('%Y-%m-%d %H:%M')} UTC")

signal_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Price: \$([0-9,]+\.\d+).*LONG: ([\d.]+).*SHORT: ([\d.]+)'

log_signals = []
with open(LOG_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        match = re.search(signal_pattern, line)
        if match:
            timestamp_str = match.group(1)
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            timestamp = timestamp.replace(tzinfo=pytz.UTC)

            if timestamp >= two_hours_ago:
                log_signals.append({
                    'timestamp': timestamp,
                    'price': match.group(2).replace(',', ''),
                    'long': float(match.group(3)),
                    'short': float(match.group(4))
                })

if log_signals:
    print(f"\n발견된 신호: {len(log_signals)}개")
    print(f"\nTimestamp (UTC)        Price         LONG     SHORT")
    print("-" * 60)
    for sig in log_signals[-10:]:  # Last 10
        print(f"{sig['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}  ${sig['price']:>12}  {sig['long']:.4f}  {sig['short']:.4f}")
else:
    print("\n⚠️ 최근 2시간 동안 로그 신호 없음")

# ============================================================================
# 5. Data Collection Method Comparison
# ============================================================================
print("\n" + "=" * 80)
print("5️⃣ DATA COLLECTION METHOD (데이터 수집 방식)")
print("=" * 80)

print("\n📈 Production Bot:")
print("  - Source: BingX API (실시간)")
print("  - Method: exchange.fetch_ohlcv()")
print("  - Timeframe: 5m")
print("  - Limit: 1440 candles per call")
print("  - Filtering: filter_completed_candles() - 현재 진행 중 캔들 제외")

print("\n📊 Backtest Script:")
print("  - Source: CSV file (data/historical/BTCUSDT_5m_max.csv)")
print("  - Update: Manual (python scripts/data/collect_max_data.py)")
print("  - Last Updated: 2025-10-27 03:39 KST")
print("  - Candles: 30,296 (2025-07-13 ~ 2025-10-26)")

# ============================================================================
# 6. Summary
# ============================================================================
print("\n" + "=" * 80)
print("📋 SUMMARY")
print("=" * 80)

if mismatches:
    print(f"\n❌ Configuration Mismatches: {len(mismatches)}")
    for param in mismatches:
        print(f"  - {param}")
    print("\n⚠️ ACTION REQUIRED: 백테스트 스크립트를 프로덕션 설정에 맞춰 업데이트 필요")
else:
    print("\n✅ All configurations match perfectly!")
    print("   프로덕션과 백테스트 설정이 완벽히 일치합니다.")

print("\n📊 Recent Activity:")
if log_signals:
    print(f"  Last {len(log_signals)} signals in past 2 hours")
    avg_long = sum(s['long'] for s in log_signals) / len(log_signals)
    avg_short = sum(s['short'] for s in log_signals) / len(log_signals)
    print(f"  Average LONG:  {avg_long:.4f}")
    print(f"  Average SHORT: {avg_short:.4f}")
else:
    print("  No signals in past 2 hours")

print("\n" + "=" * 80)
