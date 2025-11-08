"""
백테스트 vs 프로덕션 신호 비교 (최근 7일)

백테스트가 수익을 냈는데 프로덕션이 손실을 냈다면,
신호(확률) 계산이 다를 가능성이 높음.
"""
import pandas as pd
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Paths
project_root = Path(__file__).resolve().parents[2]
results_dir = project_root / "results"
logs_dir = project_root / "logs"

# Load backtest results
df_backtest = pd.read_csv(results_dir / "backtest_current_production_full_20251103_155037.csv")
df_backtest['entry_time'] = pd.to_datetime(df_backtest['entry_time'])

# Get recent 7-day backtest trades
cutoff = df_backtest['entry_time'].max() - timedelta(days=7)
df_recent = df_backtest[df_backtest['entry_time'] >= cutoff].copy()

print("=" * 80)
print("백테스트 vs 프로덕션 신호 비교 (최근 7일)")
print("=" * 80)
print()

# Extract production signals from logs
def extract_production_signals(log_files):
    """Extract LONG/SHORT probabilities from production logs"""
    signals = []

    for log_file in log_files:
        if not log_file.exists():
            continue

        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Pattern: [Candle HH:MM:00 KST] Price: ... | LONG: 0.XXXX | SHORT: 0.XXXX
                if 'LONG:' in line and 'SHORT:' in line and 'Candle' in line:
                    try:
                        # Extract timestamp
                        time_match = re.search(r'\[Candle (\d{2}:\d{2}:\d{2}) KST\]', line)
                        if not time_match:
                            continue
                        time_str = time_match.group(1)

                        # Extract date from filename
                        date_match = re.search(r'(\d{8})\.log', str(log_file))
                        if not date_match:
                            continue
                        date_str = date_match.group(1)

                        # Combine date and time
                        dt_str = f"{date_str} {time_str}"
                        dt_kst = datetime.strptime(dt_str, "%Y%m%d %H:%M:%S")
                        dt_utc = dt_kst - timedelta(hours=9)  # KST to UTC

                        # Extract probabilities
                        long_match = re.search(r'LONG:\s+([\d.]+)', line)
                        short_match = re.search(r'SHORT:\s+([\d.]+)', line)

                        if long_match and short_match:
                            long_prob = float(long_match.group(1))
                            short_prob = float(short_match.group(1))

                            signals.append({
                                'timestamp': dt_utc,
                                'long_prob': long_prob,
                                'short_prob': short_prob
                            })
                    except Exception as e:
                        continue

    return pd.DataFrame(signals)

# Get production logs for the period
log_files = [
    logs_dir / "opportunity_gating_bot_4x_20251025.log",
    logs_dir / "opportunity_gating_bot_4x_20251026.log",
    logs_dir / "opportunity_gating_bot_4x_20251027.log",
    logs_dir / "opportunity_gating_bot_4x_20251028.log",
    logs_dir / "opportunity_gating_bot_4x_20251029.log",
    logs_dir / "opportunity_gating_bot_4x_20251030.log",
    logs_dir / "opportunity_gating_bot_4x_20251031.log",
    logs_dir / "opportunity_gating_bot_4x_20251101.log",
    logs_dir / "opportunity_gating_bot_4x_20251102.log",
]

print("프로덕션 로그 로딩...")
df_production = extract_production_signals(log_files)
print(f"   프로덕션 신호: {len(df_production)}개")
print()

# Compare signals for each backtest trade
print("=" * 80)
print("백테스트 거래별 신호 비교")
print("=" * 80)
print()

for idx, bt_row in df_recent.iterrows():
    bt_time = bt_row['entry_time']
    bt_side = bt_row['side']
    bt_long = bt_row['entry_long_prob']
    bt_short = bt_row['entry_short_prob']
    bt_pnl = bt_row['pnl_usd']

    # Find matching production signal (within 5 minutes)
    time_window = timedelta(minutes=5)
    mask = (df_production['timestamp'] >= bt_time - time_window) & \
           (df_production['timestamp'] <= bt_time + time_window)

    matching_signals = df_production[mask]

    print(f"백테스트: {bt_time.strftime('%Y-%m-%d %H:%M')} | {bt_side}")
    print(f"   LONG: {bt_long:.4f} ({bt_long*100:.1f}%) | SHORT: {bt_short:.4f} ({bt_short*100:.1f}%)")
    print(f"   P&L: ${bt_pnl:+,.0f}")

    if len(matching_signals) > 0:
        # Find closest match
        matching_signals['time_diff'] = abs((matching_signals['timestamp'] - bt_time).dt.total_seconds())
        closest = matching_signals.loc[matching_signals['time_diff'].idxmin()]

        prod_time = closest['timestamp']
        prod_long = closest['long_prob']
        prod_short = closest['short_prob']

        # Calculate differences
        long_diff = bt_long - prod_long
        short_diff = bt_short - prod_short

        print(f"프로덕션: {prod_time.strftime('%Y-%m-%d %H:%M')} (±{closest['time_diff']:.0f}초)")
        print(f"   LONG: {prod_long:.4f} ({prod_long*100:.1f}%) | SHORT: {prod_short:.4f} ({prod_short*100:.1f}%)")

        # Highlight significant differences
        if abs(long_diff) > 0.05 or abs(short_diff) > 0.05:
            print(f"   🔴 차이: LONG {long_diff:+.4f} ({long_diff*100:+.1f}%) | SHORT {short_diff:+.4f} ({short_diff*100:+.1f}%)")
        else:
            print(f"   ✅ 차이: LONG {long_diff:+.4f} ({long_diff*100:+.1f}%) | SHORT {short_diff:+.4f} ({short_diff*100:+.1f}%)")
    else:
        print(f"   ⚠️ 프로덕션 신호 없음 (±5분 범위)")

    print()

# Summary statistics
print("=" * 80)
print("통계 요약")
print("=" * 80)
print()

# Calculate average differences for matched signals
differences = []
for idx, bt_row in df_recent.iterrows():
    bt_time = bt_row['entry_time']
    time_window = timedelta(minutes=5)
    mask = (df_production['timestamp'] >= bt_time - time_window) & \
           (df_production['timestamp'] <= bt_time + time_window)

    matching_signals = df_production[mask]
    if len(matching_signals) > 0:
        matching_signals['time_diff'] = abs((matching_signals['timestamp'] - bt_time).dt.total_seconds())
        closest = matching_signals.loc[matching_signals['time_diff'].idxmin()]

        long_diff = abs(bt_row['entry_long_prob'] - closest['long_prob'])
        short_diff = abs(bt_row['entry_short_prob'] - closest['short_prob'])

        differences.append({
            'long_diff': long_diff,
            'short_diff': short_diff,
            'max_diff': max(long_diff, short_diff)
        })

if differences:
    df_diff = pd.DataFrame(differences)

    print(f"매칭된 신호: {len(differences)}개 / {len(df_recent)}개")
    print()
    print(f"평균 차이:")
    print(f"   LONG: {df_diff['long_diff'].mean():.4f} ({df_diff['long_diff'].mean()*100:.1f}%)")
    print(f"   SHORT: {df_diff['short_diff'].mean():.4f} ({df_diff['short_diff'].mean()*100:.1f}%)")
    print(f"   최대: {df_diff['max_diff'].mean():.4f} ({df_diff['max_diff'].mean()*100:.1f}%)")
    print()

    # Check if differences are significant
    significant = (df_diff['max_diff'] > 0.05).sum()
    print(f"유의미한 차이 (>5%): {significant}개 / {len(differences)}개 ({significant/len(differences)*100:.1f}%)")

    if significant > 0:
        print()
        print("🔴 결론: 백테스트와 프로덕션의 신호 계산이 다릅니다!")
        print("   → 피처 계산 방법 차이 조사 필요")
    else:
        print()
        print("✅ 결론: 백테스트와 프로덕션의 신호가 일치합니다.")
        print("   → 수익 차이는 다른 원인 (fee, slippage, 타이밍 등)")
else:
    print("⚠️ 프로덕션 신호를 찾을 수 없습니다.")
    print("   → 프로덕션 봇이 해당 기간에 실행되지 않았을 가능성")
