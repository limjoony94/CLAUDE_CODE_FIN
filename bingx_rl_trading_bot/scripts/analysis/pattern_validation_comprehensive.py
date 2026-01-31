"""
Comprehensive Pattern Validation
- Counter-trend analysis (pattern alpha vs market beta)
- Regime-specific performance
- Logical consistency check
- Final recommendations
"""

import pandas as pd
import numpy as np
from datetime import datetime

print('=' * 70)
print('Comprehensive Pattern Validation')
print('=' * 70)

# Top candidates from discovery
LONG_CANDIDATES = [
    ('DN-DN-DN', 1.0, 3.0),
    ('DN-U-U', 1.0, 3.0),
    ('U-DN-DN', 1.0, 3.0),
    ('DN-DN-U', 1.0, 3.0),
    ('ST-DN-DN', 1.0, 3.0),
    ('DN-ST-U', 1.0, 3.0),
    ('U-ST-U', 1.0, 3.0),
    ('U-U-U', 1.5, 3.0),
]

SHORT_CANDIDATES = [
    ('DN-DN-DN', 2.0, 3.0),
    ('U-DN-DN', 2.5, 3.0),
    ('U-U-BU', 1.0, 3.0),
    ('DN-U-DN', 2.0, 3.0),
    ('DN-DN-ST', 1.5, 3.0),
    ('U-U-DN', 2.0, 3.0),
]

# Current production patterns for comparison
PRODUCTION_LONG = ['U-BU-U', 'ST-BD-DN']
PRODUCTION_SHORT = ['BD-BD-BD', 'DN-DN-BD', 'MU-ST-DN', 'IH-DN-DN', 'BD-ST-DN', 'BU-U-DN', 'D-DN-BD']

LEVERAGE = 3
FEE_PCT = 0.10

# Load and prepare data
df = pd.read_csv('data/btc_5m_90days_validation.csv')
print(f'Dataset: {len(df)} bars')
print(f'Price: ${df.iloc[0]["close"]:.0f} → ${df.iloc[-1]["close"]:.0f}')
price_change = (df.iloc[-1]['close'] / df.iloc[0]['close'] - 1) * 100
print(f'Overall change: {price_change:.1f}%')

# Classify candles (vectorized)
df['body'] = df['close'] - df['open']
df['body_abs'] = df['body'].abs()
df['range'] = df['high'] - df['low']
df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
df['avg_body'] = df['body_abs'].rolling(20).mean().fillna(1.0)  # default 1.0 for early bars: preserves range-based classification
df['norm_body'] = df['body_abs'] / df['avg_body'].replace(0, 1)
df['body_ratio'] = df['body_abs'] / df['range'].replace(0, 1)
df['upper_ratio'] = df['upper_wick'] / df['range'].replace(0, 1)
df['lower_ratio'] = df['lower_wick'] / df['range'].replace(0, 1)
df['wick_ratio'] = (df['upper_wick'] + df['lower_wick']) / df['range'].replace(0, 1)

is_bullish = df['body'] > 0
small_range = df['range'] < 1e-10
df['candle_type'] = np.where(is_bullish, 'U', 'DN')

doji_mask = df['body_ratio'] < 0.10
df.loc[doji_mask & (df['lower_ratio'] > 0.70), 'candle_type'] = 'DF'
df.loc[doji_mask & (df['upper_ratio'] > 0.70) & (df['lower_ratio'] <= 0.70), 'candle_type'] = 'GS'
df.loc[doji_mask & (df['lower_ratio'] <= 0.70) & (df['upper_ratio'] <= 0.70), 'candle_type'] = 'D'

marubozu_mask = (~doji_mask) & (df['wick_ratio'] < 0.15)
df.loc[marubozu_mask & is_bullish, 'candle_type'] = 'MU'
df.loc[marubozu_mask & ~is_bullish, 'candle_type'] = 'MD'

hammer_mask = (~doji_mask) & (~marubozu_mask) & (df['body_abs'] > 0)
h_mask = hammer_mask & (df['lower_wick'] > 2.0 * df['body_abs']) & (df['upper_wick'] < 0.3 * df['body_abs'])
ih_mask = hammer_mask & (df['upper_wick'] > 2.0 * df['body_abs']) & (df['lower_wick'] < 0.3 * df['body_abs'])
df.loc[h_mask, 'candle_type'] = 'H'
df.loc[ih_mask, 'candle_type'] = 'IH'

st_mask = (~doji_mask) & (~marubozu_mask) & (~h_mask) & (~ih_mask) & (df['norm_body'] < 0.5)
st_mask = st_mask & (df['upper_wick'] > 0.5 * df['body_abs']) & (df['lower_wick'] > 0.5 * df['body_abs'])
df.loc[st_mask, 'candle_type'] = 'ST'

big_mask = (~doji_mask) & (~marubozu_mask) & (~h_mask) & (~ih_mask) & (~st_mask) & (df['norm_body'] > 1.5)
df.loc[big_mask & is_bullish, 'candle_type'] = 'BU'
df.loc[big_mask & ~is_bullish, 'candle_type'] = 'BD'
df.loc[small_range, 'candle_type'] = 'D'

df['pattern'] = df['candle_type'].shift(2) + '-' + df['candle_type'].shift(1) + '-' + df['candle_type']

# Add regime indicators
df['ema20'] = df['close'].ewm(span=20).mean()
df['ema50'] = df['close'].ewm(span=50).mean()
df['price_change_20'] = df['close'].pct_change(20) * 100

# Regime classification
def classify_regime(row):
    if pd.isna(row['price_change_20']):
        return 'UNKNOWN'
    if row['price_change_20'] > 2:
        return 'BULL'
    elif row['price_change_20'] < -2:
        return 'BEAR'
    else:
        return 'SIDE'

df['regime'] = df.apply(classify_regime, axis=1)

# Check regime distribution
regime_counts = df['regime'].value_counts()
print('\nRegime distribution:')
for regime, count in regime_counts.items():
    print(f'  {regime}: {count} bars ({count/len(df)*100:.1f}%)')

def backtest_with_regime(pattern, direction, tp_pct, sl_pct, regime_filter=None):
    """Backtest with optional regime filter."""
    if regime_filter:
        pattern_df = df[(df['pattern'] == pattern) & (df['regime'] == regime_filter)]
    else:
        pattern_df = df[df['pattern'] == pattern]

    results = []
    for idx in pattern_df.index:
        pos = df.index.get_loc(idx)
        if pos + 1 >= len(df):
            continue

        entry = df.iloc[pos + 1]['open']

        for i in range(pos + 1, min(pos + 50, len(df))):
            bar = df.iloc[i]
            if direction == 'LONG':
                tp_price = entry * (1 + tp_pct/100)
                sl_price = entry * (1 - sl_pct/100)
                if bar['high'] >= tp_price:
                    results.append(('WIN', tp_pct * LEVERAGE - FEE_PCT))
                    break
                elif bar['low'] <= sl_price:
                    results.append(('LOSS', -sl_pct * LEVERAGE - FEE_PCT))
                    break
            else:
                tp_price = entry * (1 - tp_pct/100)
                sl_price = entry * (1 + sl_pct/100)
                if bar['low'] <= tp_price:
                    results.append(('WIN', tp_pct * LEVERAGE - FEE_PCT))
                    break
                elif bar['high'] >= sl_price:
                    results.append(('LOSS', -sl_pct * LEVERAGE - FEE_PCT))
                    break

    if not results:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'edge': 0}

    trades = len(results)
    wins = sum(1 for r in results if r[0] == 'WIN')
    wr = wins / trades * 100
    pnl = sum(r[1] for r in results)
    edge = pnl / trades

    return {'trades': trades, 'wr': wr, 'pnl': pnl, 'edge': edge}

def walk_forward(pattern, direction, tp, sl, n_folds=5):
    """Walk-forward validation."""
    fold_size = len(df) // n_folds
    profitable = 0

    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else len(df)
        fold_df = df.iloc[start:end]

        pattern_signals = fold_df[fold_df['pattern'] == pattern]
        results = []

        for idx in pattern_signals.index:
            pos = fold_df.index.get_loc(idx)
            if pos + 1 >= len(fold_df):
                continue
            entry = fold_df.iloc[pos + 1]['open']

            for j in range(pos + 1, min(pos + 50, len(fold_df))):
                bar = fold_df.iloc[j]
                if direction == 'LONG':
                    if bar['high'] >= entry * (1 + tp/100):
                        results.append('WIN')
                        break
                    elif bar['low'] <= entry * (1 - sl/100):
                        results.append('LOSS')
                        break
                else:
                    if bar['low'] <= entry * (1 - tp/100):
                        results.append('WIN')
                        break
                    elif bar['high'] >= entry * (1 + sl/100):
                        results.append('LOSS')
                        break

        if results:
            wins = sum(1 for r in results if r == 'WIN')
            pnl = wins * (tp * LEVERAGE - FEE_PCT) + (len(results) - wins) * (-sl * LEVERAGE - FEE_PCT)
            if pnl > 0:
                profitable += 1

    return profitable

# ============================================================
# VALIDATION ANALYSIS
# ============================================================

print('\n' + '=' * 70)
print('VALIDATION ANALYSIS')
print('=' * 70)

validation_results = []

# Test LONG candidates
print('\n--- LONG Pattern Candidates ---')
for pattern, tp, sl in LONG_CANDIDATES:
    # Overall performance
    overall = backtest_with_regime(pattern, 'LONG', tp, sl)

    # Counter-trend test (LONG in BEAR market = trading against trend)
    bear = backtest_with_regime(pattern, 'LONG', tp, sl, 'BEAR')
    bull = backtest_with_regime(pattern, 'LONG', tp, sl, 'BULL')
    side = backtest_with_regime(pattern, 'LONG', tp, sl, 'SIDE')

    # Walk-forward
    wf = walk_forward(pattern, 'LONG', tp, sl)

    # Counter-trend alpha = performance in opposite regime
    # For LONG, counter-trend is BEAR market
    counter_trend_wr = bear['wr'] if bear['trades'] >= 5 else None

    validation_results.append({
        'pattern': pattern,
        'direction': 'LONG',
        'tp': tp,
        'sl': sl,
        'trades': overall['trades'],
        'wr': overall['wr'],
        'pnl': overall['pnl'],
        'wf': wf,
        'bear_trades': bear['trades'],
        'bear_wr': bear['wr'],
        'bull_trades': bull['trades'],
        'bull_wr': bull['wr'],
        'side_trades': side['trades'],
        'side_wr': side['wr'],
        'counter_trend_wr': counter_trend_wr,
    })

    ct_str = f"{counter_trend_wr:.1f}%" if counter_trend_wr else "N/A"
    print(f"\n{pattern} LONG:")
    print(f"  Overall: {overall['trades']} trades, {overall['wr']:.1f}% WR, WF {wf}/5")
    print(f"  BEAR: {bear['trades']} trades, {bear['wr']:.1f}% WR (counter-trend)")
    print(f"  BULL: {bull['trades']} trades, {bull['wr']:.1f}% WR")
    print(f"  SIDE: {side['trades']} trades, {side['wr']:.1f}% WR")

# Test SHORT candidates
print('\n--- SHORT Pattern Candidates ---')
for pattern, tp, sl in SHORT_CANDIDATES:
    overall = backtest_with_regime(pattern, 'SHORT', tp, sl)
    bear = backtest_with_regime(pattern, 'SHORT', tp, sl, 'BEAR')
    bull = backtest_with_regime(pattern, 'SHORT', tp, sl, 'BULL')
    side = backtest_with_regime(pattern, 'SHORT', tp, sl, 'SIDE')
    wf = walk_forward(pattern, 'SHORT', tp, sl)

    # For SHORT, counter-trend is BULL market
    counter_trend_wr = bull['wr'] if bull['trades'] >= 5 else None

    validation_results.append({
        'pattern': pattern,
        'direction': 'SHORT',
        'tp': tp,
        'sl': sl,
        'trades': overall['trades'],
        'wr': overall['wr'],
        'pnl': overall['pnl'],
        'wf': wf,
        'bear_trades': bear['trades'],
        'bear_wr': bear['wr'],
        'bull_trades': bull['trades'],
        'bull_wr': bull['wr'],
        'side_trades': side['trades'],
        'side_wr': side['wr'],
        'counter_trend_wr': counter_trend_wr,
    })

    ct_str = f"{counter_trend_wr:.1f}%" if counter_trend_wr else "N/A"
    print(f"\n{pattern} SHORT:")
    print(f"  Overall: {overall['trades']} trades, {overall['wr']:.1f}% WR, WF {wf}/5")
    print(f"  BEAR: {bear['trades']} trades, {bear['wr']:.1f}% WR")
    print(f"  BULL: {bull['trades']} trades, {bull['wr']:.1f}% WR (counter-trend)")
    print(f"  SIDE: {side['trades']} trades, {side['wr']:.1f}% WR")

# ============================================================
# LOGICAL ANALYSIS
# ============================================================

print('\n' + '=' * 70)
print('LOGICAL ANALYSIS')
print('=' * 70)

print("""
패턴의 논리적 의미 분석:

LONG 패턴들 (하락 후 반등 기대):
- DN-DN-DN: 3연속 하락 → 과매도 반등 기대 (Mean Reversion)
- DN-U-U: 하락 후 2연속 상승 → 반등 확인 후 진입 (Trend Confirmation)
- U-DN-DN: 상승 후 2연속 하락 → 조정 완료 기대 (Pullback)
- ST-DN-DN: 횡보 후 2연속 하락 → 과매도 반등 (Support Test)
- U-U-U: 3연속 상승 → 모멘텀 추세 (Momentum)

SHORT 패턴들 (상승 후 하락 기대):
- DN-DN-DN: 3연속 하락 → 하락 모멘텀 지속 (Momentum)
- U-DN-DN: 상승 후 2연속 하락 → 반전 확인 (Reversal Confirmation)
- U-U-BU: 2연속 상승 + Big Up → 과매수 반전 (Exhaustion)
- DN-U-DN: 하락-상승-하락 → Lower High 형성 (Bearish Structure)

주의: DN-DN-DN이 LONG/SHORT 모두 높은 WR을 보임
→ 이는 변동성이 높은 구간에서 양방향 신호 모두 수익 가능함을 의미
→ 실제로는 방향 예측이 아닌 변동성 활용 전략일 가능성
""")

# ============================================================
# FINAL RECOMMENDATIONS
# ============================================================

print('\n' + '=' * 70)
print('FINAL RECOMMENDATIONS')
print('=' * 70)

# Filter criteria:
# 1. WF >= 4/5
# 2. Counter-trend WR >= 50% OR insufficient counter-trend data (benefit of doubt)
# 3. Overall WR >= 80%
# 4. Trades >= 25

print('\n승인 기준:')
print('  - Walk-Forward >= 4/5')
print('  - 전체 WR >= 80%')
print('  - 거래 수 >= 25')
print('  - Counter-trend WR >= 50% (데이터 있는 경우)')

approved_long = []
approved_short = []
rejected = []

for r in validation_results:
    reasons = []

    # Check criteria
    if r['wf'] < 4:
        reasons.append(f"WF {r['wf']}/5 < 4")
    if r['wr'] < 80:
        reasons.append(f"WR {r['wr']:.1f}% < 80%")
    if r['trades'] < 25:
        reasons.append(f"Trades {r['trades']} < 25")

    # Counter-trend check (only if we have data)
    if r['counter_trend_wr'] is not None and r['counter_trend_wr'] < 50:
        reasons.append(f"Counter-trend WR {r['counter_trend_wr']:.1f}% < 50%")

    if not reasons:
        if r['direction'] == 'LONG':
            approved_long.append(r)
        else:
            approved_short.append(r)
    else:
        r['rejection_reasons'] = reasons
        rejected.append(r)

print(f'\n✅ 승인된 LONG 패턴: {len(approved_long)}개')
for r in approved_long:
    ct_str = f", CT {r['counter_trend_wr']:.0f}%" if r['counter_trend_wr'] else ""
    print(f"  '{r['pattern']}': ({r['tp']}, {r['sl']}),  # {r['trades']} trades, WR {r['wr']:.1f}%, WF {r['wf']}/5{ct_str}")

print(f'\n✅ 승인된 SHORT 패턴: {len(approved_short)}개')
for r in approved_short:
    ct_str = f", CT {r['counter_trend_wr']:.0f}%" if r['counter_trend_wr'] else ""
    print(f"  '{r['pattern']}': ({r['tp']}, {r['sl']}),  # {r['trades']} trades, WR {r['wr']:.1f}%, WF {r['wf']}/5{ct_str}")

print(f'\n❌ 거부된 패턴: {len(rejected)}개')
for r in rejected:
    print(f"  {r['pattern']} {r['direction']}: {', '.join(r['rejection_reasons'])}")

# ============================================================
# CONSTANTS.PY UPDATE RECOMMENDATION
# ============================================================

print('\n' + '=' * 70)
print('RECOMMENDED CONSTANTS.PY UPDATE')
print('=' * 70)

print('\n# v1.16 New Patterns from Discovery Research')
print('# Research: pattern_validation_comprehensive.py (2026-01-26)')
print()

if approved_long:
    print('# NEW VALIDATED_LONG_PATTERNS additions:')
    for r in approved_long:
        print(f"    '{r['pattern']}',  # WR {r['wr']:.1f}%, {r['trades']} trades, WF {r['wf']}/5")

if approved_short:
    print('\n# NEW VALIDATED_SHORT_PATTERNS additions:')
    for r in approved_short:
        print(f"    '{r['pattern']}',  # WR {r['wr']:.1f}%, {r['trades']} trades, WF {r['wf']}/5")

if approved_long or approved_short:
    print('\n# NEW PATTERN_OPTIMAL_TPSL additions:')
    for r in approved_long + approved_short:
        print(f"    '{r['pattern']}': ({r['tp']}, {r['sl']}),  # {r['direction']}, WR {r['wr']:.1f}%")

# Save results
results_df = pd.DataFrame(validation_results)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f'results/pattern_validation_comprehensive_{timestamp}.csv'
results_df.to_csv(output_file, index=False)
print(f'\n✅ Results saved to: {output_file}')

# Summary
print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)
print(f'총 검토 패턴: {len(validation_results)}개')
print(f'승인된 패턴: {len(approved_long) + len(approved_short)}개 (LONG {len(approved_long)}, SHORT {len(approved_short)})')
print(f'거부된 패턴: {len(rejected)}개')
