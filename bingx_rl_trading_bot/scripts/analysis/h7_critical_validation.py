#!/usr/bin/env python3
"""
H7 Critical Validation — v1.43.0 TP×0.75 + TO432 비판적 검증
================================================================
6가지 비판적 우려사항을 체계적으로 검증.

Test 1: WF 판별력 검증 — 랜덤 가설도 PASS하는가?
Test 2: R:R 악화 영향 — WR 의존도 스트레스 테스트
Test 3: Cascade×TP75 상호작용 — Cascade OFF 시 H7 효과 유지?
Test 4: 패턴별 TP 축소 적정성 — 균일 0.75 vs 패턴별 최적
Test 5: Live WR 시나리오 — WR 55~60% 환경에서 H7 생존 가능?
Test 6: 과적합 감지 — Shuffled signal timing으로 edge 보존 확인

Standard Research Protocol: compound, LEVERAGE=3, fee 0.10%×3.
"""
import sys, os, json, time
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from scripts.analysis.stack_resolution_study import (
    load_and_classify, compute_atr_ratio, compute_ema_slope,
    find_neutral_window, calc_stats, clamp,
    DATA_FILE, PATTERNS_FILE,
    N_SLOTS, DIRECTION_CAP, LEVERAGE, FEE_PCT,
    TIMEOUT_BARS, ATR_CLAMP_LO, ATR_CLAMP_HI, SLIPPAGE_BUFFER,
    AGG_RISK_COUNTER, AGG_RISK_WITH, BARS_PER_DAY,
    MDD_FULL_BELOW, MDD_MIN_ABOVE, MDD_MIN_SCALE,
    MOMENTUM_LOOKBACK, MOMENTUM_THRESHOLD, MOMENTUM_COOLDOWN,
    EARLY_CONFIRM, EARLY_MIN_PROFIT,
)

from scripts.analysis.entry_improvement_hypotheses import portfolio_sim_extended


print('=' * 95)
print('H7 CRITICAL VALIDATION — v1.43.0 비판적 검증')
print(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
print('=' * 95)

# ============================================================
# LOAD DATA
# ============================================================
print('\nLoading data...')
df = load_and_classify(DATA_FILE)
atr_ratio = compute_atr_ratio(df)
ema_slope = compute_ema_slope(df['close'].values)
ns, ne = find_neutral_window(df['close'].values, 1.0)
print(f'Neutral: {ns}-{ne} ({ne-ns} bars, {(ne-ns)/288:.0f}d)')

with open(PATTERNS_FILE) as f:
    pdata = json.load(f)
pats_raw = pdata['patterns']
tpsl = pdata.get('patterns_tpsl', {})

pat_lookup = {}
for pat_name in pats_raw.get('long', []):
    tp_sl = tpsl.get(pat_name, [2.0, 3.0])
    pat_lookup[pat_name] = {'direction': 'LONG', 'tp': tp_sl[0], 'sl': tp_sl[1]}
for pat_name in pats_raw.get('short', []):
    tp_sl = tpsl.get(pat_name, [2.0, 3.0])
    pat_lookup[pat_name] = {'direction': 'SHORT', 'tp': tp_sl[0], 'sl': tp_sl[1]}
print(f'{len(pat_lookup)} patterns')

rctypes = df['rctype'].values
n_bars = len(df)
signal_tuples = []
for i in range(2, n_bars):
    tri = f"{rctypes[i-2]}-{rctypes[i-1]}-{rctypes[i]}"
    if tri in pat_lookup:
        p = pat_lookup[tri]
        signal_tuples.append((i, tri, p['direction'], p['tp'], p['sl']))
print(f'{len(signal_tuples)} signals')

opens = df['open'].values
highs = df['high'].values
lows = df['low'].values
closes = df['close'].values
tcodes = df['rctype'].values

common = dict(
    opens=opens, highs=highs, lows=lows, closes=closes,
    type_codes=tcodes, n_bars=n_bars,
    atr_ratio=atr_ratio, ema_slope=ema_slope,
)


def run_sim(start, end, **kwargs):
    trades = portfolio_sim_extended(
        signal_tuples=signal_tuples, start_bar=start, end_bar=end,
        **kwargs, **common)
    return trades, calc_stats(trades)


def run_wf(n_folds=3, **kwargs):
    total = ne - ns
    seg_size = total // (n_folds + 1)
    folds = []
    for fold in range(n_folds):
        oos_start = ns + seg_size * (fold + 1)
        oos_end = ns + seg_size * (fold + 2) if fold < n_folds - 1 else ne
        _, stats_f = run_sim(oos_start, oos_end, **kwargs)
        folds.append(stats_f)
    n_pass = sum(1 for f in folds if f['pnl'] > 0)
    return {
        'folds': folds,
        'n_pass': n_pass,
        'avg_oos': np.mean([f['pnl'] for f in folds]),
        'min_oos': min(f['pnl'] for f in folds),
        'verdict': f'{n_pass}/3 PASS' if n_pass == 3 else f'{n_pass}/3 FAIL',
    }


results = {}

# ============================================================
# TEST 1: WF 판별력 검증 — 랜덤 TP multiplier도 PASS하는가?
# ============================================================
print('\n' + '=' * 95)
print('TEST 1: WF 판별력 — 랜덤 TP multiplier들의 WF PASS 비율')
print('=' * 95)
print('(모든 tp_mult가 PASS하면 WF 판별력 부족 = 과적합 경고)')

rng = np.random.RandomState(42)
tp_mults_test = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0,
                 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0]
# Also test with timeout variations
test1_results = []
for tm in tp_mults_test:
    for to_bars in [432, 864]:
        _, stats = run_sim(ns, ne, tp_mult=tm, timeout_bars=to_bars)
        wf = run_wf(tp_mult=tm, timeout_bars=to_bars)
        test1_results.append({
            'tp_mult': tm, 'timeout': to_bars,
            'is_pnl': stats['pnl'], 'is_mdd': stats['mdd'],
            'is_pm': stats['pnl_mdd'],
            'oos_avg': wf['avg_oos'], 'oos_min': wf['min_oos'],
            'wf': wf['verdict'],
        })

print(f'\n{"tp_mult":>8} {"TO":>5} {"IS PnL":>8} {"IS P/M":>8} {"OOS Avg":>8} {"OOS Min":>8} {"WF":>10}')
print('-' * 65)
n_pass_total = 0
n_fail_total = 0
for r in test1_results:
    mark = ' <<<' if r['tp_mult'] == 0.75 and r['timeout'] == 432 else ''
    print(f'{r["tp_mult"]:>8.2f} {r["timeout"]:>5} {r["is_pnl"]:>+8.1f} '
          f'{r["is_pm"]:>8.1f} {r["oos_avg"]:>+8.1f} {r["oos_min"]:>+8.1f} {r["wf"]:>10}{mark}')
    if '3/3' in r['wf']:
        n_pass_total += 1
    else:
        n_fail_total += 1

pass_rate = n_pass_total / len(test1_results) * 100
print(f'\nWF PASS rate: {n_pass_total}/{len(test1_results)} ({pass_rate:.0f}%)')
if pass_rate > 80:
    print('⚠️ WARNING: >80% PASS → WF 판별력 낮음, 과적합 위험')
elif pass_rate > 50:
    print('⚡ CAUTION: 50-80% PASS → 중간 판별력')
else:
    print('✅ WF 판별력 양호 (<50% PASS)')

results['test1_wf_discrimination'] = {
    'pass_rate_pct': round(pass_rate, 1),
    'n_pass': n_pass_total, 'n_total': len(test1_results),
    'details': test1_results,
}

# ============================================================
# TEST 2: R:R 악화 스트레스 테스트
# ============================================================
print('\n' + '=' * 95)
print('TEST 2: R:R 악화 스트레스 — WR degradation 시나리오')
print('=' * 95)
print('TP×0.75 = R:R 25% 악화. WR 하락 시 얼마나 빨리 손실 전환?')

# IS에서 H0 vs H7의 WR별 시뮬레이션 결과 비교
# H0과 H7의 trades를 뽑아서 WR별 breakeven 분석
_, h0_stats = run_sim(ns, ne)  # baseline
h0_trades, _ = run_sim(ns, ne)
_, h7_stats = run_sim(ns, ne, tp_mult=0.75, timeout_bars=432)
h7_trades, _ = run_sim(ns, ne, tp_mult=0.75, timeout_bars=432)

# Calculate avg win/loss for each
h0_wins = [t['pnl_slot'] for t in h0_trades if t['pnl_slot'] > 0]
h0_losses = [t['pnl_slot'] for t in h0_trades if t['pnl_slot'] <= 0]
h7_wins = [t['pnl_slot'] for t in h7_trades if t['pnl_slot'] > 0]
h7_losses = [t['pnl_slot'] for t in h7_trades if t['pnl_slot'] <= 0]

h0_avg_win = np.mean(h0_wins) if h0_wins else 0
h0_avg_loss = abs(np.mean(h0_losses)) if h0_losses else 1
h7_avg_win = np.mean(h7_wins) if h7_wins else 0
h7_avg_loss = abs(np.mean(h7_losses)) if h7_losses else 1

h0_rr = h0_avg_win / h0_avg_loss if h0_avg_loss > 0 else 0
h7_rr = h7_avg_win / h7_avg_loss if h7_avg_loss > 0 else 0
h0_be_wr = h0_avg_loss / (h0_avg_win + h0_avg_loss) * 100 if (h0_avg_win + h0_avg_loss) > 0 else 50
h7_be_wr = h7_avg_loss / (h7_avg_win + h7_avg_loss) * 100 if (h7_avg_win + h7_avg_loss) > 0 else 50

print(f'\n{"":>20} {"H0 Baseline":>15} {"H7 TP75+TO432":>15} {"Delta":>10}')
print('-' * 65)
print(f'{"Avg Win %":>20} {h0_avg_win:>+15.2f} {h7_avg_win:>+15.2f} {h7_avg_win-h0_avg_win:>+10.2f}')
print(f'{"Avg Loss %":>20} {-h0_avg_loss:>15.2f} {-h7_avg_loss:>15.2f} {h0_avg_loss-h7_avg_loss:>+10.2f}')
print(f'{"R:R Ratio":>20} {h0_rr:>15.3f} {h7_rr:>15.3f} {h7_rr-h0_rr:>+10.3f}')
print(f'{"Breakeven WR %":>20} {h0_be_wr:>15.1f} {h7_be_wr:>15.1f} {h7_be_wr-h0_be_wr:>+10.1f}')
print(f'{"Actual WR %":>20} {h0_stats["wr"]:>15.1f} {h7_stats["wr"]:>15.1f} {h7_stats["wr"]-h0_stats["wr"]:>+10.1f}')
print(f'{"WR Margin (pp)":>20} {h0_stats["wr"]-h0_be_wr:>15.1f} {h7_stats["wr"]-h7_be_wr:>15.1f} '
      f'{(h7_stats["wr"]-h7_be_wr)-(h0_stats["wr"]-h0_be_wr):>+10.1f}')

wr_margin_h0 = h0_stats['wr'] - h0_be_wr
wr_margin_h7 = h7_stats['wr'] - h7_be_wr

if wr_margin_h7 < wr_margin_h0:
    print(f'\n⚠️ H7 WR margin ({wr_margin_h7:.1f}pp) < H0 ({wr_margin_h0:.1f}pp) = 안전마진 축소')
else:
    print(f'\n✅ H7 WR margin ({wr_margin_h7:.1f}pp) ≥ H0 ({wr_margin_h0:.1f}pp)')

# Simulate WR degradation: what if live WR drops to 55-65%?
print(f'\n  WR 하락 시나리오 (per-trade expected PnL):')
for sim_wr in [50, 55, 58, 60, 62, 65, 68, 70, 72, 75]:
    h0_exp = (sim_wr/100 * h0_avg_win) - ((100-sim_wr)/100 * h0_avg_loss)
    h7_exp = (sim_wr/100 * h7_avg_win) - ((100-sim_wr)/100 * h7_avg_loss)
    better = 'H7' if h7_exp > h0_exp else 'H0'
    mark = '  ← live range' if 55 <= sim_wr <= 60 else ''
    print(f'    WR {sim_wr}%: H0 E[trade]={h0_exp:+.3f}%, H7 E[trade]={h7_exp:+.3f}% → {better}{mark}')

results['test2_rr_stress'] = {
    'h0_rr': round(h0_rr, 4), 'h7_rr': round(h7_rr, 4),
    'h0_be_wr': round(h0_be_wr, 2), 'h7_be_wr': round(h7_be_wr, 2),
    'h0_wr_margin': round(wr_margin_h0, 2), 'h7_wr_margin': round(wr_margin_h7, 2),
    'h0_avg_win': round(h0_avg_win, 4), 'h0_avg_loss': round(h0_avg_loss, 4),
    'h7_avg_win': round(h7_avg_win, 4), 'h7_avg_loss': round(h7_avg_loss, 4),
}

# ============================================================
# TEST 3: Cascade × TP75 상호작용
# ============================================================
print('\n' + '=' * 95)
print('TEST 3: Cascade × TP75 상호작용 — Cascade OFF 시 H7 효과 유지?')
print('=' * 95)

from scripts.analysis.stack_resolution_study import portfolio_sim

# portfolio_sim has cascade_enabled parameter
# We need to test: H0+cascade, H7+cascade, H0-cascade, H7-cascade
cascade_configs = {
    'H0_cascade_ON':  {'tp_mult': 1.0,  'timeout_bars': 864},
    'H7_cascade_ON':  {'tp_mult': 0.75, 'timeout_bars': 432},
}
# For cascade OFF, we use portfolio_sim directly
print(f'\n{"Config":<25} {"PnL%":>8} {"MDD%":>8} {"P/M":>8} {"WR%":>8} {"Trades":>8}')
print('-' * 70)

# H0+cascade ON (via extended sim)
_, s = run_sim(ns, ne)
print(f'{"H0_cascade_ON":<25} {s["pnl"]:>+8.1f} {s["mdd"]:>8.2f} {s["pnl_mdd"]:>8.1f} {s["wr"]:>8.1f} {s["trades"]:>8}')
h0_on = s['pnl_mdd']

# H7+cascade ON (via extended sim)
_, s = run_sim(ns, ne, tp_mult=0.75, timeout_bars=432)
print(f'{"H7_cascade_ON":<25} {s["pnl"]:>+8.1f} {s["mdd"]:>8.2f} {s["pnl_mdd"]:>8.1f} {s["wr"]:>8.1f} {s["trades"]:>8}')
h7_on = s['pnl_mdd']

# For cascade OFF, we need to modify portfolio_sim or use the cascade_enabled param
# The stack_resolution_study.portfolio_sim has cascade_enabled param
# But it doesn't have tp_mult or timeout override...
# So we need to modify signals for tp_mult and use portfolio_sim cascade_enabled=False

# Build signal tuples with modified TP for H7
signal_tuples_h7 = [(bar, pat, d, tp*0.75, sl) for bar, pat, d, tp, sl in signal_tuples]

# H0_cascade_OFF
trades_off, _ = portfolio_sim(
    signal_tuples=signal_tuples, start_bar=ns, end_bar=ne,
    cascade_enabled=False, **common)
s = calc_stats(trades_off)
print(f'{"H0_cascade_OFF":<25} {s["pnl"]:>+8.1f} {s["mdd"]:>8.2f} {s["pnl_mdd"]:>8.1f} {s["wr"]:>8.1f} {s["trades"]:>8}')
h0_off = s['pnl_mdd']

# H7_cascade_OFF — need to use tp_mult in signal tuples AND timeout override
# portfolio_sim doesn't have timeout override, but stack_resolution_study's has default 864
# We can't easily override timeout in portfolio_sim without modifying it
# So let's use portfolio_sim_extended with cascade disabled manually
# Actually, entry_improvement_hypotheses.py's portfolio_sim_extended always has cascade ON
# Let's just use the H7 TP signals + portfolio_sim cascade_enabled=False
trades_off_h7, _ = portfolio_sim(
    signal_tuples=signal_tuples_h7, start_bar=ns, end_bar=ne,
    cascade_enabled=False, **common)
s = calc_stats(trades_off_h7)
print(f'{"H7tp_cascade_OFF":<25} {s["pnl"]:>+8.1f} {s["mdd"]:>8.2f} {s["pnl_mdd"]:>8.1f} {s["wr"]:>8.1f} {s["trades"]:>8}')
h7_off = s['pnl_mdd']

# Interaction effect
h7_lift_on = h7_on - h0_on
h7_lift_off = h7_off - h0_off
cascade_lift_h0 = h0_on - h0_off
cascade_lift_h7 = h7_on - h7_off

print(f'\n  H7 lift (cascade ON):  {h7_lift_on:+.1f} PnL/MDD')
print(f'  H7 lift (cascade OFF): {h7_lift_off:+.1f} PnL/MDD')
print(f'  Cascade lift (H0):     {cascade_lift_h0:+.1f} PnL/MDD')
print(f'  Cascade lift (H7):     {cascade_lift_h7:+.1f} PnL/MDD')

interaction = (h7_on - h0_on) - (h7_off - h0_off)
print(f'  Interaction effect:    {interaction:+.1f}')
if interaction > 10:
    print('  ⚠️ H7 효과가 Cascade에 크게 의존 — Cascade 없이는 H7 효과 감소')
elif interaction < -10:
    print('  ⚠️ H7이 Cascade를 약화시킴 — 부정적 상호작용')
else:
    print('  ✅ H7 효과는 Cascade 독립적 (상호작용 ±10 이내)')

results['test3_cascade_interaction'] = {
    'h0_cascade_on': round(h0_on, 2), 'h7_cascade_on': round(h7_on, 2),
    'h0_cascade_off': round(h0_off, 2), 'h7_cascade_off': round(h7_off, 2),
    'h7_lift_on': round(h7_lift_on, 2), 'h7_lift_off': round(h7_lift_off, 2),
    'interaction': round(interaction, 2),
}

# ============================================================
# TEST 4: 패턴별 TP 축소 적정성
# ============================================================
print('\n' + '=' * 95)
print('TEST 4: 패턴별 TP 축소 적정성 — 균일 0.75 vs 개별 분석')
print('=' * 95)

# Run H7 and collect per-pattern stats
h7_trades_all, _ = run_sim(ns, ne, tp_mult=0.75, timeout_bars=432)
h0_trades_all, _ = run_sim(ns, ne)

# Per-pattern comparison
pat_stats_h0 = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0})
pat_stats_h7 = defaultdict(lambda: {'wins': 0, 'losses': 0, 'pnl': 0})

for t in h0_trades_all:
    p = t['pattern']
    pat_stats_h0[p]['pnl'] += t['pnl_slot']
    if t['pnl_slot'] > 0:
        pat_stats_h0[p]['wins'] += 1
    else:
        pat_stats_h0[p]['losses'] += 1

for t in h7_trades_all:
    p = t['pattern']
    pat_stats_h7[p]['pnl'] += t['pnl_slot']
    if t['pnl_slot'] > 0:
        pat_stats_h7[p]['wins'] += 1
    else:
        pat_stats_h7[p]['losses'] += 1

# Compare
hurt_patterns = []
helped_patterns = []
for pat in sorted(set(list(pat_stats_h0.keys()) + list(pat_stats_h7.keys()))):
    h0_n = pat_stats_h0[pat]['wins'] + pat_stats_h0[pat]['losses']
    h7_n = pat_stats_h7[pat]['wins'] + pat_stats_h7[pat]['losses']
    if h0_n < 3 and h7_n < 3:
        continue
    h0_pnl = pat_stats_h0[pat]['pnl']
    h7_pnl = pat_stats_h7[pat]['pnl']
    delta = h7_pnl - h0_pnl
    if delta < -5:
        hurt_patterns.append((pat, h0_n, h7_n, h0_pnl, h7_pnl, delta))
    elif delta > 5:
        helped_patterns.append((pat, h0_n, h7_n, h0_pnl, h7_pnl, delta))

print(f'\n  패턴 수: H0 {len(pat_stats_h0)}, H7 {len(pat_stats_h7)}')
print(f'  H7에서 악화된 패턴 (delta < -5%): {len(hurt_patterns)}')
print(f'  H7에서 개선된 패턴 (delta > +5%): {len(helped_patterns)}')

if hurt_patterns:
    print(f'\n  ⚠️ Hurt patterns (top 10):')
    for pat, n0, n7, p0, p7, d in sorted(hurt_patterns, key=lambda x: x[5])[:10]:
        print(f'    {pat:<20} H0: {n0:>3} trades, PnL={p0:>+7.1f}% | '
              f'H7: {n7:>3} trades, PnL={p7:>+7.1f}% | Δ={d:>+7.1f}%')

if helped_patterns:
    print(f'\n  ✅ Helped patterns (top 10):')
    for pat, n0, n7, p0, p7, d in sorted(helped_patterns, key=lambda x: -x[5])[:10]:
        print(f'    {pat:<20} H0: {n0:>3} trades, PnL={p0:>+7.1f}% | '
              f'H7: {n7:>3} trades, PnL={p7:>+7.1f}% | Δ={d:>+7.1f}%')

n_hurt = len(hurt_patterns)
n_helped = len(helped_patterns)
n_neutral = len(set(list(pat_stats_h0.keys()) + list(pat_stats_h7.keys()))) - n_hurt - n_helped
print(f'\n  Summary: {n_helped} helped, {n_hurt} hurt, {n_neutral} neutral')

results['test4_pattern_impact'] = {
    'n_helped': n_helped, 'n_hurt': n_hurt, 'n_neutral': n_neutral,
    'hurt_patterns': [(p, round(d, 2)) for p, _, _, _, _, d in sorted(hurt_patterns, key=lambda x: x[5])[:10]],
    'helped_patterns': [(p, round(d, 2)) for p, _, _, _, _, d in sorted(helped_patterns, key=lambda x: -x[5])[:10]],
}

# ============================================================
# TEST 5: Live WR 시나리오 — WR 55-60%에서 H7 생존?
# ============================================================
print('\n' + '=' * 95)
print('TEST 5: Live WR 스트레스 — 랜덤 일부 거래를 강제 패배로 전환')
print('=' * 95)
print('현재 live WR ~55.6%. 백테스트 WR을 강제로 낮춰서 H7 생존성 확인.')

n_mc = 50  # reduced from 200 for speed
target_wrs = [55, 58, 60, 65]

# Pre-compute trades once, then degrade WR via MC
h0_trades_all, _ = run_sim(ns, ne)
h7_trades_all, _ = run_sim(ns, ne, tp_mult=0.75, timeout_bars=432)

for target_wr in target_wrs:
    h0_survive = 0
    h7_survive = 0
    h0_pnls = []
    h7_pnls = []

    for seed in range(n_mc):
        rng_s = np.random.RandomState(seed)

        # Use pre-computed trades
        h0_t = h0_trades_all
        h7_t = h7_trades_all

        # Degrade WR: randomly flip wins to losses
        def degrade(trades_list, target_wr_pct, rng_):
            wins = [t for t in trades_list if t['pnl_slot'] > 0]
            losses = [t for t in trades_list if t['pnl_slot'] <= 0]
            actual_wr = len(wins) / len(trades_list) * 100 if trades_list else 0
            if actual_wr <= target_wr_pct:
                return sum(t['pnl_portfolio'] for t in trades_list)
            # How many wins to flip
            n_to_flip = int(len(wins) - target_wr_pct / 100 * len(trades_list))
            if n_to_flip <= 0:
                return sum(t['pnl_portfolio'] for t in trades_list)
            flip_idx = rng_.choice(len(wins), size=min(n_to_flip, len(wins)), replace=False)
            total_pnl = 0
            flipped = set(flip_idx)
            win_i = 0
            for t in trades_list:
                if t['pnl_slot'] > 0:
                    if win_i in flipped:
                        # Flip to loss: use average loss magnitude
                        avg_loss = np.mean([abs(l['pnl_portfolio']) for l in losses]) if losses else abs(t['pnl_portfolio'])
                        total_pnl -= avg_loss
                    else:
                        total_pnl += t['pnl_portfolio']
                    win_i += 1
                else:
                    total_pnl += t['pnl_portfolio']
            return total_pnl

        h0_pnl = degrade(h0_t, target_wr, rng_s)
        h7_pnl = degrade(h7_t, target_wr, np.random.RandomState(seed + 10000))

        h0_pnls.append(h0_pnl)
        h7_pnls.append(h7_pnl)
        if h0_pnl > 0:
            h0_survive += 1
        if h7_pnl > 0:
            h7_survive += 1

    h0_sr = h0_survive / n_mc * 100
    h7_sr = h7_survive / n_mc * 100
    better = 'H7' if h7_sr > h0_sr else ('H0' if h0_sr > h7_sr else 'TIE')
    print(f'  WR {target_wr}%: H0 survive {h0_sr:.0f}% (med PnL {np.median(h0_pnls):+.1f}%) | '
          f'H7 survive {h7_sr:.0f}% (med PnL {np.median(h7_pnls):+.1f}%) → {better}')

    results[f'test5_wr{target_wr}'] = {
        'h0_survive_pct': round(h0_sr, 1), 'h7_survive_pct': round(h7_sr, 1),
        'h0_med_pnl': round(float(np.median(h0_pnls)), 2),
        'h7_med_pnl': round(float(np.median(h7_pnls)), 2),
    }

# ============================================================
# TEST 6: 과적합 감지 — Shuffled signal timing
# ============================================================
print('\n' + '=' * 95)
print('TEST 6: 과적합 감지 — 신호 타이밍 셔플 (패턴 매칭 유지, 시점 랜덤)')
print('=' * 95)
print('진짜 edge는 타이밍에 있어야 함. 셔플 후에도 동일 성과면 과적합.')

n_shuffle = 30  # reduced from 100 for speed
shuffle_pnls_h0 = []
shuffle_pnls_h7 = []
original_h0_pnl = h0_stats['pnl']
original_h7_pnl = h7_stats['pnl']

for seed in range(n_shuffle):
    rng_s = np.random.RandomState(seed)
    # Shuffle signal bars (keep pattern/direction/tp/sl)
    bars_only = [s[0] for s in signal_tuples]
    rng_s.shuffle(bars_only)
    shuffled_signals = [(bars_only[i], s[1], s[2], s[3], s[4])
                        for i, s in enumerate(signal_tuples)]

    # Run H0
    trades_s = portfolio_sim_extended(
        signal_tuples=shuffled_signals, start_bar=ns, end_bar=ne,
        opens=opens, highs=highs, lows=lows, closes=closes,
        type_codes=tcodes, n_bars=n_bars,
        atr_ratio=atr_ratio, ema_slope=ema_slope)
    s_stats = calc_stats(trades_s)
    shuffle_pnls_h0.append(s_stats['pnl'])

    # Run H7
    trades_s7 = portfolio_sim_extended(
        signal_tuples=shuffled_signals, start_bar=ns, end_bar=ne,
        tp_mult=0.75, timeout_bars=432,
        opens=opens, highs=highs, lows=lows, closes=closes,
        type_codes=tcodes, n_bars=n_bars,
        atr_ratio=atr_ratio, ema_slope=ema_slope)
    s7_stats = calc_stats(trades_s7)
    shuffle_pnls_h7.append(s7_stats['pnl'])

# p-value: fraction of shuffled >= original
p_h0 = sum(1 for x in shuffle_pnls_h0 if x >= original_h0_pnl) / n_shuffle
p_h7 = sum(1 for x in shuffle_pnls_h7 if x >= original_h7_pnl) / n_shuffle

print(f'\n  H0 Original PnL: {original_h0_pnl:+.1f}%')
print(f'  H0 Shuffled: mean={np.mean(shuffle_pnls_h0):+.1f}%, '
      f'median={np.median(shuffle_pnls_h0):+.1f}%, p-value={p_h0:.3f}')
print(f'\n  H7 Original PnL: {original_h7_pnl:+.1f}%')
print(f'  H7 Shuffled: mean={np.mean(shuffle_pnls_h7):+.1f}%, '
      f'median={np.median(shuffle_pnls_h7):+.1f}%, p-value={p_h7:.3f}')

if p_h0 > 0.05:
    print(f'\n  ⚠️ H0 p={p_h0:.3f} > 0.05: 패턴 타이밍 무의미 — 수익의 원천이 패턴이 아님')
if p_h7 > 0.05:
    print(f'  ⚠️ H7 p={p_h7:.3f} > 0.05: TP75+TO432에서도 타이밍 edge 없음')
if p_h0 < 0.05 and p_h7 < 0.05:
    print(f'\n  ✅ 양쪽 p < 0.05: 패턴 타이밍에 유의한 edge 존재')

results['test6_overfit'] = {
    'h0_original_pnl': round(original_h0_pnl, 2),
    'h0_shuffle_mean': round(float(np.mean(shuffle_pnls_h0)), 2),
    'h0_p_value': round(p_h0, 4),
    'h7_original_pnl': round(original_h7_pnl, 2),
    'h7_shuffle_mean': round(float(np.mean(shuffle_pnls_h7)), 2),
    'h7_p_value': round(p_h7, 4),
}

# ============================================================
# FINAL VERDICT
# ============================================================
print('\n' + '=' * 95)
print('FINAL VERDICT')
print('=' * 95)

concerns = []
passes = []

# Test 1
if pass_rate > 80:
    concerns.append(f'T1: WF 판별력 낮음 ({pass_rate:.0f}% PASS)')
else:
    passes.append(f'T1: WF 판별력 양호 ({pass_rate:.0f}% PASS)')

# Test 2
if wr_margin_h7 < 5:
    concerns.append(f'T2: WR margin 위험 수준 ({wr_margin_h7:.1f}pp)')
elif wr_margin_h7 < wr_margin_h0:
    concerns.append(f'T2: WR margin 축소 ({wr_margin_h7:.1f}pp < {wr_margin_h0:.1f}pp)')
else:
    passes.append(f'T2: WR margin 유지 ({wr_margin_h7:.1f}pp)')

# Test 3
if abs(interaction) > 30:
    concerns.append(f'T3: 강한 Cascade 의존 (interaction {interaction:+.1f})')
else:
    passes.append(f'T3: Cascade 독립적 (interaction {interaction:+.1f})')

# Test 4
if n_hurt > n_helped * 2:
    concerns.append(f'T4: 다수 패턴 악화 ({n_hurt} hurt > {n_helped} helped)')
else:
    passes.append(f'T4: 패턴 영향 균형 ({n_helped} helped, {n_hurt} hurt)')

# Test 5 (WR 58% — realistic live scenario)
t5_58 = results.get('test5_wr58', {})
if t5_58.get('h7_survive_pct', 0) < 50:
    concerns.append(f'T5: WR 58%에서 H7 생존률 낮음 ({t5_58.get("h7_survive_pct", 0):.0f}%)')
else:
    passes.append(f'T5: WR 58%에서 H7 생존 가능 ({t5_58.get("h7_survive_pct", 0):.0f}%)')

# Test 6
if p_h7 > 0.05:
    concerns.append(f'T6: 패턴 타이밍 edge 없음 (p={p_h7:.3f})')
else:
    passes.append(f'T6: 패턴 타이밍 유의 (p={p_h7:.3f})')

print(f'\n✅ PASS ({len(passes)}):')
for p in passes:
    print(f'  {p}')
print(f'\n⚠️ CONCERNS ({len(concerns)}):')
for c in concerns:
    print(f'  {c}')

if len(concerns) >= 3:
    verdict = 'ROLLBACK RECOMMENDED'
elif len(concerns) >= 2:
    verdict = 'CAUTION — monitor closely'
elif len(concerns) >= 1:
    verdict = 'ACCEPTABLE with caveats'
else:
    verdict = 'STRONG — proceed'

print(f'\n>>> VERDICT: {verdict}')
print(f'    ({len(passes)} pass, {len(concerns)} concerns out of 6 tests)')

results['verdict'] = {
    'decision': verdict,
    'n_pass': len(passes), 'n_concern': len(concerns),
    'passes': passes, 'concerns': concerns,
}

# Save
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'results', 'h7_critical_validation.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f'\nSaved: {out_path}')
print(f'Runtime: {time.time():.0f}s')
