"""
Post-BUG-fix + Pre-F-v1-Canary Legacy Gap Analysis — 2026-04-18 12:00 ~ 2026-04-22 11:00 UTC
=============================================================================================

Purpose
-------
BUG#62~65 (2026-04-18) closed many backtest-vs-live parity gaps. F v1 Canary began
2026-04-22 ~14:00 UTC. The ~4 day window between these events used legacy
TRAILING_STOP_MARKET (pre-activation) + baton STOP_MARKET (post-activation) + fractal
SL. The gap in this period was never quantified per-trade — this script does that.

Inputs
------
- LIVE trades: results/c1_breakout_state.json, filtered to exit window
- BT candles: BingX swap 15m via CCXT
- Config: time-sliced — progressive_trail.enabled flip at 2026-04-21 08:41 UTC (commit d193f28)

Method
------
1. Load 16 LIVE trades from state.json
2. Run BT starting from ~04-18 00:00 UTC, tracking each BT signal that enters
3. Match LIVE<->BT by entry_time (±15m tolerance)
4. Classify each trade: A (≈match), B (LIVE TRAILING intrabar worse), C (LIVE prematurely
   exited, BT would continue), D (reason mismatch, economically different)
5. Decompose per-trade gap: entry_slip, exit_slip (when matched-reason), timing, reason_mismatch
6. F v2 counterfactual: BT outcome replaces LIVE outcome for trades where BT > LIVE
"""
from __future__ import annotations

import sys, json, math
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings
)
from scripts.production.c1_breakout.signals import C1BreakoutSignal

# ---------------------------------------------------------------------------
# Configuration — time-sliced for progressive_trail flag

PROG_ENABLE_UTC = datetime(2026, 4, 21, 8, 41, 0, tzinfo=timezone.utc)

BASE_CONFIG = dict(
    channel_period=15, body_min_ratio=0.4, atr_period=14,
    trail_K=2.5, max_sl_atr=3.3, emergency_sl_pct=3.0,
    max_hold_bars=192, sl_min_pct=0.15, sl_max_pct=3.0,
    min_bars_between=2, trail_activation_pct=0.05, fractal_lookback=10,
)
CONFIG_PRE_PROG = {**BASE_CONFIG,
    'progressive_trail': {'enabled': False, 'threshold_pct': 0.9, 'trail_K_post': 0.5}}
CONFIG_POST_PROG = {**BASE_CONFIG,
    'progressive_trail': {'enabled': True, 'threshold_pct': 0.9, 'trail_K_post': 0.5}}

FEE_RT_PCT = 0.10
LEVERAGE = 3

WINDOW_START = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
WINDOW_END   = datetime(2026, 4, 22, 11, 0, tzinfo=timezone.utc)
# Fetch wider to warm up indicators AND catch trade #1 (SHORT entered ~04-17 ~00:00 UTC)
FETCH_START  = datetime(2026, 4, 5, 0, 0, tzinfo=timezone.utc)
FETCH_END    = datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)

# True exit-slip measurement from slippage_diagnosis_20260422.md:
#   TRAILING_STOP_MARKET adverse fill: 0.007% (1x), 18 orders
#   STOP_MARKET (baton) adverse fill: 0.008% (1x), 2 orders
# Per matched trade, avg 0.0075% × 3x = ~0.0225pp of true execution slip.
TRUE_EXIT_SLIP_1X_PER_TRADE = 0.0075


# ---------------------------------------------------------------------------
def load_live_trades() -> list[dict]:
    """Load LIVE trades from state.json that exited inside the window.
    pnl_pct is already 3x fee-deducted (verified by spot-check)."""
    with open(ROOT / 'results' / 'c1_breakout_state.json') as f:
        st = json.load(f)
    out = []
    for t in st['trade_history']:
        if '2026-04-18T12:00' <= t['exit_time'] < '2026-04-22T11:00':
            # Derive entry_time from exit_time - bars_held × 15m
            exit_dt = datetime.fromisoformat(t['exit_time']).replace(tzinfo=timezone.utc)
            entry_dt = exit_dt - timedelta(minutes=15 * t['bars_held'])
            out.append({**t, 'entry_time_utc': entry_dt, 'exit_time_utc': exit_dt})
    return out


def fetch_candles():
    import ccxt
    ex = ccxt.bingx({'options': {'defaultType': 'swap'}})
    st = int(FETCH_START.timestamp() * 1000)
    en = int(FETCH_END.timestamp() * 1000)
    all_c, since = [], st
    while since < en:
        cs = ex.fetch_ohlcv('BTC-USDT', '15m', since=since, limit=1000)
        if not cs: break
        all_c.extend(cs)
        if cs[-1][0] <= since: break
        since = cs[-1][0] + 1
    seen, uniq = set(), []
    for c in all_c:
        if c[0] not in seen:
            seen.add(c[0]); uniq.append(c)
    uniq = [c for c in uniq if st <= c[0] < en]
    uniq.sort(key=lambda x: x[0])
    return uniq


# ---------------------------------------------------------------------------
def run_bt(candles: list) -> list[dict]:
    """Run BT with time-sliced progressive_trail config.
    Returns list of trades with entry/exit prices and times (UTC).
    """
    ts = [datetime.fromtimestamp(c[0]/1000, tz=timezone.utc) for c in candles]
    opens  = [c[1] for c in candles]
    highs  = [c[2] for c in candles]
    lows   = [c[3] for c in candles]
    closes = [c[4] for c in candles]
    n = len(closes)

    atr = compute_atr(highs, lows, closes, BASE_CONFIG['atr_period'])
    ch_h, ch_l = compute_channel(highs, lows, BASE_CONFIG['channel_period'])
    sw_l, sw_h = compute_fractal_swings(highs, lows, BASE_CONFIG['fractal_lookback'])

    sig_pre  = C1BreakoutSignal(CONFIG_PRE_PROG)
    sig_post = C1BreakoutSignal(CONFIG_POST_PROG)

    # Extend backwards far enough that an entry on 04-17 can be BT-simulated (trade #1)
    s_idx = next(i for i, t in enumerate(ts) if t >= WINDOW_START - timedelta(days=5))
    e_idx = next(i for i in range(n-1, -1, -1) if ts[i] < WINDOW_END + timedelta(hours=48))

    trades = []
    in_pos = False
    cd = 0
    pdir = pprice = psl = pbest = None
    pheld = 0
    pt_entry_ts = None

    for i in range(s_idx, e_idx + 1):
        if in_pos:
            pheld += 1
            pbest = max(pbest, highs[i]) if pdir == 'LONG' else min(pbest, lows[i])
            # Use config matching entry_time
            signal = sig_post if pt_entry_ts >= PROG_ENABLE_UTC else sig_pre
            er = signal.check_exit(
                direction=pdir, entry_price=pprice, best_price=pbest,
                current_high=highs[i], current_low=lows[i], current_close=closes[i],
                sl_price=psl, atr_val=atr[i] if not math.isnan(atr[i]) else atr[i-1],
                bars_held=pheld)
            if er:
                xp, rs = er['exit_price'], er['reason']
                pnl = (xp/pprice - 1)*100 if pdir == 'LONG' else (1 - xp/pprice)*100
                pnl_net = pnl - FEE_RT_PCT  # 1x net after fee
                trades.append({
                    'dir': pdir,
                    'entry_price': round(pprice, 2),
                    'exit_price': round(xp, 2),
                    'pnl_1x_net': round(pnl_net, 4),
                    'pnl_3x': round(pnl_net * LEVERAGE, 4),
                    'reason': rs,
                    'bars': pheld,
                    'entry_time': pt_entry_ts.isoformat(),
                    'exit_time': ts[i].isoformat(),
                    'prog_active_at_entry': pt_entry_ts >= PROG_ENABLE_UTC,
                })
                in_pos = False
                cd = i + BASE_CONFIG['min_bars_between']
                pdir = None

        if not in_pos and i >= cd and i < e_idx:
            if math.isnan(atr[i]) or math.isnan(ch_h[i]):
                continue
            signal = sig_post if ts[i] >= PROG_ENABLE_UTC else sig_pre
            es = signal.check_entry(
                bar_open=opens[i], bar_high=highs[i], bar_low=lows[i], bar_close=closes[i],
                channel_high=ch_h[i], channel_low=ch_l[i], atr_val=atr[i],
                last_swing_low=sw_l[i], last_swing_high=sw_h[i])
            if es:
                ni = i + 1
                if ni > e_idx: continue
                pdir = es['direction']
                pprice = opens[ni]
                pt_entry_ts = ts[ni]
                psl = es['sl_price']
                pheld = 0
                in_pos = True
                pbest = highs[ni] if pdir == 'LONG' else lows[ni]
    return trades


# ---------------------------------------------------------------------------
def match_trades(live: list[dict], bt: list[dict], tol_minutes: int = 30) -> list[dict]:
    """Pair LIVE trades with BT trades by entry time. Returns per-trade match records.

    Note: LIVE entry time is derived from exit_time - bars_held*15m; LIVE exit_time
    can be a timestamp of exchange-side event (not bar close). So entry tolerance
    may need ±30min.
    """
    used = set()
    out = []
    for lt in live:
        best_j = -1
        best_gap = timedelta.max
        for j, bt_t in enumerate(bt):
            if j in used: continue
            if bt_t['dir'] != lt['direction']: continue
            bt_et = datetime.fromisoformat(bt_t['entry_time'])
            gap = abs(bt_et - lt['entry_time_utc'])
            # Price sanity: entries within 1% ensures same signal, not a different trade
            price_diff_pct = abs(bt_t['entry_price'] - lt['entry_price']) / lt['entry_price'] * 100
            if gap < best_gap and gap <= timedelta(minutes=tol_minutes) and price_diff_pct < 1.0:
                best_gap = gap
                best_j = j
        if best_j >= 0:
            used.add(best_j)
            out.append({'live': lt, 'bt': bt[best_j], 'matched': True,
                        'entry_gap_sec': int(best_gap.total_seconds())})
        else:
            out.append({'live': lt, 'bt': None, 'matched': False, 'entry_gap_sec': None})

    # Also report unmatched BT trades
    bt_extra = [bt[j] for j in range(len(bt)) if j not in used]
    return out, bt_extra


# ---------------------------------------------------------------------------
def classify_pair(rec: dict) -> dict:
    """Classify a matched LIVE-BT pair.

    A: exit prices within 0.15% AND reason semantically same
    B: LIVE TRAILING fired intrabar at worse price than BT would (BT still held or exited later at better)
    C: LIVE exited with loss while BT continued to profit (or vice versa)
    D: Reason mismatch — LIVE EXCHANGE_SL but actually profitable (baton STOP triggered above entry)
       versus BT TRAIL_TP label. Economically equivalent — flag separately.
    """
    if not rec['matched']:
        return {'class': 'UNMATCHED', 'note': 'no BT counterpart'}

    lt = rec['live']; bt = rec['bt']
    is_long = lt['direction'] == 'LONG'
    # exit slip pct (LIVE vs BT) in direction-favorable terms (positive = LIVE worse)
    if is_long:
        exit_slip_pct = (bt['exit_price'] - lt['exit_price']) / bt['exit_price'] * 100  # LIVE lower → positive slip
        entry_slip_pct = (lt['entry_price'] - bt['entry_price']) / bt['entry_price'] * 100  # LIVE higher → positive slip
    else:
        exit_slip_pct = (lt['exit_price'] - bt['exit_price']) / bt['exit_price'] * 100  # LIVE higher → worse for SHORT
        entry_slip_pct = (bt['entry_price'] - lt['entry_price']) / bt['entry_price'] * 100

    # LIVE pnl_pct is already 3x net; BT pnl_3x is 3x net → same unit
    live_pnl_3x = lt['pnl_pct']
    bt_pnl_3x = bt['pnl_3x']
    delta_3x = live_pnl_3x - bt_pnl_3x  # negative = LIVE worse

    # Reason comparison
    # LIVE reasons: EXCHANGE_SL (STOP_MARKET triggered — either initial SL or baton post-activation),
    #               EXCHANGE_TRAIL (TRAILING_STOP_MARKET pre-activation),
    #               TRAIL_TP (bot cycle check_exit fired — rare in this period, typically break-even)
    # BT reasons: SL, EMERGENCY, TIMEOUT, TRAIL_TP
    live_r = lt['reason']
    bt_r = bt['reason']

    reason_label = f"{live_r} vs {bt_r}"

    # Class D: LIVE EXCHANGE_SL but positive pnl → baton/trail stepped SL up above entry
    if live_r == 'EXCHANGE_SL' and live_pnl_3x > 0 and bt_r == 'TRAIL_TP':
        cls = 'D'
        note = 'LIVE SL stepped above entry (baton) ≈ BT TRAIL_TP'
    elif live_r == 'TRAIL_TP' and bt_r == 'TRAIL_TP' and abs(delta_3x) < 0.5:
        cls = 'A'
        note = 'both cycle-TRAIL_TP, close match'
    elif abs(delta_3x) < 1.0 and (
        (live_r == 'EXCHANGE_TRAIL' and bt_r in ('TRAIL_TP',)) or
        (live_r == bt_r)
    ):
        cls = 'A'
        note = 'exit outcome within 1pp; minor timing/slip'
    elif delta_3x <= -1.0 and live_r == 'EXCHANGE_TRAIL' and bt_r in ('TRAIL_TP', 'TIMEOUT'):
        # LIVE trailing fired and was worse than BT would have been
        if bt_pnl_3x > 0 > live_pnl_3x:
            cls = 'C'
            note = 'LIVE TRAILING took loss; BT would have profited'
        else:
            cls = 'B'
            note = 'LIVE TRAILING tighter than BT trail'
    elif delta_3x <= -1.0 and live_r == 'EXCHANGE_SL' and bt_r == 'TRAIL_TP':
        cls = 'C'
        note = 'LIVE SL hit; BT TRAIL_TP would be better'
    elif delta_3x <= -1.0:
        cls = 'C'
        note = f'significant loss vs BT ({reason_label})'
    elif delta_3x >= 1.0:
        cls = 'A+'
        note = f'LIVE better than BT (+{delta_3x:.2f}pp)'
    else:
        cls = 'A'
        note = 'minor delta'

    return {
        'class': cls,
        'note': note,
        'entry_slip_pct': round(entry_slip_pct, 4),
        'exit_slip_pct': round(exit_slip_pct, 4),
        'live_pnl_3x': live_pnl_3x,
        'bt_pnl_3x': bt_pnl_3x,
        'delta_3x': round(delta_3x, 4),
        'reason_compare': reason_label,
    }


# ---------------------------------------------------------------------------
def attribute_gap(pairs: list[dict]) -> dict:
    """Decompose total gap into components."""
    total_live = sum(p['live']['pnl_pct'] for p in pairs)  # 3x
    total_bt = sum(p['bt']['pnl_3x'] for p in pairs if p['matched'])
    total_live_matched = sum(p['live']['pnl_pct'] for p in pairs if p['matched'])

    gap_matched_3x = total_live_matched - total_bt
    gap_total_3x = total_live - total_bt

    # --------------------------------------------------------------------
    # Attribution (corrected per advisor):
    #   "exit price diff" is NOT the same as "execution slippage". It conflates:
    #     (a) true execution slip (TRAILING/STOP adverse fill: 0.007-0.008% measured)
    #     (b) structural timing gap (LIVE TRAILING fires intrabar at tight tick;
    #         BT check_exit evaluates at bar close, where price often recovered)
    #   We report both with the (a) floor pulled from slippage_diagnosis_20260422.
    # --------------------------------------------------------------------
    n_matched = sum(1 for p in pairs if p['matched'])
    sum_entry_slip_1x = sum(p['cls']['entry_slip_pct'] for p in pairs if p['matched'])
    sum_exit_price_diff_1x = sum(p['cls']['exit_slip_pct'] for p in pairs if p['matched'])

    # Sign convention: positive slip_pct = LIVE worse → negative contribution to gap
    entry_slip_contrib_3x   = -sum_entry_slip_1x     * LEVERAGE
    exit_price_diff_3x      = -sum_exit_price_diff_1x * LEVERAGE

    # Split exit price diff into true execution slip + structural timing
    # Only trades where trail-class exit fired had execution slip applied
    n_trail_like = sum(1 for p in pairs if p['matched'] and
                       p['live']['reason'] in ('EXCHANGE_TRAIL', 'EXCHANGE_SL', 'TRAIL_TP'))
    true_exit_slip_3x = -TRUE_EXIT_SLIP_1X_PER_TRADE * LEVERAGE * n_trail_like
    structural_timing_3x = exit_price_diff_3x - true_exit_slip_3x

    # Reason-mismatch: class-D trades (economically equivalent but labeled differently)
    reason_mismatch_delta = sum(p['cls']['delta_3x'] for p in pairs
                                if p['matched'] and p['cls']['class'] == 'D')

    # Sanity residual (should be ~0)
    check_sum = entry_slip_contrib_3x + exit_price_diff_3x
    residual = gap_matched_3x - check_sum

    return {
        'n_pairs': len(pairs),
        'n_matched': n_matched,
        'n_unmatched_live': sum(1 for p in pairs if not p['matched']),
        'total_live_pnl_3x_matched_only': round(total_live_matched, 2),
        'total_live_pnl_3x_all_in_window': round(total_live, 2),
        'total_bt_pnl_3x_matched_only':   round(total_bt, 2),
        'gap_matched_3x':  round(gap_matched_3x, 2),
        'gap_total_3x':    round(gap_total_3x, 2),
        # --- Raw price diffs ---
        'sum_entry_price_diff_1x_pct': round(sum_entry_slip_1x, 3),
        'sum_exit_price_diff_1x_pct':  round(sum_exit_price_diff_1x, 3),
        # --- 3x contribution components ---
        'entry_slip_contrib_3x_pp':      round(entry_slip_contrib_3x, 2),
        'exit_price_diff_contrib_3x_pp': round(exit_price_diff_3x, 2),
        'true_exit_slip_3x_pp_floor':    round(true_exit_slip_3x, 3),
        'structural_timing_3x_pp':       round(structural_timing_3x, 2),
        # --- Sanity ---
        'decomp_residual_3x_pp':         round(residual, 2),
        'reason_mismatch_D_delta_3x':    round(reason_mismatch_delta, 2),
        # --- Diagnostic ---
        'n_trail_like_trades_for_slip': n_trail_like,
    }


def f_v2_counterfactual(pairs: list[dict]) -> dict:
    """F v2 runs check_exit every cycle and MARKETs out on TRAIL_TP — mathematically
    identical to BT semantic (modulo ~0.02pp execution slip per trade).

    NOTE: For matched trades this is a tautology (fv2_gain ≡ -delta). The meaningful
    output is: (a) how many trades have delta > threshold → 'F v2 addresses it';
    (b) unmatched LIVE trades are OUTSIDE F v2's reach (entry-sequencing divergence),
    constituting the irreducible gap floor even with F v2 deployed."""
    improve_count = 0
    worse_count = 0
    same_count = 0
    total_gain = 0.0
    details = []
    for p in pairs:
        if not p['matched']:
            continue
        delta = p['cls']['delta_3x']  # live - bt
        # F v2 would change LIVE outcome to BT outcome. Gain for LIVE = -delta (if delta<0, gain>0)
        fv2_gain = -delta
        if fv2_gain > 0.3:
            improve_count += 1
        elif fv2_gain < -0.3:
            worse_count += 1
        else:
            same_count += 1
        total_gain += fv2_gain
        details.append({
            'entry_time': p['live']['entry_time_utc'].isoformat(),
            'dir': p['live']['direction'],
            'live_pnl_3x': p['live']['pnl_pct'],
            'bt_pnl_3x': p['bt']['pnl_3x'],
            'fv2_gain_3x': round(fv2_gain, 3),
            'class': p['cls']['class'],
        })
    # Unmatched LIVE trades: F v2 cannot fix these — entry-sequencing divergence.
    # Their LIVE pnl is an irreducible gap floor.
    unmatched_live_pnl_3x = sum(p['live']['pnl_pct'] for p in pairs if not p['matched'])
    return {
        'fv2_addresses_count':       improve_count,   # matched trades where BT > LIVE
        'fv2_would_worsen_count':    worse_count,     # matched trades where BT < LIVE
        'fv2_neutral_count':         same_count,      # |delta| < 0.3pp
        'matched_gap_closable_by_fv2_3x': round(total_gain, 2),
        'unmatched_live_pnl_3x':     round(unmatched_live_pnl_3x, 2),
        'n_unmatched_live':          sum(1 for p in pairs if not p['matched']),
        'note': 'matched_gap_closable == -gap_matched_3x by construction; '
                'unmatched trades are outside F v2 reach.',
        'per_trade': details,
    }


# ---------------------------------------------------------------------------
def main():
    print("Loading LIVE trades (04-18 12:00 ~ 04-22 11:00 UTC)...")
    live = load_live_trades()
    print(f"  {len(live)} LIVE trades")

    print("Fetching BingX 15m candles (04-12 ~ 04-22 12:00 UTC)...")
    candles = fetch_candles()
    print(f"  {len(candles)} candles")

    print("Running BT (time-sliced progressive_trail)...")
    bt_all = run_bt(candles)
    # Filter BT trades that could match LIVE entries in this window (LIVE#1 entered 04-17)
    bt = [t for t in bt_all
          if datetime.fromisoformat(t['entry_time']) >= WINDOW_START - timedelta(days=2) and
             datetime.fromisoformat(t['entry_time']) < WINDOW_END]
    print(f"  {len(bt_all)} BT trades total, {len(bt)} within window")

    print("Matching LIVE <-> BT...")
    pairs_raw, bt_extra = match_trades(live, bt, tol_minutes=60)
    pairs = []
    for rec in pairs_raw:
        cls = classify_pair(rec)
        pairs.append({**rec, 'cls': cls})
    print(f"  matched={sum(1 for p in pairs if p['matched'])}, "
          f"unmatched_live={sum(1 for p in pairs if not p['matched'])}, "
          f"unmatched_bt={len(bt_extra)}")

    # Classification table
    print("\n" + "=" * 110)
    print(f"{'#':>2} {'dir':<5} {'entry(LIVE)':>11} {'entry(BT)':>10} {'exit(LIVE)':>10} {'exit(BT)':>10} "
          f"{'pnl_L':>7} {'pnl_B':>7} {'Δ':>6} {'cls':<4} {'note':<40}")
    print("-" * 110)
    for i, p in enumerate(pairs):
        lt = p['live']
        if p['matched']:
            bt_t = p['bt']
            print(f"{i+1:>2} {lt['direction']:<5} {lt['entry_price']:>11.1f} {bt_t['entry_price']:>10.1f} "
                  f"{lt['exit_price']:>10.1f} {bt_t['exit_price']:>10.1f} "
                  f"{lt['pnl_pct']:>+7.2f} {bt_t['pnl_3x']:>+7.2f} {p['cls']['delta_3x']:>+6.2f} "
                  f"{p['cls']['class']:<4} {p['cls']['note'][:40]:<40}")
        else:
            print(f"{i+1:>2} {lt['direction']:<5} {lt['entry_price']:>11.1f} {'--':>10} "
                  f"{lt['exit_price']:>10.1f} {'--':>10} "
                  f"{lt['pnl_pct']:>+7.2f} {'--':>7} {'--':>6} UNMAT  no BT match")
    print("=" * 110)

    # Class counts
    cls_counts: dict[str, int] = {}
    for p in pairs:
        c = p['cls']['class']
        cls_counts[c] = cls_counts.get(c, 0) + 1
    print("\nClassification counts:")
    for c, n in sorted(cls_counts.items()):
        print(f"  {c}: {n}")

    # Attribution
    attr = attribute_gap(pairs)
    print("\nGap Attribution (3x, pp):")
    for k, v in attr.items():
        print(f"  {k}: {v}")

    # F v2 counterfactual
    fv2 = f_v2_counterfactual(pairs)
    print(f"\nF v2 Counterfactual:")
    print(f"  fv2_addresses_count  (BT > LIVE by >0.3pp): {fv2['fv2_addresses_count']}")
    print(f"  fv2_would_worsen     (BT < LIVE by >0.3pp): {fv2['fv2_would_worsen_count']}")
    print(f"  fv2_neutral          (|delta| <= 0.3pp):    {fv2['fv2_neutral_count']}")
    print(f"  matched_gap_closable_by_fv2_3x:              {fv2['matched_gap_closable_by_fv2_3x']:+.2f}pp")
    print(f"  unmatched_live_pnl_3x (outside F v2 reach):  {fv2['unmatched_live_pnl_3x']:+.2f}pp ({fv2['n_unmatched_live']} trades)")

    # Deep-dive on worst matched trade (biggest |delta|)
    matched = [p for p in pairs if p['matched']]
    if matched:
        worst = max(matched, key=lambda p: abs(p['cls']['delta_3x']))
        lt = worst['live']; bt_t = worst['bt']
        print(f"\n[WORST MATCHED TRADE — deep-dive]")
        print(f"  dir={lt['direction']} entry_time={lt['entry_time_utc'].isoformat()}")
        print(f"  LIVE: entry={lt['entry_price']:.1f} exit={lt['exit_price']:.1f} "
              f"pnl3x={lt['pnl_pct']:+.2f}% reason={lt['reason']} bars={lt['bars_held']}")
        print(f"  BT:   entry={bt_t['entry_price']:.1f} exit={bt_t['exit_price']:.1f} "
              f"pnl3x={bt_t['pnl_3x']:+.2f}% reason={bt_t['reason']} bars={bt_t['bars']}")
        print(f"  delta_3x: {worst['cls']['delta_3x']:+.2f}pp")
        print(f"  class: {worst['cls']['class']} — {worst['cls']['note']}")

    # Extra BT trades that weren't matched
    if bt_extra:
        print(f"\nUnmatched BT trades ({len(bt_extra)}):")
        for t in bt_extra:
            print(f"  {t['dir']:<5} entry={t['entry_price']:>9.1f} @ {t['entry_time']}  pnl3x={t['pnl_3x']:+.2f}")

    out = {
        'window_utc': [WINDOW_START.isoformat(), WINDOW_END.isoformat()],
        'prog_flip_utc': PROG_ENABLE_UTC.isoformat(),
        'n_live': len(live),
        'n_bt_window': len(bt),
        'n_bt_all': len(bt_all),
        'class_counts': cls_counts,
        'attribution': attr,
        'fv2_counterfactual': fv2,
        'pairs': [
            {
                'live': {
                    'direction': p['live']['direction'],
                    'entry_price': p['live']['entry_price'],
                    'exit_price': p['live']['exit_price'],
                    'pnl_3x': p['live']['pnl_pct'],
                    'reason': p['live']['reason'],
                    'bars_held': p['live']['bars_held'],
                    'entry_time': p['live']['entry_time_utc'].isoformat(),
                    'exit_time': p['live']['exit_time'],
                },
                'bt': p['bt'],
                'cls': p['cls'],
                'entry_gap_sec': p['entry_gap_sec'],
            } for p in pairs
        ],
        'unmatched_bt': bt_extra,
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }
    outpath = ROOT / 'results' / f'post_fix_legacy_gap_20260424.json'
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {outpath}")


if __name__ == '__main__':
    main()
