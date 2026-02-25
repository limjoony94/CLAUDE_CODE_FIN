"""
72-Hour Signal Analysis — Production Logic Simulation
=====================================================
Fetches 72h of BTC/USDT 5m OHLCV from BingX, runs production candle
classification, and simulates the full N=5 FIFO trading logic including:
- Signal detection (59 patterns: 12L + 47S)
- ATR-scaled per-pattern TP/SL
- Proportional vol_mult cap
- Same-direction slot routing (OPEN / SKIP / CLOSE_OLDEST)
- Duplicate pattern blocking
- Intrabar TP/SL resolution (distance-based)
- 288-bar timeout (DROP)
- 3-consecutive-loss pause
- Daily loss limit (13%)

Output: comprehensive statistics + per-signal detail log.
"""
import sys, json, time, yaml
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import ccxt
import pandas as pd
import numpy as np

sys.path.insert(0, '.')
from scripts.production.pattern_5m.indicators import calculate_indicators, get_volatility_multiplier
from scripts.production.pattern_5m.constants import (
    SLIPPAGE_BUFFER_PCT, MAX_CONSECUTIVE_LOSSES,
)

# ── Config ──────────────────────────────────────────────
LEVERAGE = 3
FEE_PCT = 0.10          # round-trip in price-space
FEE = FEE_PCT * LEVERAGE / 100  # in capital-space fraction
MAX_POSITIONS = 5
MAX_BARS = 288           # 24h timeout
MAX_DAILY_LOSS_PCT = 13
ANALYSIS_HOURS = 72
SLIPPAGE = SLIPPAGE_BUFFER_PCT  # 0.02%

# ── Load config & patterns ──────────────────────────────
with open('config/pattern_5m_config.yaml') as f:
    config = yaml.safe_load(f)

with open('results/dynamic_patterns.json') as f:
    dp = json.load(f)

LONG_PATTERNS = set(dp['patterns']['long'])
SHORT_PATTERNS = set(dp['patterns']['short'])
PATTERNS_TPSL = dp.get('patterns_tpsl', {})

print(f"Patterns: {len(LONG_PATTERNS)}L + {len(SHORT_PATTERNS)}S = {len(LONG_PATTERNS) + len(SHORT_PATTERNS)}")

# ── Fetch OHLCV data ────────────────────────────────────
with open('config/api_keys.yaml') as f:
    api_cfg = yaml.safe_load(f)
mn = api_cfg['bingx']['mainnet']
ex = ccxt.bingx({
    'apiKey': mn['api_key'], 'secret': mn['secret_key'],
    'enableRateLimit': True, 'options': {'defaultType': 'swap'}
})

now_utc = datetime.now(timezone.utc)
# Need 72h of analysis + 600 bars pre-history for indicators
pre_history = 600
total_bars = int(ANALYSIS_HOURS * 12) + pre_history  # 864 + 600 = 1464

# Fetch in batches of 1000 (BingX supports up to 1000)
all_ohlcv = []
since_ts = int((now_utc - timedelta(hours=ANALYSIS_HOURS) - timedelta(minutes=5 * pre_history)).timestamp() * 1000)
now_ts = int(now_utc.timestamp() * 1000)

print(f"Fetching ~{total_bars} bars from {datetime.fromtimestamp(since_ts/1000, tz=timezone.utc):%Y-%m-%d %H:%M} UTC...")

max_attempts = 10
for attempt in range(max_attempts):
    batch = ex.fetch_ohlcv('BTC/USDT:USDT', '5m', since=since_ts, limit=1000)
    if not batch:
        break
    all_ohlcv.extend(batch)
    last_fetched_ts = batch[-1][0]
    since_ts = last_fetched_ts + 1
    time.sleep(0.5)
    # Stop if we've reached current time
    if last_fetched_ts >= now_ts - 300000:  # within 5 min of now
        break
    if len(batch) < 100:
        break

# Deduplicate by timestamp
seen = set()
unique = []
for bar in all_ohlcv:
    if bar[0] not in seen:
        seen.add(bar[0])
        unique.append(bar)
all_ohlcv = sorted(unique, key=lambda x: x[0])

df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
print(f"Fetched {len(df)} bars: {df['time'].iloc[0]} ~ {df['time'].iloc[-1]} UTC")

# ── Run production indicators ───────────────────────────
df_ind = calculate_indicators(df, config)
print(f"Indicators calculated. Columns: {list(df_ind.columns)}")

# ── Determine analysis window ───────────────────────────
analysis_start = now_utc - timedelta(hours=ANALYSIS_HOURS)
analysis_start_ts = analysis_start.timestamp() * 1000
start_idx = df_ind[df_ind['timestamp'] >= analysis_start_ts].index[0]
end_idx = len(df_ind) - 1  # exclude last forming candle

print(f"Analysis window: idx {start_idx} to {end_idx} ({end_idx - start_idx} bars = {(end_idx-start_idx)*5/60:.1f}h)")
print()

# ── ATR scaling helper ──────────────────────────────────
atr_cfg = config.get('strategy', {}).get('atr_scale', {})
ATR_ENABLED = atr_cfg.get('enabled', False)
ATR_WINDOW = atr_cfg.get('window', 576)
ATR_CLAMP_LO = atr_cfg.get('clamp_lo', 0.6)
ATR_CLAMP_HI = atr_cfg.get('clamp_hi', 1.7)


def get_vol_mult(bar_idx):
    """Calculate ATR-based vol multiplier at given bar index."""
    if not ATR_ENABLED or 'atr' not in df_ind.columns:
        return 1.0
    current_atr = df_ind.iloc[bar_idx]['atr']
    if pd.isna(current_atr):
        return 1.0
    window = min(ATR_WINDOW, bar_idx + 1)
    atr_series = df_ind['atr'].iloc[:bar_idx + 1].dropna().tail(window)
    if len(atr_series) < window // 2:
        return 1.0
    median_atr = atr_series.median()
    if median_atr <= 0:
        return 1.0
    ratio = current_atr / median_atr
    return max(ATR_CLAMP_LO, min(ATR_CLAMP_HI, ratio))


def effective_vol_mult(vol_mult, base_sl_pct):
    """Proportional cap."""
    max_sl = MAX_DAILY_LOSS_PCT / LEVERAGE
    max_mult = max_sl / base_sl_pct if base_sl_pct > 0 else vol_mult
    return min(vol_mult, max_mult)


def calc_tp_sl(entry_price, direction, pattern, vol_mult):
    """Calculate TP/SL prices matching production logic."""
    tpsl = PATTERNS_TPSL.get(pattern)
    if tpsl:
        base_tp, base_sl = tpsl[0], tpsl[1]
    else:
        base_tp = config['strategy']['tp_pct']
        base_sl = config['strategy']['sl_pct']
    eff = effective_vol_mult(vol_mult, base_sl)
    tp_adj = base_tp * eff + SLIPPAGE
    sl_adj = max(0.1, base_sl * eff - SLIPPAGE)
    d = 1 if direction == 'LONG' else -1
    tp_price = entry_price * (1 + d * tp_adj / 100)
    sl_price = entry_price * (1 - d * sl_adj / 100)
    return tp_price, sl_price, tp_adj, sl_adj


# ── Simulation state ────────────────────────────────────
class Slot:
    def __init__(self, pattern, direction, entry_price, tp, sl, tp_pct, sl_pct, vol_mult, entry_bar, entry_time):
        self.pattern = pattern
        self.direction = direction
        self.entry_price = entry_price
        self.tp = tp
        self.sl = sl
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.vol_mult = vol_mult
        self.entry_bar = entry_bar
        self.entry_time = entry_time
        self.exit_price = None
        self.exit_bar = None
        self.exit_time = None
        self.exit_reason = None
        self.pnl_pct = None
        self.portfolio_pnl = None


slots = {}  # slot_id -> Slot
slot_counter = 0
active_direction = None

all_signals = []       # every signal detected
taken_signals = []     # signals that resulted in entry
skipped_signals = []   # signals skipped with reason
closed_trades = []     # completed trades

daily_pnl = 0.0
daily_trades = 0
consecutive_losses = 0
current_day = None
last_signal_ts = None

# ── Main simulation loop ────────────────────────────────
for bar_idx in range(start_idx, end_idx):
    row = df_ind.iloc[bar_idx]
    bar_time = row['time']
    bar_ts = row['timestamp']

    # Daily reset (matches state.py _reset_daily_fields)
    bar_day = bar_time.date()
    if current_day is not None and bar_day != current_day:
        daily_pnl = 0.0
        daily_trades = 0
        consecutive_losses = 0  # production resets this on daily boundary
    current_day = bar_day

    # 1. Check TP/SL for active positions (using NEXT bar's OHLCV)
    next_bar = df_ind.iloc[bar_idx + 1] if bar_idx + 1 < len(df_ind) else None
    if next_bar is not None:
        slots_to_close = []
        for sid, s in list(slots.items()):
            h = next_bar['high']
            l = next_bar['low']
            o = next_bar['open']

            if s.direction == 'LONG':
                tp_hit = h >= s.tp
                sl_hit = l <= s.sl
            else:
                tp_hit = l <= s.tp
                sl_hit = h >= s.sl

            if tp_hit and sl_hit:
                # Same-bar resolution: distance from bar open
                tp_dist = abs(s.tp - o)
                sl_dist = abs(s.sl - o)
                if tp_dist <= sl_dist:
                    tp_hit, sl_hit = True, False
                else:
                    tp_hit, sl_hit = False, True

            if tp_hit:
                d = 1 if s.direction == 'LONG' else -1
                pnl = d * (s.tp / s.entry_price - 1) * 100 * LEVERAGE - FEE_PCT * LEVERAGE / 100 * 100
                # Actually: pnl_pct = direction * (exit/entry - 1) * 100 * leverage - fee_pct * leverage
                price_move = d * (s.tp - s.entry_price) / s.entry_price * 100
                pnl = price_move * LEVERAGE - FEE_PCT * LEVERAGE / 100 * 100
                port_pnl = pnl / MAX_POSITIONS
                s.exit_price = s.tp
                s.exit_bar = bar_idx + 1
                s.exit_time = next_bar['time']
                s.exit_reason = 'TP'
                s.pnl_pct = pnl
                s.portfolio_pnl = port_pnl
                slots_to_close.append(sid)
            elif sl_hit:
                d = 1 if s.direction == 'LONG' else -1
                price_move = d * (s.sl - s.entry_price) / s.entry_price * 100
                pnl = price_move * LEVERAGE - FEE_PCT * LEVERAGE / 100 * 100
                port_pnl = pnl / MAX_POSITIONS
                s.exit_price = s.sl
                s.exit_bar = bar_idx + 1
                s.exit_time = next_bar['time']
                s.exit_reason = 'SL'
                s.pnl_pct = pnl
                s.portfolio_pnl = port_pnl
                slots_to_close.append(sid)

        for sid in slots_to_close:
            s = slots.pop(sid)
            closed_trades.append(s)
            daily_pnl += s.portfolio_pnl
            daily_trades += 1
            if s.pnl_pct >= 0:
                consecutive_losses = 0
            else:
                consecutive_losses += 1
            if len(slots) == 0:
                active_direction = None

    # 2. Check timeouts (288 bars)
    for sid in list(slots.keys()):
        s = slots[sid]
        if bar_idx - s.entry_bar >= MAX_BARS:
            # DROP: close at current bar's close
            d = 1 if s.direction == 'LONG' else -1
            price_move = d * (row['close'] - s.entry_price) / s.entry_price * 100
            pnl = price_move * LEVERAGE - FEE_PCT * LEVERAGE / 100 * 100
            port_pnl = pnl / MAX_POSITIONS
            s.exit_price = row['close']
            s.exit_bar = bar_idx
            s.exit_time = bar_time
            s.exit_reason = 'TIMEOUT'
            s.pnl_pct = pnl
            s.portfolio_pnl = port_pnl
            closed_trades.append(slots.pop(sid))
            daily_pnl += port_pnl
            daily_trades += 1
            if pnl >= 0:
                consecutive_losses = 0
            else:
                consecutive_losses += 1
            if len(slots) == 0:
                active_direction = None

    # 3. Signal detection on last completed candle (df.iloc[-2] equivalent)
    # In our loop, bar_idx is the "last completed candle"
    pattern = row.get('pattern_3')
    if not pattern:
        continue

    # Prevent duplicate signal on same candle
    if bar_ts == last_signal_ts:
        continue

    # Check pattern match
    if pattern in LONG_PATTERNS:
        sig_dir = 'LONG'
    elif pattern in SHORT_PATTERNS:
        sig_dir = 'SHORT'
    else:
        continue

    # Record signal
    signal_info = {
        'time': bar_time,
        'pattern': pattern,
        'direction': sig_dir,
        'bar_idx': bar_idx,
        'price': row['close'],
    }
    all_signals.append(signal_info)
    last_signal_ts = bar_ts

    # ─── Routing logic ────────────────────────────
    # Check duplicate pattern
    active_patterns = {s.pattern for s in slots.values()}
    if pattern in active_patterns:
        skipped_signals.append({**signal_info, 'reason': 'DUPLICATE_PATTERN'})
        continue

    # Check consecutive losses (production: pause + reset when no positions)
    # bot.py:281-288: only triggers when consecutive_losses >= 3 AND no positions open
    # After pause, resets consecutive_losses = 0
    # If positions still open, bot does NOT check this → signals flow normally
    if consecutive_losses >= MAX_CONSECUTIVE_LOSSES and len(slots) == 0:
        skipped_signals.append({**signal_info, 'reason': f'CONSECUTIVE_LOSS_PAUSE_{consecutive_losses}'})
        consecutive_losses = 0  # models bot.py:286 reset after pause
        continue
    # Note: if consecutive_losses >= 3 but slots > 0, signal is allowed (matches production)

    # Check daily loss limit
    if daily_pnl <= -MAX_DAILY_LOSS_PCT:
        skipped_signals.append({**signal_info, 'reason': f'DAILY_LIMIT_{daily_pnl:.1f}%'})
        continue

    # Route signal
    n_slots = len(slots)

    if n_slots == 0:
        # Empty: just open
        action = 'OPEN'
    elif active_direction == sig_dir:
        # Same direction
        if n_slots < MAX_POSITIONS:
            action = 'OPEN'
        else:
            action = 'SKIP_FULL'
    else:
        # Opposite direction: CLOSE_OLDEST
        action = 'CLOSE_OLDEST'

    if action == 'SKIP_FULL':
        skipped_signals.append({**signal_info, 'reason': 'SLOTS_FULL'})
        continue

    if action == 'CLOSE_OLDEST':
        # Find oldest slot
        oldest_sid = min(slots.keys(), key=lambda s: slots[s].entry_bar)
        oldest = slots.pop(oldest_sid)
        # Close at next bar open (simulate market close)
        if bar_idx + 1 < len(df_ind):
            exit_price = df_ind.iloc[bar_idx + 1]['open']
        else:
            exit_price = row['close']
        d = 1 if oldest.direction == 'LONG' else -1
        price_move = d * (exit_price - oldest.entry_price) / oldest.entry_price * 100
        pnl = price_move * LEVERAGE - FEE_PCT * LEVERAGE / 100 * 100
        port_pnl = pnl / MAX_POSITIONS
        oldest.exit_price = exit_price
        oldest.exit_bar = bar_idx
        oldest.exit_time = bar_time
        oldest.exit_reason = 'CLOSE_OLDEST'
        oldest.pnl_pct = pnl
        oldest.portfolio_pnl = port_pnl
        closed_trades.append(oldest)
        daily_pnl += port_pnl
        if pnl >= 0:
            consecutive_losses = 0
        else:
            consecutive_losses += 1

        if len(slots) == 0:
            active_direction = None
            # Re-route: now open in new direction
            action = 'OPEN'
        else:
            # Still have same-direction slots, skip the opposite signal
            skipped_signals.append({**signal_info, 'reason': 'CLOSE_OLDEST_PARTIAL'})
            continue

    if action == 'OPEN':
        # Entry on next bar open
        if bar_idx + 1 >= len(df_ind):
            skipped_signals.append({**signal_info, 'reason': 'NO_NEXT_BAR'})
            continue
        entry_price = df_ind.iloc[bar_idx + 1]['open']
        vm = get_vol_mult(bar_idx + 1)
        tp, sl, tp_pct, sl_pct = calc_tp_sl(entry_price, sig_dir, pattern, vm)

        slot_counter += 1
        sid = f's{slot_counter}'
        new_slot = Slot(
            pattern=pattern, direction=sig_dir,
            entry_price=entry_price, tp=tp, sl=sl,
            tp_pct=tp_pct, sl_pct=sl_pct, vol_mult=vm,
            entry_bar=bar_idx + 1, entry_time=df_ind.iloc[bar_idx + 1]['time']
        )
        slots[sid] = new_slot
        active_direction = sig_dir
        taken_signals.append({**signal_info, 'entry_price': entry_price, 'vol_mult': vm,
                              'tp': tp, 'sl': sl, 'tp_pct': tp_pct, 'sl_pct': sl_pct})

# ── Close still-open positions at last bar ──────────────
for sid, s in slots.items():
    last = df_ind.iloc[end_idx]
    d = 1 if s.direction == 'LONG' else -1
    price_move = d * (last['close'] - s.entry_price) / s.entry_price * 100
    pnl = price_move * LEVERAGE - FEE_PCT * LEVERAGE / 100 * 100
    port_pnl = pnl / MAX_POSITIONS
    s.exit_price = last['close']
    s.exit_bar = end_idx
    s.exit_time = last['time']
    s.exit_reason = 'STILL_OPEN'
    s.pnl_pct = pnl
    s.portfolio_pnl = port_pnl
    closed_trades.append(s)

# ═══════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════
print("=" * 70)
print(f"  72-HOUR SIGNAL ANALYSIS ({ANALYSIS_HOURS}h)")
print(f"  Period: {df_ind.iloc[start_idx]['time']} ~ {df_ind.iloc[end_idx]['time']} UTC")
print(f"  Bars analyzed: {end_idx - start_idx}")
print("=" * 70)

# Section 1: Signal overview
print("\n── SECTION 1: SIGNAL OVERVIEW ──")
print(f"  Total signals detected:  {len(all_signals)}")
print(f"  Signals taken (entered): {len(taken_signals)}")
print(f"  Signals skipped:         {len(skipped_signals)}")
hit_rate = len(taken_signals) / len(all_signals) * 100 if all_signals else 0
print(f"  Entry rate:              {hit_rate:.1f}%")

# Signal direction breakdown
long_sigs = [s for s in all_signals if s['direction'] == 'LONG']
short_sigs = [s for s in all_signals if s['direction'] == 'SHORT']
print(f"  LONG signals:  {len(long_sigs)} | SHORT signals: {len(short_sigs)}")
print(f"  Signal per bar: {len(all_signals) / (end_idx - start_idx) * 100:.1f}%")

# Section 2: Skip reasons
print("\n── SECTION 2: SKIP REASONS ──")
skip_reasons = defaultdict(int)
for s in skipped_signals:
    skip_reasons[s['reason']] += 1
for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
    pct = count / len(skipped_signals) * 100 if skipped_signals else 0
    print(f"  {reason:30s}: {count:4d} ({pct:.1f}%)")

# Section 3: Closed trades
print("\n── SECTION 3: TRADE RESULTS ──")
completed = [t for t in closed_trades if t.exit_reason != 'STILL_OPEN']
still_open = [t for t in closed_trades if t.exit_reason == 'STILL_OPEN']

tp_trades = [t for t in completed if t.exit_reason == 'TP']
sl_trades = [t for t in completed if t.exit_reason == 'SL']
timeout_trades = [t for t in completed if t.exit_reason == 'TIMEOUT']
oldest_trades = [t for t in completed if t.exit_reason == 'CLOSE_OLDEST']

print(f"  Completed trades: {len(completed)}")
print(f"    TP hits:        {len(tp_trades)}")
print(f"    SL hits:        {len(sl_trades)}")
print(f"    Timeouts:       {len(timeout_trades)}")
print(f"    CLOSE_OLDEST:   {len(oldest_trades)}")
print(f"  Still open:       {len(still_open)}")

if completed:
    win_trades = [t for t in completed if t.pnl_pct >= 0]
    loss_trades = [t for t in completed if t.pnl_pct < 0]
    wr = len(win_trades) / len(completed) * 100
    avg_win = np.mean([t.pnl_pct for t in win_trades]) if win_trades else 0
    avg_loss = np.mean([abs(t.pnl_pct) for t in loss_trades]) if loss_trades else 0
    total_pnl_slot = sum(t.pnl_pct for t in completed)
    total_pnl_port = sum(t.portfolio_pnl for t in completed)

    print(f"\n  Win rate:         {wr:.1f}% ({len(win_trades)}W / {len(loss_trades)}L)")
    print(f"  Avg win (slot):   +{avg_win:.2f}%")
    print(f"  Avg loss (slot):  -{avg_loss:.2f}%")
    print(f"  R:R ratio:        {avg_win/avg_loss:.3f}" if avg_loss > 0 else "  R:R ratio:        N/A")
    print(f"  Total PnL (slot): {total_pnl_slot:+.2f}%")
    print(f"  Total PnL (port): {total_pnl_port:+.2f}%")
    edge_per_trade = total_pnl_port / len(completed) if completed else 0
    print(f"  Edge/trade (port):{edge_per_trade:+.3f}%")

# Section 4: By exit reason PnL
print("\n── SECTION 4: PnL BY EXIT REASON ──")
for reason, trades in [('TP', tp_trades), ('SL', sl_trades), ('TIMEOUT', timeout_trades), ('CLOSE_OLDEST', oldest_trades)]:
    if trades:
        pnls = [t.pnl_pct for t in trades]
        print(f"  {reason:15s}: n={len(trades):3d} | avg={np.mean(pnls):+.2f}% | total={sum(pnls):+.2f}% | range=[{min(pnls):+.2f}, {max(pnls):+.2f}]")

# Section 5: Directional breakdown
print("\n── SECTION 5: DIRECTIONAL BREAKDOWN ──")
for direction in ['LONG', 'SHORT']:
    dir_trades = [t for t in completed if t.direction == direction]
    if dir_trades:
        dir_wins = [t for t in dir_trades if t.pnl_pct >= 0]
        dir_wr = len(dir_wins) / len(dir_trades) * 100
        dir_pnl = sum(t.portfolio_pnl for t in dir_trades)
        print(f"  {direction:6s}: {len(dir_trades):3d} trades | WR {dir_wr:.1f}% | PnL(port) {dir_pnl:+.2f}%")

# Section 6: Pattern frequency
print("\n── SECTION 6: TOP PATTERNS (by signal count) ──")
pattern_signals = defaultdict(int)
pattern_taken = defaultdict(int)
pattern_results = defaultdict(list)

for s in all_signals:
    pattern_signals[s['pattern']] += 1
for s in taken_signals:
    pattern_taken[s['pattern']] += 1
for t in completed:
    pattern_results[t.pattern].append(t)

sorted_patterns = sorted(pattern_signals.items(), key=lambda x: -x[1])[:20]
print(f"  {'Pattern':15s} {'Signals':>8s} {'Taken':>6s} {'Trades':>7s} {'WR':>6s} {'AvgPnL':>8s}")
for pat, count in sorted_patterns:
    taken = pattern_taken.get(pat, 0)
    results = pattern_results.get(pat, [])
    if results:
        wr = len([t for t in results if t.pnl_pct >= 0]) / len(results) * 100
        avg_pnl = np.mean([t.pnl_pct for t in results])
        print(f"  {pat:15s} {count:8d} {taken:6d} {len(results):7d} {wr:5.1f}% {avg_pnl:+7.2f}%")
    else:
        print(f"  {pat:15s} {count:8d} {taken:6d}       0     -        -")

# Section 7: Vol mult distribution
print("\n── SECTION 7: VOLATILITY MULTIPLIER ──")
vol_mults = [s['vol_mult'] for s in taken_signals]
if vol_mults:
    print(f"  Mean:   {np.mean(vol_mults):.3f}x")
    print(f"  Median: {np.median(vol_mults):.3f}x")
    print(f"  Min:    {min(vol_mults):.3f}x")
    print(f"  Max:    {max(vol_mults):.3f}x")
    print(f"  Std:    {np.std(vol_mults):.3f}")

# Section 8: Timing analysis
print("\n── SECTION 8: SIGNAL TIMING ──")
if len(all_signals) > 1:
    times = [s['time'] for s in all_signals]
    gaps = [(times[i+1] - times[i]).total_seconds() / 60 for i in range(len(times)-1)]
    print(f"  Avg gap between signals: {np.mean(gaps):.1f} min")
    print(f"  Min gap:                 {min(gaps):.1f} min")
    print(f"  Max gap:                 {max(gaps):.1f} min")
    print(f"  Median gap:              {np.median(gaps):.1f} min")

# Section 9: Hold time analysis
print("\n── SECTION 9: HOLD TIME ──")
if completed:
    hold_bars = [(t.exit_bar - t.entry_bar) for t in completed]
    hold_mins = [b * 5 for b in hold_bars]
    print(f"  Avg hold:    {np.mean(hold_mins):.0f} min ({np.mean(hold_bars):.0f} bars)")
    print(f"  Median hold: {np.median(hold_mins):.0f} min ({np.median(hold_bars):.0f} bars)")
    print(f"  Min hold:    {min(hold_mins):.0f} min")
    print(f"  Max hold:    {max(hold_mins):.0f} min")

    for reason in ['TP', 'SL', 'TIMEOUT', 'CLOSE_OLDEST']:
        reason_trades = [t for t in completed if t.exit_reason == reason]
        if reason_trades:
            bars = [(t.exit_bar - t.entry_bar) for t in reason_trades]
            print(f"  {reason:15s}: avg {np.mean(bars)*5:.0f}min | median {np.median(bars)*5:.0f}min")

# Section 10: Equity curve snapshot
print("\n── SECTION 10: EQUITY CURVE (cumulative portfolio PnL) ──")
if completed:
    sorted_trades = sorted(completed, key=lambda t: t.exit_bar)
    cum_pnl = 0
    max_pnl = 0
    max_dd = 0
    equity_points = []
    for t in sorted_trades:
        if t.exit_reason == 'STILL_OPEN':
            continue
        cum_pnl += t.portfolio_pnl
        max_pnl = max(max_pnl, cum_pnl)
        dd = max_pnl - cum_pnl
        max_dd = max(max_dd, dd)
        equity_points.append((t.exit_time, cum_pnl, dd))

    if equity_points:
        print(f"  Final PnL:  {cum_pnl:+.2f}%")
        print(f"  Max PnL:    {max_pnl:+.2f}%")
        print(f"  Max DD:     {max_dd:.2f}%")
        if max_dd > 0:
            print(f"  PnL/MDD:    {abs(cum_pnl)/max_dd:.2f}x")

        # Show 5 snapshots
        step = max(1, len(equity_points) // 5)
        print(f"  Snapshots:")
        for i in range(0, len(equity_points), step):
            t, p, d = equity_points[i]
            print(f"    {t}: PnL={p:+.2f}% DD={d:.2f}%")
        if equity_points[-1] != equity_points[min(len(equity_points)-1, (len(equity_points)//step)*step)]:
            t, p, d = equity_points[-1]
            print(f"    {t}: PnL={p:+.2f}% DD={d:.2f}%")

# Section 11: Still-open positions
print("\n── SECTION 11: CURRENTLY OPEN POSITIONS ──")
for t in still_open:
    d = 1 if t.direction == 'LONG' else -1
    unrealized = d * (t.exit_price - t.entry_price) / t.entry_price * 100 * LEVERAGE
    bars_held = end_idx - t.entry_bar
    print(f"  {t.pattern:12s} {t.direction:5s} @ ${t.entry_price:.1f} | TP ${t.tp:.1f} SL ${t.sl:.1f} | "
          f"Unrealized: {unrealized:+.2f}% | Held: {bars_held*5}min | VM: {t.vol_mult:.3f}")

# Section 12: Trade log
print("\n── SECTION 12: FULL TRADE LOG (completed) ──")
sorted_completed = sorted([t for t in closed_trades if t.exit_reason != 'STILL_OPEN'], key=lambda t: t.entry_bar)
for i, t in enumerate(sorted_completed):
    hold_min = (t.exit_bar - t.entry_bar) * 5
    print(f"  #{i+1:3d} | {t.entry_time} | {t.direction:5s} {t.pattern:12s} | "
          f"${t.entry_price:.1f}→${t.exit_price:.1f} | {t.exit_reason:14s} | "
          f"Slot {t.pnl_pct:+.2f}% Port {t.portfolio_pnl:+.2f}% | {hold_min}min | VM {t.vol_mult:.2f}")

print("\n" + "=" * 70)
print("  ANALYSIS COMPLETE")
print("=" * 70)

# Save results to JSON
results = {
    'analysis_period_h': ANALYSIS_HOURS,
    'analysis_start': str(df_ind.iloc[start_idx]['time']),
    'analysis_end': str(df_ind.iloc[end_idx]['time']),
    'bars_analyzed': end_idx - start_idx,
    'total_signals': len(all_signals),
    'taken_signals': len(taken_signals),
    'skipped_signals': len(skipped_signals),
    'entry_rate_pct': hit_rate,
    'completed_trades': len(completed),
    'still_open': len(still_open),
    'win_rate': len([t for t in completed if t.pnl_pct >= 0]) / len(completed) * 100 if completed else 0,
    'total_pnl_port': sum(t.portfolio_pnl for t in completed),
    'skip_reasons': dict(skip_reasons),
    'exit_reasons': {
        'TP': len(tp_trades), 'SL': len(sl_trades),
        'TIMEOUT': len(timeout_trades), 'CLOSE_OLDEST': len(oldest_trades)
    },
}
with open('results/last_72h_signal_analysis.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to results/last_72h_signal_analysis.json")
