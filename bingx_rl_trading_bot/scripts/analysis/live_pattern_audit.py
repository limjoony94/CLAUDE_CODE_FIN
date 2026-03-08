"""Live pattern performance audit — identify losing patterns for removal."""
import json
import os
from collections import defaultdict, Counter

BASE = r"C:\Users\J\OneDrive\CLAUDE_CODE_FIN\bingx_rl_trading_bot"
METRICS = os.path.join(BASE, "results", "pattern_5m_metrics.json")
PATTERNS = os.path.join(BASE, "results", "dynamic_patterns.json")

with open(METRICS) as f:
    m = json.load(f)
with open(PATTERNS) as f:
    dp = json.load(f)

history = m.get('trade_history', [])
validated = dp.get('patterns', {})

print(f"Total live trades: {len(history)}")
print(f"Overall WR: {sum(1 for t in history if (t.get('pnl_slot',0) or 0) > 0)/len(history)*100:.1f}%")
print(f"Overall PnL: {sum(t.get('pnl_slot',0) or 0 for t in history):+.2f}%")

# ============================================================
# 1. PER-PATTERN BREAKDOWN
# ============================================================
print("\n" + "="*80)
print("1. PER-PATTERN LIVE PERFORMANCE (sorted by PnL)")
print("="*80)

pat_stats = defaultdict(lambda: {'n':0, 'w':0, 'pnl':0.0, 'losses': [], 'wins': [], 'dirs': Counter()})
for t in history:
    p = t.get('pattern') or 'N/A'
    d = t.get('direction', 'UNK')
    pnl = t.get('pnl_slot', 0) or 0
    pat_stats[p]['n'] += 1
    pat_stats[p]['pnl'] += pnl
    pat_stats[p]['dirs'][d] += 1
    if pnl > 0:
        pat_stats[p]['w'] += 1
        pat_stats[p]['wins'].append(pnl)
    else:
        pat_stats[p]['losses'].append(pnl)

print(f"\n{'Pattern':20s} {'N':>4s} {'WR':>6s} {'PnL':>8s} {'AvgW':>7s} {'AvgL':>7s} {'Dir':>12s} {'InDP':>5s}")
print("-"*80)
for p, s in sorted(pat_stats.items(), key=lambda x: x[1]['pnl']):
    wr = s['w']/s['n']*100 if s['n'] else 0
    avg_w = sum(s['wins'])/len(s['wins']) if s['wins'] else 0
    avg_l = sum(s['losses'])/len(s['losses']) if s['losses'] else 0
    dirs = '+'.join(f"{d[0]}{c}" for d,c in s['dirs'].items())
    in_dp = 'YES' if p in validated else 'NO'
    print(f"{p:20s} {s['n']:4d} {wr:5.1f}% {s['pnl']:+7.2f}% {avg_w:+6.2f}% {avg_l:+6.2f}% {dirs:>12s} {in_dp:>5s}")

# ============================================================
# 2. LOSING PATTERNS ANALYSIS
# ============================================================
print("\n" + "="*80)
print("2. LOSING PATTERNS (candidates for removal)")
print("="*80)

losers = [(p, s) for p, s in pat_stats.items() if s['pnl'] < 0 and s['n'] >= 2]
winners = [(p, s) for p, s in pat_stats.items() if s['pnl'] >= 0]
loser_pnl = sum(s['pnl'] for _, s in losers)
winner_pnl = sum(s['pnl'] for _, s in winners)

print(f"\nLosing patterns (>=2 trades): {len(losers)} patterns, PnL {loser_pnl:+.2f}%")
print(f"Winning patterns: {len(winners)} patterns, PnL {winner_pnl:+.2f}%")

# Compare with backtest edge
print(f"\n{'Pattern':20s} {'LiveN':>5s} {'LiveWR':>7s} {'LivePnL':>8s} {'BT_WR':>7s} {'BT_Edge':>8s} {'Gap':>6s}")
print("-"*80)
for p, s in sorted(losers, key=lambda x: x[1]['pnl']):
    wr_live = s['w']/s['n']*100 if s['n'] else 0
    # Check backtest stats
    bt = validated.get(p, {})
    bt_wr = bt.get('wr', 0)
    bt_edge = bt.get('edge_pp', 0)
    gap = wr_live - bt_wr if bt_wr else 0
    print(f"{p:20s} {s['n']:5d} {wr_live:6.1f}% {s['pnl']:+7.2f}% {bt_wr:6.1f}% {bt_edge:+7.1f}pp {gap:+5.1f}pp")

# ============================================================
# 3. DIRECTION ANALYSIS
# ============================================================
print("\n" + "="*80)
print("3. DIRECTION ANALYSIS")
print("="*80)

for d in ['LONG', 'SHORT']:
    dt = [t for t in history if t.get('direction') == d]
    if not dt:
        continue
    wr = sum(1 for t in dt if (t.get('pnl_slot',0) or 0) > 0)/len(dt)*100
    pnl = sum(t.get('pnl_slot',0) or 0 for t in dt)
    avg_w = sum(t.get('pnl_slot',0) or 0 for t in dt if (t.get('pnl_slot',0) or 0) > 0)
    avg_l = sum(t.get('pnl_slot',0) or 0 for t in dt if (t.get('pnl_slot',0) or 0) <= 0)
    n_w = sum(1 for t in dt if (t.get('pnl_slot',0) or 0) > 0)
    n_l = len(dt) - n_w
    print(f"\n{d}: {len(dt)}t, WR {wr:.1f}%, PnL {pnl:+.2f}%")
    print(f"  Wins:   {n_w}t, total {avg_w:+.2f}%")
    print(f"  Losses: {n_l}t, total {avg_l:+.2f}%")
    if n_w > 0:
        print(f"  Avg win:  {avg_w/n_w:+.3f}%")
    if n_l > 0:
        print(f"  Avg loss: {avg_l/n_l:+.3f}%")

# ============================================================
# 4. EXIT REASON ANALYSIS
# ============================================================
print("\n" + "="*80)
print("4. EXIT REASON BREAKDOWN")
print("="*80)

exits = defaultdict(lambda: {'n':0, 'pnl':0.0})
for t in history:
    r = t.get('exit_reason', 'UNK')
    pnl = t.get('pnl_slot', 0) or 0
    exits[r]['n'] += 1
    exits[r]['pnl'] += pnl

print(f"\n{'Reason':15s} {'N':>5s} {'PnL':>9s} {'Avg':>8s}")
print("-"*40)
for r, s in sorted(exits.items(), key=lambda x: x[1]['pnl']):
    avg = s['pnl']/s['n'] if s['n'] else 0
    print(f"{r:15s} {s['n']:5d} {s['pnl']:+8.2f}% {avg:+7.3f}%")

# ============================================================
# 5. RECENT vs EARLY PERFORMANCE
# ============================================================
print("\n" + "="*80)
print("5. PERFORMANCE BY DATE PERIOD")
print("="*80)

daily = defaultdict(lambda: {'n':0, 'pnl':0.0, 'w':0})
for t in history:
    day = str(t.get('timestamp', ''))[:10]
    pnl = t.get('pnl_slot', 0) or 0
    daily[day]['n'] += 1
    daily[day]['pnl'] += pnl
    if pnl > 0:
        daily[day]['w'] += 1

print(f"\n{'Date':12s} {'N':>4s} {'WR':>6s} {'PnL':>8s} {'CumPnL':>9s}")
print("-"*45)
cum = 0.0
for day in sorted(daily.keys()):
    s = daily[day]
    wr = s['w']/s['n']*100 if s['n'] else 0
    cum += s['pnl']
    print(f"{day:12s} {s['n']:4d} {wr:5.1f}% {s['pnl']:+7.2f}% {cum:+8.2f}%")

# ============================================================
# 6. ACTIONABLE SUMMARY
# ============================================================
print("\n" + "="*80)
print("6. ACTIONABLE SUMMARY")
print("="*80)

# Patterns with >=3 trades and WR < 40%
bad_pats = [(p,s) for p,s in pat_stats.items()
            if s['n'] >= 3 and s['w']/s['n']*100 < 40 and p != 'N/A']
if bad_pats:
    print(f"\n⚠️ Patterns with >=3 trades and WR < 40% (remove candidates):")
    for p, s in sorted(bad_pats, key=lambda x: x[1]['pnl']):
        wr = s['w']/s['n']*100
        bt = validated.get(p, {})
        bt_wr = bt.get('wr', 0)
        print(f"  {p}: {s['n']}t, WR {wr:.0f}%, PnL {s['pnl']:+.2f}% (backtest WR {bt_wr:.0f}%)")

# Patterns never traded
traded_pats = set(pat_stats.keys())
never_traded = set(validated.keys()) - traded_pats
print(f"\nPatterns never traded yet: {len(never_traded)}/{len(validated)}")

# Overall recommendation
total_pnl = sum(t.get('pnl_slot',0) or 0 for t in history)
if bad_pats:
    bad_pnl = sum(s['pnl'] for _,s in bad_pats)
    print(f"\n💡 Removing {len(bad_pats)} bad patterns would improve PnL by {abs(bad_pnl):.2f}%")
    print(f"   Current: {total_pnl:+.2f}% → Projected: {total_pnl-bad_pnl:+.2f}%")
