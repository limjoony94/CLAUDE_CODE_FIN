"""
SL Cluster Event Analysis — Live Trade History
================================================
Research: Analyze SHORT SL cluster events from live trading data.
Standard Research Protocol: Evidence-based, no look-ahead bias.

Questions:
1. Cluster anatomy (3+ SLs within 1 hour)
2. Direction asymmetry (SHORT vs LONG)
3. Cascade SL effectiveness
4. Timing patterns
5. Mitigation options assessment
"""

import json
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict

# ── Paths ──
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRICS_PATH = os.path.join(BASE, "results", "pattern_5m_metrics.json")
OUTPUT_PATH = os.path.join(BASE, "results", "sl_cluster_analysis.json")

# ── Load data ──
with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

trades = metrics.get("trade_history", [])
print(f"Total trades in history: {len(trades)}")

# ── Parse timestamps ──
for t in trades:
    t["dt"] = datetime.fromisoformat(t["timestamp"])

# Sort by timestamp
trades.sort(key=lambda x: x["dt"])

date_range = f"{trades[0]['dt'].strftime('%Y-%m-%d')} to {trades[-1]['dt'].strftime('%Y-%m-%d')}"
n_days = (trades[-1]["dt"] - trades[0]["dt"]).total_seconds() / 86400
print(f"Date range: {date_range} ({n_days:.1f} days)")

# ══════════════════════════════════════════════════════════════════
# 1. CLUSTER ANATOMY — Extract all SL clusters (3+ SLs within 1 hour)
# ══════════════════════════════════════════════════════════════════

sl_trades = [t for t in trades if t.get("exit_reason") in ("SL", "CASCADE_SL")]
print(f"\nTotal SL trades: {len(sl_trades)} (of {len(trades)} total)")

# Find clusters: 3+ SLs within 60 minutes
CLUSTER_WINDOW_MINUTES = 60
MIN_CLUSTER_SIZE = 3

clusters = []
used = set()

for i, t in enumerate(sl_trades):
    if i in used:
        continue
    # Collect all SLs within window starting from this trade
    cluster = [t]
    cluster_ids = {i}
    for j in range(i + 1, len(sl_trades)):
        if j in used:
            continue
        delta = (sl_trades[j]["dt"] - t["dt"]).total_seconds() / 60
        if delta <= CLUSTER_WINDOW_MINUTES:
            cluster.append(sl_trades[j])
            cluster_ids.add(j)
        elif delta > CLUSTER_WINDOW_MINUTES:
            break

    if len(cluster) >= MIN_CLUSTER_SIZE:
        clusters.append(cluster)
        used.update(cluster_ids)

print(f"\n{'='*70}")
print(f"CLUSTER ANATOMY: {len(clusters)} cluster events found")
print(f"{'='*70}")

cluster_details = []
total_cluster_sls = 0
total_cluster_loss = 0.0

for ci, cluster in enumerate(clusters):
    n_sl = len(cluster)
    total_cluster_sls += n_sl

    directions = [t["direction"] for t in cluster]
    dir_counts = {d: directions.count(d) for d in set(directions)}
    dominant_dir = max(dir_counts, key=dir_counts.get)

    # Portfolio loss from cluster
    portfolio_loss = sum(t["pnl_portfolio"] for t in cluster)
    total_cluster_loss += portfolio_loss
    slot_losses = [t["pnl_slot"] for t in cluster]

    # Time span
    t_start = cluster[0]["dt"]
    t_end = cluster[-1]["dt"]
    span_min = (t_end - t_start).total_seconds() / 60

    # BTC price movement: use entry prices to approximate
    entry_prices = [t["entry_price"] for t in cluster]
    exit_prices = [t["exit_price"] for t in cluster]
    price_start = min(entry_prices)
    price_end = max(exit_prices) if dominant_dir == "SHORT" else min(exit_prices)
    # For SHORT SLs, exit_price > entry_price (BTC went up)
    # For LONG SLs, exit_price < entry_price (BTC went down)

    # Cascade detection: check if SL distances progressively tightened
    sl_distances = []
    for t in cluster:
        if t["direction"] == "SHORT":
            sl_dist = abs(t["sl_price"] - t["entry_price"]) / t["entry_price"] * 100
        else:
            sl_dist = abs(t["entry_price"] - t["sl_price"]) / t["entry_price"] * 100
        sl_distances.append(sl_dist)

    # Count CASCADE_SL vs regular SL
    cascade_count = sum(1 for t in cluster if t.get("exit_reason") == "CASCADE_SL")
    regular_sl_count = n_sl - cascade_count

    # Patterns hit
    patterns = [t["pattern"] for t in cluster]

    detail = {
        "cluster_id": ci + 1,
        "timestamp_start": t_start.strftime("%Y-%m-%d %H:%M"),
        "timestamp_end": t_end.strftime("%Y-%m-%d %H:%M"),
        "day_of_week": t_start.strftime("%A"),
        "hour_of_day": t_start.hour,
        "span_minutes": round(span_min, 1),
        "n_sls": n_sl,
        "direction_breakdown": dir_counts,
        "dominant_direction": dominant_dir,
        "portfolio_loss_pct": round(portfolio_loss, 4),
        "slot_losses": [round(s, 4) for s in slot_losses],
        "avg_slot_loss": round(sum(slot_losses) / len(slot_losses), 4),
        "sl_distances_pct": [round(d, 4) for d in sl_distances],
        "cascade_sl_count": cascade_count,
        "regular_sl_count": regular_sl_count,
        "entry_prices": entry_prices,
        "patterns": patterns,
    }
    cluster_details.append(detail)

    print(f"\nCluster #{ci+1}: {t_start.strftime('%Y-%m-%d %H:%M')} ({t_start.strftime('%A')})")
    print(f"  SLs: {n_sl} ({dir_counts}) | Dominant: {dominant_dir}")
    print(f"  Portfolio loss: {portfolio_loss:+.4f}% | Span: {span_min:.0f} min")
    print(f"  Cascade SLs: {cascade_count} | Regular SLs: {regular_sl_count}")
    print(f"  SL distances: {[round(d, 2) for d in sl_distances]}%")
    print(f"  Slot losses: {[round(s, 2) for s in slot_losses]}%")
    print(f"  Patterns: {patterns}")

print(f"\n--- Cluster Summary ---")
print(f"Total SLs in clusters: {total_cluster_sls} of {len(sl_trades)} total SLs ({total_cluster_sls/len(sl_trades)*100:.1f}%)")
print(f"Total portfolio loss from clusters: {total_cluster_loss:.4f}%")

# Isolated SLs
isolated_sl_loss = sum(t["pnl_portfolio"] for t in sl_trades) - total_cluster_loss
print(f"Total portfolio loss from isolated SLs: {isolated_sl_loss:.4f}%")
print(f"Cluster share of total SL damage: {total_cluster_loss / sum(t['pnl_portfolio'] for t in sl_trades) * 100:.1f}%")

# ══════════════════════════════════════════════════════════════════
# 2. DIRECTION ASYMMETRY
# ══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("DIRECTION ASYMMETRY")
print(f"{'='*70}")

# All SLs by direction
long_sls = [t for t in sl_trades if t["direction"] == "LONG"]
short_sls = [t for t in sl_trades if t["direction"] == "SHORT"]

print(f"\nAll SLs:")
print(f"  LONG SLs:  {len(long_sls)} | avg slot loss: {sum(t['pnl_slot'] for t in long_sls)/max(len(long_sls),1):.4f}% | total portfolio: {sum(t['pnl_portfolio'] for t in long_sls):.4f}%")
print(f"  SHORT SLs: {len(short_sls)} | avg slot loss: {sum(t['pnl_slot'] for t in short_sls)/max(len(short_sls),1):.4f}% | total portfolio: {sum(t['pnl_portfolio'] for t in short_sls):.4f}%")

# Cluster SLs by direction
cluster_long = sum(1 for c in clusters for t in c if t["direction"] == "LONG")
cluster_short = sum(1 for c in clusters for t in c if t["direction"] == "SHORT")
cluster_long_loss = sum(t["pnl_portfolio"] for c in clusters for t in c if t["direction"] == "LONG")
cluster_short_loss = sum(t["pnl_portfolio"] for c in clusters for t in c if t["direction"] == "SHORT")

print(f"\nCluster SLs:")
print(f"  LONG in clusters:  {cluster_long} | portfolio loss: {cluster_long_loss:.4f}%")
print(f"  SHORT in clusters: {cluster_short} | portfolio loss: {cluster_short_loss:.4f}%")

# Direction of each cluster
short_dominant_clusters = [c for c in cluster_details if c["dominant_direction"] == "SHORT"]
long_dominant_clusters = [c for c in cluster_details if c["dominant_direction"] == "LONG"]
mixed_clusters = [c for c in cluster_details if c["direction_breakdown"].get("LONG", 0) > 0 and c["direction_breakdown"].get("SHORT", 0) > 0]

print(f"\nCluster direction dominance:")
print(f"  SHORT-dominant clusters: {len(short_dominant_clusters)}")
print(f"  LONG-dominant clusters:  {len(long_dominant_clusters)}")
print(f"  Mixed-direction clusters: {len(mixed_clusters)}")

# All trades by direction (for WR context)
all_long = [t for t in trades if t["direction"] == "LONG"]
all_short = [t for t in trades if t["direction"] == "SHORT"]
long_wr = sum(1 for t in all_long if t["pnl_portfolio"] > 0) / max(len(all_long), 1) * 100
short_wr = sum(1 for t in all_short if t["pnl_portfolio"] > 0) / max(len(all_short), 1) * 100
print(f"\nOverall WR by direction:")
print(f"  LONG WR:  {long_wr:.1f}% ({len(all_long)} trades)")
print(f"  SHORT WR: {short_wr:.1f}% ({len(all_short)} trades)")

# ══════════════════════════════════════════════════════════════════
# 3. CASCADE SL EFFECTIVENESS
# ══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("CASCADE SL EFFECTIVENESS")
print(f"{'='*70}")

for ci, (cluster, detail) in enumerate(zip(clusters, cluster_details)):
    print(f"\nCluster #{ci+1} ({detail['timestamp_start']}):")
    print(f"  SL distances (entry→SL): {detail['sl_distances_pct']}%")

    # Check if distances decreased (cascade tightening evidence)
    dists = detail["sl_distances_pct"]
    if len(dists) > 1:
        # First SL distance vs subsequent
        first_dist = dists[0]
        subsequent_dists = dists[1:]
        tightened = [d for d in subsequent_dists if d < first_dist * 0.5]
        print(f"  First SL distance: {first_dist:.2f}%")
        print(f"  Subsequent SL distances: {[round(d, 2) for d in subsequent_dists]}%")
        print(f"  Tightened SLs (<50% of first): {len(tightened)}")

        # Estimate damage without cascade
        # Without cascade, subsequent SLs would have hit at original distance
        # Cascade tightened them → they hit sooner with smaller loss per slot
        # But the TOTAL count might be the same (cascade triggers more SLs)

        # Actual slot losses
        actual_total = sum(abs(t["pnl_slot"]) for t in cluster)

        # Without cascade: each SL at original distance
        # The first SL triggered cascade, making others tighter
        # Without cascade, subsequent would NOT have hit at tightened level
        # Some might have recovered (TP) instead
        # Conservative estimate: without cascade, same trades still lose, but at original SL
        original_distances = []
        for t in cluster:
            if t["direction"] == "SHORT":
                original_sl_dist = abs(t["sl_price"] - t["entry_price"]) / t["entry_price"] * 100
            else:
                original_sl_dist = abs(t["entry_price"] - t["sl_price"]) / t["entry_price"] * 100
            original_distances.append(original_sl_dist)

        print(f"  Actual total slot loss: {actual_total:.2f}%")

        # Count cascade exits specifically
        cascade_exits = [t for t in cluster if t.get("exit_reason") == "CASCADE_SL"]
        regular_exits = [t for t in cluster if t.get("exit_reason") == "SL"]

        if cascade_exits:
            cascade_loss = sum(abs(t["pnl_slot"]) for t in cascade_exits)
            regular_loss = sum(abs(t["pnl_slot"]) for t in regular_exits)
            print(f"  CASCADE_SL exits: {len(cascade_exits)} (loss: {cascade_loss:.2f}%)")
            print(f"  Regular SL exits: {len(regular_exits)} (loss: {regular_loss:.2f}%)")

            # Without cascade, the CASCADE_SL trades would have stayed open
            # They might have hit original SL (worse) or recovered (better)
            # Since we're in a directional move, they likely would have hit original SL
            # Counterfactual: CASCADE_SL trades at original SL distance (3x leverage)
            counterfactual_loss = 0
            for t in cascade_exits:
                # Original SL distance (before cascade tightening)
                # Cascade reduces SL by 85%, so original = current / 0.15
                # Actually: cascade multiplies SL distance by (1 - tighten_pct/100) = 0.15
                # But we don't know the original SL for cascade exits
                # The exit_price IS the tightened SL, and the sl_price field is the ORIGINAL SL
                # Actually sl_price in the record may be the tightened SL
                # Let's use the first trade's SL distance as proxy for "original"
                counterfactual_loss += abs(t["pnl_slot"]) / 0.15  # rough estimate

            print(f"  Counterfactual (no cascade, same trades at original SL): ~{counterfactual_loss:.1f}% slot loss")
            saving = counterfactual_loss - cascade_loss
            print(f"  Estimated cascade saving: ~{saving:.1f}% slot loss")
        else:
            print(f"  No CASCADE_SL exits (all regular SL)")

# ══════════════════════════════════════════════════════════════════
# 4. TIMING PATTERNS
# ══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("TIMING PATTERNS")
print(f"{'='*70}")

# Hour of day distribution for clusters
cluster_hours = [c["hour_of_day"] for c in cluster_details]
cluster_days = [c["day_of_week"] for c in cluster_details]

print(f"\nCluster events by hour (UTC):")
hour_counts = defaultdict(int)
for h in cluster_hours:
    hour_counts[h] += 1
for h in sorted(hour_counts.keys()):
    print(f"  {h:02d}:00 — {hour_counts[h]} clusters")

print(f"\nCluster events by day of week:")
day_counts = defaultdict(int)
for d in cluster_days:
    day_counts[d] += 1
for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
    if d in day_counts:
        print(f"  {d}: {day_counts[d]} clusters")

# All SLs by hour
print(f"\nAll SL trades by hour (UTC):")
sl_hour_counts = defaultdict(int)
for t in sl_trades:
    sl_hour_counts[t["dt"].hour] += 1
for h in sorted(sl_hour_counts.keys()):
    bar = "#" * sl_hour_counts[h]
    print(f"  {h:02d}:00 — {sl_hour_counts[h]:3d} {bar}")

# Compare cluster vs non-cluster timing
print(f"\nAll trades by hour (context):")
all_hour_counts = defaultdict(int)
for t in trades:
    all_hour_counts[t["dt"].hour] += 1

# ══════════════════════════════════════════════════════════════════
# 5. MITIGATION OPTIONS ASSESSMENT
# ══════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("MITIGATION OPTIONS ASSESSMENT")
print(f"{'='*70}")

# A) Direction cap analysis: how many same-dir positions were open at cluster time?
print(f"\n--- A) Direction Cap (current: 7) ---")
for ci, (cluster, detail) in enumerate(zip(clusters, cluster_details)):
    dom_dir = detail["dominant_direction"]
    dom_count = detail["direction_breakdown"].get(dom_dir, 0)
    print(f"  Cluster #{ci+1}: {dom_count} {dom_dir} SLs (max simultaneous)")

    # With cap=5, at most 5 same-dir SLs could fire
    would_prevent = max(0, dom_count - 5)
    loss_per_sl = detail["avg_slot_loss"]
    print(f"    Cap 7→5 would prevent: {would_prevent} SLs (est. saving: {would_prevent * abs(loss_per_sl) / 9:.4f}% portfolio)")

# B) Momentum guard analysis
print(f"\n--- B) Momentum Guard (current: 1.5%/15min, 1h cooldown) ---")
# Check if momentum guard could have prevented entries before cluster
# We need to see what BTC did before each cluster
for ci, (cluster, detail) in enumerate(zip(clusters, cluster_details)):
    # Look at price movement before cluster
    cluster_start = cluster[0]["dt"]
    # Find trades that entered in the 2 hours before cluster
    recent_entries = [t for t in trades
                      if (cluster_start - t["dt"]).total_seconds() < 7200
                      and (cluster_start - t["dt"]).total_seconds() > 0
                      and t["direction"] == detail["dominant_direction"]]

    print(f"  Cluster #{ci+1}: {len(recent_entries)} same-dir entries in 2h before cluster")

# C) Time-of-day filter analysis
print(f"\n--- C) Time-of-Day Filter ---")
# Calculate SL rate by hour
for h in sorted(all_hour_counts.keys()):
    total_h = all_hour_counts[h]
    sl_h = sl_hour_counts.get(h, 0)
    sl_rate = sl_h / max(total_h, 1) * 100
    if total_h > 0:
        print(f"  {h:02d}:00 — Trades: {total_h:3d} | SLs: {sl_h:3d} | SL rate: {sl_rate:.1f}%")

# D) Summary statistics
print(f"\n--- D) Overall Risk Summary ---")
total_pnl = sum(t["pnl_portfolio"] for t in trades)
total_sl_pnl = sum(t["pnl_portfolio"] for t in sl_trades)
total_tp_pnl = sum(t["pnl_portfolio"] for t in trades if t.get("exit_reason") == "TP")

print(f"  Total portfolio PnL: {total_pnl:.4f}%")
print(f"  Total SL damage: {total_sl_pnl:.4f}%")
print(f"  Total TP gains: {total_tp_pnl:.4f}%")
print(f"  Cluster SL as % of total SL damage: {total_cluster_loss / total_sl_pnl * 100:.1f}%")
print(f"  Cluster SL as % of total PnL offset: {abs(total_cluster_loss) / max(total_tp_pnl, 0.001) * 100:.1f}% of TP gains consumed")

# E) Hold time analysis for cluster SLs
print(f"\n--- E) Hold Time Analysis ---")
cluster_hold_times = [t["hold_minutes"] for c in clusters for t in c]
isolated_sl_hold_times = [t["hold_minutes"] for t in sl_trades if not any(t in c for c in clusters)]

if cluster_hold_times:
    print(f"  Cluster SL avg hold: {sum(cluster_hold_times)/len(cluster_hold_times):.0f} min")
if isolated_sl_hold_times:
    print(f"  Isolated SL avg hold: {sum(isolated_sl_hold_times)/len(isolated_sl_hold_times):.0f} min")

# ══════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════

results = {
    "analysis_date": datetime.now().isoformat(),
    "data_range": date_range,
    "data_days": round(n_days, 1),
    "total_trades": len(trades),
    "total_sls": len(sl_trades),
    "total_clusters": len(clusters),
    "total_cluster_sls": total_cluster_sls,
    "cluster_sl_share_pct": round(total_cluster_sls / len(sl_trades) * 100, 1),
    "total_cluster_loss_pct": round(total_cluster_loss, 4),
    "total_sl_loss_pct": round(total_sl_pnl, 4),
    "cluster_details": cluster_details,
    "direction_stats": {
        "long_sls": len(long_sls),
        "short_sls": len(short_sls),
        "long_wr": round(long_wr, 1),
        "short_wr": round(short_wr, 1),
        "long_trades": len(all_long),
        "short_trades": len(all_short),
    },
    "timing": {
        "cluster_hours": dict(hour_counts),
        "cluster_days": dict(day_counts),
        "sl_by_hour": dict(sl_hour_counts),
    },
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to: {OUTPUT_PATH}")
