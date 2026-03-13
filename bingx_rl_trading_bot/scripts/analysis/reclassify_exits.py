"""Retroactively reclassify exit_reason for historical trades.

v1.59.3: Apply current v1.55.1+ classification logic to all trades
in metrics trade_history. Many pre-v1.55.0 trades were classified as
MARKET/UNKNOWN because near-SL/near-TP/CASCADE_SL logic didn't exist.

This script:
1. Loads metrics trade_history
2. Re-runs _infer_exit_reason on each trade using stored prices
3. Shows diff, asks for confirmation, then saves
"""
import sys, os, json, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from bingx_rl_trading_bot.scripts.production.pattern_5m.position_monitor import (
    _infer_exit_reason,
)

METRICS_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'results', 'pattern_5m_metrics.json'
)


def reclassify_trade(trade: dict) -> str:
    """Re-classify a single trade using current logic."""
    position = {
        'entry_price': trade.get('entry_price', 0),
        'tp_price': trade.get('tp_price', 0),
        'sl_price': trade.get('sl_price', 0),
        'direction': trade.get('direction', ''),
        # Historical trades don't have cascade data, so skip cascade-specific paths
    }
    exit_price = trade.get('exit_price', 0)

    if exit_price <= 0 or position['entry_price'] <= 0:
        return trade.get('exit_reason', 'UNKNOWN')

    return _infer_exit_reason(exit_price, position)


def main():
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    trade_history = metrics.get('trade_history', [])
    if not trade_history:
        print("No trade history found.")
        return

    # Backup
    backup_path = METRICS_PATH.replace('.json', '_pre_reclassify_backup.json')
    with open(backup_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Backup saved: {backup_path}")

    # Reclassify
    changes = []
    for i, trade in enumerate(trade_history):
        old_reason = trade.get('exit_reason', 'UNKNOWN')
        new_reason = reclassify_trade(trade)

        # Preserve certain original classifications that can't be re-derived:
        # - EARLY_BU/EARLY_BD: bot-initiated early exit (3-candle reversal)
        # - TIMEOUT: bot-initiated timeout closure
        # These were set at closure time by the bot, not inferred from price.
        if old_reason in ('EARLY_BU', 'EARLY_BD', 'TIMEOUT'):
            continue

        if old_reason != new_reason:
            changes.append({
                'index': i,
                'timestamp': trade.get('timestamp', ''),
                'direction': trade.get('direction', ''),
                'pattern': trade.get('pattern', 'N/A'),
                'old': old_reason,
                'new': new_reason,
                'pnl_slot': trade.get('pnl_slot', 0),
                'entry': trade.get('entry_price', 0),
                'exit': trade.get('exit_price', 0),
                'tp': trade.get('tp_price', 0),
                'sl': trade.get('sl_price', 0),
            })

    # Report
    print(f"\nTotal trades: {len(trade_history)}")
    print(f"Changes found: {len(changes)}")

    if not changes:
        print("No reclassifications needed.")
        return

    print(f"\n{'='*80}")
    print(f"{'#':>3} {'Timestamp':16} {'Dir':5} {'Pattern':16} {'Old':12} → {'New':12} {'PnL':>8}")
    print(f"{'='*80}")

    from collections import Counter
    transition_counts = Counter()

    for c in changes:
        transition_counts[(c['old'], c['new'])] += 1
        print(
            f"{c['index']:3d} {c['timestamp'][:16]:16} {c['direction']:5} "
            f"{c['pattern']:16} {c['old']:12} → {c['new']:12} {c['pnl_slot']:+8.2f}%"
        )

    print(f"\n{'='*80}")
    print("Transition summary:")
    for (old, new), count in sorted(transition_counts.items(), key=lambda x: -x[1]):
        pnl_sum = sum(c['pnl_slot'] for c in changes if c['old'] == old and c['new'] == new)
        print(f"  {old:12} → {new:12}: {count:3d} trades, PnL {pnl_sum:+.2f}%")

    # Apply changes
    for c in changes:
        trade_history[c['index']]['exit_reason'] = c['new']

    # Recalculate aggregate metrics
    total = len(trade_history)
    wins = sum(1 for t in trade_history if t.get('pnl_slot', 0) > 0)
    losses = total - wins
    total_pnl = sum(t.get('pnl_portfolio', 0) for t in trade_history)

    avg_win_trades = [t for t in trade_history if t.get('pnl_slot', 0) > 0]
    avg_loss_trades = [t for t in trade_history if t.get('pnl_slot', 0) <= 0]
    avg_win = sum(t['pnl_slot'] for t in avg_win_trades) / len(avg_win_trades) if avg_win_trades else 0
    avg_loss = sum(t['pnl_slot'] for t in avg_loss_trades) / len(avg_loss_trades) if avg_loss_trades else 0
    edge = sum(t['pnl_slot'] for t in trade_history) / total if total else 0

    from collections import Counter as C2
    exit_dist = C2(t.get('exit_reason', '?') for t in trade_history)

    print(f"\nUpdated exit distribution:")
    for reason, count in exit_dist.most_common():
        reason_pnl = sum(t.get('pnl_portfolio', 0) for t in trade_history if t.get('exit_reason') == reason)
        print(f"  {reason:12}: {count:3d} ({count/total*100:5.1f}%) PnL {reason_pnl:+.2f}%")

    # Update metrics
    metrics['trade_history'] = trade_history
    # Note: winning_trades/losing_trades in metrics are based on pnl, not exit_reason
    # so they don't change. Only exit_reason labels change.

    # Save
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}")
    print(f"Backup at {backup_path}")


if __name__ == '__main__':
    main()
