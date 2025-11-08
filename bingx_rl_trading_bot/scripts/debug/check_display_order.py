#!/usr/bin/env python3
"""Check display order and calculations"""

import json

with open('results/opportunity_gating_bot_4x_state.json', 'r') as f:
    state = json.load(f)

trades = state['trades']
closed = [t for t in trades if t.get('status') == 'CLOSED']

print('Trade Order (as they appear in reversed recent 5):')
print('='*80)

recent = closed[-5:]
for i, t in enumerate(reversed(recent), 1):
    side = t.get('side', 'N/A')
    entry = t.get('entry_price', 0)
    exit_p = t.get('exit_price', 0)
    pnl_net = t.get('pnl_usd_net', 0)
    pos_val = t.get('position_value', 0)
    qty = t.get('quantity', 0)

    if pos_val > 0:
        margin = pos_val
    else:
        margin = (qty * entry) / 4

    pnl_pct = (pnl_net / margin * 100) if margin > 0 else 0

    print(f'#{i} {side:>6s}: Entry ${entry:>10,.2f} → Exit ${exit_p:>10,.2f} | {pnl_pct:>+7.2f}% (${pnl_net:>+9.2f}) | Margin: ${margin:,.2f}')
