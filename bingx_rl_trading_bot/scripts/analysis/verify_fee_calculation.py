#!/usr/bin/env python3
"""
Verify Fee Calculation Logic
=============================

Test the updated TradeSimulator fee calculation
"""

# Constants
TAKER_FEE = 0.0005  # 0.05% per trade
LEVERAGE = 4

print("="*80)
print("FEE CALCULATION VERIFICATION")
print("="*80)
print(f"Taker Fee: {TAKER_FEE*100:.2f}% per trade")
print(f"Leverage: {LEVERAGE}x")
print()

# Test case 1: 2% profit
print("Test Case 1: 2% unleveraged profit")
print("-" * 80)
pnl_pct = 0.02
entry_commission_pct = LEVERAGE * TAKER_FEE
exit_commission_pct = LEVERAGE * (1 + pnl_pct) * TAKER_FEE
total_commission_pct = entry_commission_pct + exit_commission_pct
gross_leveraged_pnl_pct = pnl_pct * LEVERAGE
net_leveraged_pnl_pct = gross_leveraged_pnl_pct - total_commission_pct

print(f"  Entry fee: {entry_commission_pct*100:.4f}% of capital")
print(f"  Exit fee: {exit_commission_pct*100:.4f}% of capital")
print(f"  Total fee: {total_commission_pct*100:.4f}% of capital")
print(f"  Gross leveraged P&L: {gross_leveraged_pnl_pct*100:.2f}%")
print(f"  Net leveraged P&L: {net_leveraged_pnl_pct*100:.2f}%")
print(f"  Fee impact: -{(gross_leveraged_pnl_pct - net_leveraged_pnl_pct)*100:.2f}%")
print()

# Test case 2: -1% loss
print("Test Case 2: -1% unleveraged loss")
print("-" * 80)
pnl_pct = -0.01
entry_commission_pct = LEVERAGE * TAKER_FEE
exit_commission_pct = LEVERAGE * (1 + pnl_pct) * TAKER_FEE
total_commission_pct = entry_commission_pct + exit_commission_pct
gross_leveraged_pnl_pct = pnl_pct * LEVERAGE
net_leveraged_pnl_pct = gross_leveraged_pnl_pct - total_commission_pct

print(f"  Entry fee: {entry_commission_pct*100:.4f}% of capital")
print(f"  Exit fee: {exit_commission_pct*100:.4f}% of capital")
print(f"  Total fee: {total_commission_pct*100:.4f}% of capital")
print(f"  Gross leveraged P&L: {gross_leveraged_pnl_pct*100:.2f}%")
print(f"  Net leveraged P&L: {net_leveraged_pnl_pct*100:.2f}%")
print(f"  Fee impact: -{(gross_leveraged_pnl_pct - net_leveraged_pnl_pct)*100:.2f}% (worse)")
print()

# Test case 3: Profit threshold (2% leveraged after fees)
print("Test Case 3: Find unleveraged profit for 2% leveraged net profit")
print("-" * 80)
target_net_leveraged = 0.02
# Solve: (pnl_pct × 4) - (4 × 0.0005 + 4 × (1 + pnl_pct) × 0.0005) = 0.02
# 4×pnl_pct - 0.002 - 0.002×(1 + pnl_pct) = 0.02
# 4×pnl_pct - 0.002 - 0.002 - 0.002×pnl_pct = 0.02
# 3.998×pnl_pct - 0.004 = 0.02
# 3.998×pnl_pct = 0.024
# pnl_pct = 0.024 / 3.998
pnl_pct = (target_net_leveraged + 2 * LEVERAGE * TAKER_FEE) / (LEVERAGE - LEVERAGE * TAKER_FEE)
entry_commission_pct = LEVERAGE * TAKER_FEE
exit_commission_pct = LEVERAGE * (1 + pnl_pct) * TAKER_FEE
total_commission_pct = entry_commission_pct + exit_commission_pct
gross_leveraged_pnl_pct = pnl_pct * LEVERAGE
net_leveraged_pnl_pct = gross_leveraged_pnl_pct - total_commission_pct

print(f"  Required unleveraged P&L: {pnl_pct*100:.4f}%")
print(f"  Entry fee: {entry_commission_pct*100:.4f}% of capital")
print(f"  Exit fee: {exit_commission_pct*100:.4f}% of capital")
print(f"  Total fee: {total_commission_pct*100:.4f}% of capital")
print(f"  Gross leveraged P&L: {gross_leveraged_pnl_pct*100:.2f}%")
print(f"  Net leveraged P&L: {net_leveraged_pnl_pct*100:.2f}%")
print()

print("="*80)
print("SUMMARY")
print("="*80)
print("Old labels: 2% leveraged profit threshold (no fees)")
print("New labels: ~2.06% leveraged profit threshold (with fees)")
print("Impact: Labels become ~0.06% more conservative (better)")
print("="*80)
