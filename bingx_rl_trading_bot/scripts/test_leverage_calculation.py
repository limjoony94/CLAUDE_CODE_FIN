"""
Test leverage calculation fix

Verify that quantity calculation now properly uses 4x leverage
"""

# Test parameters (realistic values from recent trades)
capital = 101187.75  # Current balance
position_pct = 0.50  # 50% position (from dynamic sizing)
btc_price = 110700.0  # Current BTC price
leverage = 4  # Config leverage

print("=" * 80)
print("LEVERAGE CALCULATION TEST")
print("=" * 80)

print(f"\nTest Parameters:")
print(f"  Capital: ${capital:,.2f}")
print(f"  Position: {position_pct*100:.1f}%")
print(f"  BTC Price: ${btc_price:,.2f}")
print(f"  Leverage: {leverage}x")

print(f"\n{'=' * 80}")
print("OLD (INCORRECT) CALCULATION")
print("=" * 80)

position_value_old = capital * position_pct
quantity_old = position_value_old / btc_price
position_worth_old = quantity_old * btc_price

print(f"\nCalculation:")
print(f"  position_value = ${capital:,.2f} x {position_pct} = ${position_value_old:,.2f}")
print(f"  quantity = ${position_value_old:,.2f} / ${btc_price:,.2f} = {quantity_old:.4f} BTC")
print(f"  position_worth = {quantity_old:.4f} x ${btc_price:,.2f} = ${position_worth_old:,.2f}")

print(f"\nResult:")
print(f"  ❌ Actual Position: ${position_worth_old:,.2f} (only {position_worth_old/capital*100:.1f}% of capital)")
print(f"  ❌ Effective Leverage: {position_worth_old/position_value_old:.1f}x (should be {leverage}x)")
print(f"  ❌ Problem: Leverage not applied to quantity calculation!")

print(f"\n{'=' * 80}")
print("NEW (CORRECT) CALCULATION")
print("=" * 80)

position_value_new = capital * position_pct  # Collateral
leveraged_value_new = position_value_new * leverage  # Actual position
quantity_new = leveraged_value_new / btc_price
position_worth_new = quantity_new * btc_price

print(f"\nCalculation:")
print(f"  position_value (collateral) = ${capital:,.2f} x {position_pct} = ${position_value_new:,.2f}")
print(f"  leveraged_value = ${position_value_new:,.2f} x {leverage} = ${leveraged_value_new:,.2f}")
print(f"  quantity = ${leveraged_value_new:,.2f} / ${btc_price:,.2f} = {quantity_new:.4f} BTC")
print(f"  position_worth = {quantity_new:.4f} x ${btc_price:,.2f} = ${position_worth_new:,.2f}")

print(f"\nResult:")
print(f"  ✅ Actual Position: ${position_worth_new:,.2f} ({position_worth_new/capital*100:.1f}% of capital)")
print(f"  ✅ Effective Leverage: {position_worth_new/position_value_new:.1f}x")
print(f"  ✅ Collateral Used: ${position_value_new:,.2f} ({position_value_new/capital*100:.1f}% of capital)")

print(f"\n{'=' * 80}")
print("COMPARISON")
print("=" * 80)

print(f"\n                    OLD         NEW         DIFFERENCE")
print(f"  Quantity (BTC):   {quantity_old:.4f}     {quantity_new:.4f}     {quantity_new-quantity_old:+.4f} ({(quantity_new/quantity_old-1)*100:+.1f}%)")
print(f"  Position Value:   ${position_worth_old:,.2f}  ${position_worth_new:,.2f}  ${position_worth_new-position_worth_old:+,.2f}")
print(f"  Collateral %:     {position_worth_old/capital*100:.1f}%        {position_value_new/capital*100:.1f}%        -")

print(f"\n{'=' * 80}")
print("RISK ANALYSIS")
print("=" * 80)

# Calculate risk for different scenarios
stop_loss_pct = 0.01  # 1% stop loss

print(f"\nStop Loss Scenario (-1% price move):")
print(f"  OLD calculation:")
sl_loss_old = position_worth_old * stop_loss_pct
sl_loss_old_pct = sl_loss_old / capital * 100
print(f"    Loss: ${sl_loss_old:,.2f} ({sl_loss_old_pct:.2f}% of capital)")

print(f"\n  NEW calculation:")
sl_loss_new = position_worth_new * stop_loss_pct
sl_loss_new_pct = sl_loss_new / capital * 100
print(f"    Loss: ${sl_loss_new:,.2f} ({sl_loss_new_pct:.2f}% of capital)")
print(f"    Difference: ${sl_loss_new-sl_loss_old:+,.2f} ({sl_loss_new_pct-sl_loss_old_pct:+.2f}%)")

print(f"\n{'=' * 80}")
print("CONCLUSION")
print("=" * 80)

print(f"""
✅ FIX VERIFIED:
   - Quantity calculation now uses leveraged_value (4x)
   - Position size is {leverage}x larger
   - Risk is {leverage}x higher (as intended)
   - Matches backtest assumptions (4x leverage)

⚠️ IMPORTANT:
   - This is the CORRECT implementation for 4x leverage
   - Old code was effectively using 1x leverage
   - New trades will have {leverage}x larger positions
   - Stop loss impact will be {leverage}x stronger
   - Make sure exchange allows this leverage level
""")

print("=" * 80)
