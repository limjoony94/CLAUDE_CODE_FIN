"""
C1 Breakout v2 — Signal Generator
====================================
Channel breakout entry + Fractal SL + ATR Trailing TP.

Entry: 15-bar channel breakout + body > 40% of range
SL: Fractal swing point capped at 3.3×ATR from entry
TP: Trailing from best price at 2.5×ATR

All parameters from config, dynamic (ATR-based), no look-ahead.
"""

import math
from .indicators import compute_atr, compute_channel, compute_fractal_swings


class C1BreakoutSignal:
    def __init__(self, config: dict):
        self.channel_period = config.get('channel_period', 15)
        self.body_min_ratio = config.get('body_min_ratio', 0.4)
        self.atr_period = config.get('atr_period', 14)
        self.trail_K = config.get('trail_K', 2.5)
        self.max_sl_atr = config.get('max_sl_atr', 3.3)
        self.emergency_sl_pct = config.get('emergency_sl_pct', 3.0)
        self.max_hold_bars = config.get('max_hold_bars', 192)
        self.sl_min_pct = config.get('sl_min_pct', 0.15)
        self.sl_max_pct = config.get('sl_max_pct', 3.0)
        self.min_bars_between_trades = config.get('min_bars_between', 2)
        self.trail_activation_pct = config.get('trail_activation_pct', 0.05)

    def check_entry(self, bar_open, bar_high, bar_low, bar_close,
                    channel_high, channel_low, atr_val,
                    last_swing_low, last_swing_high):
        """Check if current bar qualifies as a breakout entry.

        Args:
            bar_open/high/low/close: Current bar OHLC
            channel_high/low: N-bar channel values (computed on past bars)
            atr_val: Current ATR value
            last_swing_low/high: Most recent fractal swing points

        Returns:
            dict with 'direction', 'sl_price', 'sl_pct' or None
        """
        if math.isnan(channel_high) or math.isnan(channel_low) or math.isnan(atr_val) or atr_val <= 0:
            return None

        # BUG#53: channel sanity — flat/inverted data would produce spurious signals
        if channel_high <= channel_low:
            return None

        # Check breakout
        direction = None
        if bar_close > channel_high:
            direction = 'LONG'
        elif bar_close < channel_low:
            direction = 'SHORT'

        if direction is None:
            return None

        # Body confirmation: body > body_min_ratio × range
        rng = bar_high - bar_low
        if rng <= 0:
            return None
        body = bar_close - bar_open
        if abs(body) / rng < self.body_min_ratio:
            return None

        # Body direction must match breakout
        if direction == 'LONG' and body <= 0:
            return None
        if direction == 'SHORT' and body >= 0:
            return None

        # Compute SL price (fractal + ATR cap)
        # Entry will be next bar open, but we approximate with current close
        entry_approx = bar_close

        if direction == 'LONG':
            fractal_sl = last_swing_low if not math.isnan(last_swing_low) else entry_approx - self.max_sl_atr * atr_val
            atr_sl = entry_approx - self.max_sl_atr * atr_val
            sl_price = max(fractal_sl, atr_sl)  # Tighter of two
        else:
            fractal_sl = last_swing_high if not math.isnan(last_swing_high) else entry_approx + self.max_sl_atr * atr_val
            atr_sl = entry_approx + self.max_sl_atr * atr_val
            sl_price = min(fractal_sl, atr_sl)

        # Validate SL distance
        sl_pct = abs(entry_approx - sl_price) / entry_approx * 100
        if sl_pct < self.sl_min_pct or sl_pct > self.sl_max_pct:
            return None

        return {
            'direction': direction,
            'sl_price': sl_price,
            'sl_pct': sl_pct,
        }

    def check_exit(self, direction, entry_price, best_price,
                   current_high, current_low, current_close,
                   sl_price, atr_val, bars_held):
        """Check exit conditions for an open position.

        Priority: SL → Emergency → Timeout → Trail
        SL checked first: in reality price hits SL before Emergency
        (SL is always closer to entry than Emergency 3%).

        Returns:
            dict with 'reason', 'exit_price' or None (hold)
        """
        # 1. Fractal SL (closer to entry → triggers first in real market)
        if direction == 'LONG' and current_low <= sl_price:
            return {'reason': 'SL', 'exit_price': sl_price}
        elif direction == 'SHORT' and current_high >= sl_price:
            return {'reason': 'SL', 'exit_price': sl_price}

        # 2. Emergency SL (only if SL didn't catch it — gap/extreme move)
        if direction == 'LONG':
            worst_pnl = (current_low / entry_price - 1) * 100
        else:
            worst_pnl = (1 - current_high / entry_price) * 100

        if worst_pnl <= -self.emergency_sl_pct:
            if direction == 'LONG':
                exit_price = entry_price * (1 - self.emergency_sl_pct / 100)
            else:
                exit_price = entry_price * (1 + self.emergency_sl_pct / 100)
            return {'reason': 'EMERGENCY', 'exit_price': exit_price}

        # 3. Timeout
        if bars_held >= self.max_hold_bars:
            return {'reason': 'TIMEOUT', 'exit_price': current_close}

        # 4. Trailing TP
        if direction == 'LONG':
            best_pnl = (best_price / entry_price - 1) * 100
            cur_pnl = (current_close / entry_price - 1) * 100
        else:
            best_pnl = (1 - best_price / entry_price) * 100
            cur_pnl = (1 - current_close / entry_price) * 100

        if best_pnl > self.trail_activation_pct and not math.isnan(atr_val) and atr_val > 0:
            trail_dist_pct = self.trail_K * atr_val / current_close * 100
            drawdown = best_pnl - cur_pnl
            if drawdown >= trail_dist_pct:
                realized_pnl = max(0, best_pnl - trail_dist_pct)
                if direction == 'LONG':
                    exit_price = entry_price * (1 + realized_pnl / 100)
                else:
                    exit_price = entry_price * (1 - realized_pnl / 100)
                return {'reason': 'TRAIL_TP', 'exit_price': exit_price}

        return None  # Hold
