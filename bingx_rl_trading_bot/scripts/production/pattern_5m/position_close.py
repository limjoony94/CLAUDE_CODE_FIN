"""
Pattern 5m Bot - Position Closing and Recovery
Functions for closing positions and crash recovery.
"""

import logging
import os
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import ccxt

from .constants import (
    FEE_PCT,
    QTY_TOLERANCE,
    QUANTITY_ROUND_DECIMALS,
    CONFIDENCE_LOG_FILE,
    STATE_FILE,
    LOG_DIR,
    BOT_NAME,
)
from .position_open import calculate_tp_sl, setup_scale_out
from .models import APICache, CircuitBreaker, PerformanceMetrics
from .exchange import fetch_positions_cached, fetch_ticker_cached, fetch_balance_cached
from .state import save_state, save_metrics
from .orders import (
    place_tp_sl_orders, cancel_remaining_orders, update_emergency_sl,
    cancel_emergency_sl, _get_position_side,
)
from .utils import extract_pattern_name

logger = logging.getLogger('pattern_5m')


def detect_ghost_positions(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    cache: APICache,
    circuit_breaker: Optional[CircuitBreaker] = None,
    metrics: Optional[PerformanceMetrics] = None,
) -> None:
    """
    Ghost position detection - warn if exchange position not tracked locally.

    Ghost positions는 거래소에 있지만 local state에 없는 포지션입니다.
    이는 봇 재시작, crash, 또는 manual trading으로 발생할 수 있습니다.

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration
        cache: APICache instance
        circuit_breaker: Optional CircuitBreaker
        metrics: Optional PerformanceMetrics
    """
    symbol = config['symbol']

    try:
        positions = fetch_positions_cached(exchange, symbol, cache, force_refresh=True,
                                           circuit_breaker=circuit_breaker, metrics=metrics)

        # Build exchange map by direction
        exchange_map = {}
        for pos in positions:
            contracts = float(pos.get('contracts', 0))
            if contracts > 0:
                exchange_map[pos.get('side')] = pos

        # Build set of directions tracked by bot
        bot_slots = state.get('positions') or {}
        bot_tracked_sides = set()
        for slot in bot_slots.values():
            d = slot.get('direction', '')
            bot_tracked_sides.add('long' if d == 'LONG' else 'short')

        # Ghost: exchange has position in a direction bot doesn't track
        for side, ex_pos in exchange_map.items():
            if side not in bot_tracked_sides:
                direction = 'LONG' if side == 'long' else 'SHORT'
                qty = float(ex_pos.get('contracts', 0))
                entry_price = float(ex_pos.get('entryPrice', 0))

                logger.warning(
                    f"👻 GHOST POSITION DETECTED on exchange:\n"
                    f"   Symbol: {symbol}\n"
                    f"   Direction: {direction}\n"
                    f"   Quantity: {qty}\n"
                    f"   Entry Price: {entry_price}\n"
                    f"   → This position is not tracked in local state\n"
                    f"   → Crash recovery will attempt to reconcile"
                )
    except Exception as e:
        logger.exception(f"❌ Ghost position detection failed: {e}")


def calculate_pnl(
    entry_price: float,
    exit_price: float,
    direction: int,
    leverage: int,
) -> Tuple[float, float]:
    """
    Calculate leveraged and price-basis PnL percentage.

    Args:
        entry_price: Entry price (must be > 0)
        exit_price: Exit price
        direction: 1 for LONG, -1 for SHORT
        leverage: Leverage multiplier

    Returns:
        Tuple of (pnl_pct_leveraged, pnl_pct_price)
    """
    price_move = direction * (exit_price / entry_price - 1) * 100
    pnl_pct = price_move * leverage - 2 * FEE_PCT * leverage
    price_pnl_pct = price_move - 2 * FEE_PCT
    return pnl_pct, price_pnl_pct


def record_closed_position(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    exit_price: float,
    exit_reason: str,
    cache: APICache,
    metrics: Optional[PerformanceMetrics] = None,
    position: Optional[Dict] = None,
) -> None:
    """
    Record a closed position and update statistics.

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration
        exit_price: Exit price
        exit_reason: Reason for exit (TP, SL, etc.)
        cache: APICache instance
        metrics: Optional PerformanceMetrics
        position: Specific slot to close (if None, uses first slot)
    """
    if position is None:
        positions = state.get('positions') or {}
        if not positions:
            return
        position = next(iter(positions.values()))

    # v1.55.2: Duplicate trade guard — skip if identical trade was already recorded
    # Prevents mass closure loop from recording same position multiple times
    if metrics and hasattr(metrics, 'trade_history') and metrics.trade_history:
        recent = metrics.trade_history[-10:]  # check last 10 entries
        for prev in recent:
            if (abs(prev.get('entry_price', 0) - position.get('entry_price', -1)) < 0.5
                    and abs(prev.get('exit_price', 0) - exit_price) < 0.5
                    and prev.get('direction') == position.get('direction')):
                logger.info(
                    f"⏭️ Duplicate trade guard: {position['direction']} "
                    f"entry=${position['entry_price']:.1f} exit=${exit_price:.1f} "
                    f"already recorded — skipping"
                )
                # Still remove the slot from state but don't record duplicate trade
                # v1.57.1: Cancel remaining TP/SL orders for this slot (prevents orphan orders)
                if exchange:
                    cancel_remaining_orders(exchange, state, config, position=position)
                slot_id = position.get('slot_id')
                positions_dict = state.get('positions') or {}
                if slot_id and slot_id in positions_dict:
                    del positions_dict[slot_id]
                state['has_position'] = len(positions_dict) > 0
                if not positions_dict:
                    state['active_direction'] = None
                save_state(state, is_trade_close=True)
                cache.invalidate_all()
                return

    if exchange:
        cancel_remaining_orders(exchange, state, config, position=position)

    direction = 1 if position['direction'] == 'LONG' else -1

    if not position.get('entry_price') or position['entry_price'] <= 0:
        logger.error(f"Invalid entry_price={position.get('entry_price')} — cannot calculate PnL, recording 0%")
        pnl_pct = 0.0
        price_pnl_pct = 0.0
    else:
        # v1.39.0: Use per-slot effective leverage if available (adaptive leverage)
        slot_leverage = position.get('effective_leverage', config['leverage'])
        pnl_pct, price_pnl_pct = calculate_pnl(
            entry_price=position['entry_price'],
            exit_price=exit_price,
            direction=direction,
            leverage=slot_leverage,
        )

    # Extract pattern name: slot field > reason regex > log recovery > fallback 'N/A'
    # v1.55.2: Prefer direct pattern_name field (set during entry/recovery)
    pattern_name = (
        position.get('pattern_name')
        or extract_pattern_name(position.get('reason', ''))
    )
    # v1.59.4: Last-resort log recovery — prevents N/A contamination of trade_history
    if not pattern_name:
        try:
            log_pats = _recover_patterns_from_logs(position.get('direction', ''))
            if log_pats:
                pattern_name = log_pats[0][0]
                logger.info(f"✅ Last-resort pattern recovery: '{pattern_name}' from logs")
        except Exception:
            pass
    if not pattern_name:
        pattern_name = 'N/A'

    # Calculate hold time
    entry_time_str = position.get('entry_time', '')
    hold_minutes = 0
    if entry_time_str:
        try:
            entry_dt = datetime.fromisoformat(entry_time_str)
            hold_minutes = int((datetime.now() - entry_dt).total_seconds() / 60)
        except (ValueError, TypeError):
            pass

    # Scale PnL to portfolio basis (1/N sizing)
    max_positions = config.get('max_positions', 1)
    portfolio_pnl_pct = pnl_pct / max_positions

    logger.info(
        f"🏁 TRADE CLOSED | {position['direction']} {pattern_name} | "
        f"Entry: ${position['entry_price']:.1f} → Exit: ${exit_price:.1f} | "
        f"PnL: {pnl_pct:+.2f}% (slot) / {portfolio_pnl_pct:+.2f}% (portfolio) | "
        f"TP: ${position.get('tp_price', 0):.1f} SL: ${position.get('sl_price', 0):.1f} | "
        f"Reason: {exit_reason} | Hold: {hold_minutes}m"
    )

    # Update metrics (portfolio-scaled PnL) + trade history
    if metrics:
        trade_detail = {
            'timestamp': datetime.now().isoformat(),
            'close_time': datetime.now().isoformat(),
            'pattern': pattern_name,
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl_pct': round(pnl_pct, 4),
            'pnl_slot': round(pnl_pct, 4),
            'pnl_portfolio': round(portfolio_pnl_pct, 4),
            'tp_price': position.get('tp_price', 0),
            'sl_price': position.get('sl_price', 0),
            'exit_reason': exit_reason,
            'hold_minutes': hold_minutes,
            'effective_leverage': position.get('effective_leverage', config['leverage']),
            'vol_mult': position.get('vol_mult', 1.0),
            'equity_curve_scale': position.get('equity_curve_scale', 1.0),
        }
        metrics.update_trade(portfolio_pnl_pct, trade_detail=trade_detail)

    # Update state statistics (portfolio-scaled PnL)
    state['total_trades'] += 1
    state['total_pnl'] += portfolio_pnl_pct
    state['daily_trades'] += 1
    state['daily_pnl'] += portfolio_pnl_pct

    if pnl_pct >= 0:
        state['winning_trades'] += 1
        state['consecutive_losses'] = 0
    else:
        state['consecutive_losses'] = state.get('consecutive_losses', 0) + 1

    state['last_trade'] = {
        'direction': position['direction'],
        'entry_price': position['entry_price'],
        'exit_price': exit_price,
        'pnl_pct': pnl_pct,
        'portfolio_pnl_pct': portfolio_pnl_pct,
        'exit_reason': exit_reason,
        'closed_at': datetime.now().isoformat(),
    }

    state['last_signal_time'] = datetime.now().isoformat()

    # Remove this slot from positions dict
    slot_id = position.get('slot_id')
    positions = state.get('positions') or {}
    if slot_id and slot_id in positions:
        del positions[slot_id]
    elif slot_id is None:
        # Legacy fallback: remove first slot matching direction
        for sid, s in list(positions.items()):
            if s.get('direction') == position.get('direction'):
                del positions[sid]
                break
    state['has_position'] = len(positions) > 0
    # Update active_direction
    if not positions:
        state['active_direction'] = None

    # v1.29.0: Update emergency SL after slot removal
    if exchange and positions:
        update_emergency_sl(exchange, state, config)
    elif exchange and not positions:
        cancel_emergency_sl(exchange, state, config)

    save_state(state, is_trade_close=True)

    # Save metrics immediately after trade close
    if metrics:
        save_metrics(metrics)
        logger.info(f"📊 Metrics saved: {metrics.total_trades} trades, {metrics.actual_win_rate:.1f}% WR")

    # Update confidence log with trade outcome
    outcome = "WIN" if pnl_pct >= 0 else "LOSS"
    _update_confidence_log_outcome(position.get('entry_time'), outcome, pnl_pct)

    # Invalidate position cache (don't set empty — forces fresh fetch next time)
    cache.invalidate_all()


def _recover_pattern_from_history(
    entry_price: float,
    direction: str,
    config: Dict[str, Any],
) -> Tuple[Optional[str], float]:
    """Search metrics trade_history for pattern_name matching entry_price+direction.

    v1.55.0: When crash recovery can't find pattern_name from state (e.g. after
    false mass closure), search recent trade_history for the same position that
    was just incorrectly recorded as closed.

    Returns:
        Tuple of (pattern_name, vol_mult). pattern_name may be None.
    """
    from .state import load_metrics
    try:
        metrics = load_metrics()
        if not metrics:
            return None, 1.0
        history = getattr(metrics, 'trade_history', None) or []
        # Search last 20 trades for matching entry_price + direction
        for trade in reversed(history[-20:]):
            if (trade.get('direction') == direction
                    and abs(trade.get('entry_price', 0) - entry_price) < 0.5):
                pat = trade.get('pattern')
                if pat and pat != 'N/A':
                    vol_mult = trade.get('vol_mult', 1.0)
                    logger.info(
                        f"✅ Recovery: pattern_name '{pat}' recovered from trade_history "
                        f"(entry=${entry_price:.1f}, {direction})"
                    )
                    return pat, vol_mult
    except Exception as e:
        logger.debug(f"Pattern recovery from history failed: {e}")
    return None, 1.0


def _recover_patterns_from_state_backups(
    direction: str,
) -> List[Tuple[str, float]]:
    """Search state backup files for pattern_name+vol_mult of positions matching direction.

    v1.59.4: BingX merges same-direction positions into averaged entry_price,
    so entry_price matching is unreliable. We match by direction only.

    Search order: .bak → .new → timestamped backups (newest first).
    Stops at first backup that yields results.

    Returns:
        List of (pattern_name, vol_mult) tuples. Empty list on failure.
    """
    import json

    results = []

    def _extract_from_state_data(data: Dict) -> List[Tuple[str, float]]:
        """Extract (pattern_name, vol_mult) for matching direction from state dict."""
        found = []
        positions = data.get('positions') or {}
        for slot in positions.values():
            if slot.get('direction') != direction:
                continue
            pat = slot.get('pattern_name')
            if not pat or pat == 'N/A':
                pat = extract_pattern_name(slot.get('reason', ''))
            if pat and pat != 'N/A':
                found.append((pat, slot.get('vol_mult', 1.0)))
        return found

    try:
        # 1. Try .bak file
        bak_file = STATE_FILE + '.bak'
        if os.path.exists(bak_file):
            try:
                with open(bak_file, 'r') as f:
                    data = json.load(f)
                results = _extract_from_state_data(data)
                if results:
                    logger.info(
                        f"✅ Recovery: {len(results)} pattern(s) recovered from .bak "
                        f"for {direction}: {[r[0] for r in results]}"
                    )
                    return results
            except (json.JSONDecodeError, IOError, OSError):
                pass

        # 2. Try .new file
        new_file = STATE_FILE + '.new'
        if os.path.exists(new_file):
            try:
                with open(new_file, 'r') as f:
                    data = json.load(f)
                results = _extract_from_state_data(data)
                if results:
                    logger.info(
                        f"✅ Recovery: {len(results)} pattern(s) recovered from .new "
                        f"for {direction}: {[r[0] for r in results]}"
                    )
                    return results
            except (json.JSONDecodeError, IOError, OSError):
                pass

        # 3. Try timestamped backups (newest first)
        state_dir = os.path.dirname(STATE_FILE)
        state_name = os.path.basename(STATE_FILE)
        backup_prefix = f"{state_name}.backup_"

        backups = []
        if os.path.isdir(state_dir):
            for filename in os.listdir(state_dir):
                if filename.startswith(backup_prefix):
                    filepath = os.path.join(state_dir, filename)
                    backups.append((filepath, os.path.getmtime(filepath)))

        if backups:
            backups.sort(key=lambda x: x[1], reverse=True)
            for filepath, _ in backups:
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    results = _extract_from_state_data(data)
                    if results:
                        logger.info(
                            f"✅ Recovery: {len(results)} pattern(s) recovered from "
                            f"{os.path.basename(filepath)} for {direction}: "
                            f"{[r[0] for r in results]}"
                        )
                        return results
                except (json.JSONDecodeError, IOError, OSError):
                    continue

    except Exception as e:
        logger.debug(f"Pattern recovery from state backups failed: {e}")

    return []


def _recover_patterns_from_logs(
    direction: str,
) -> List[Tuple[str, float]]:
    """Search recent log files for open positions matching direction that were never closed.

    v1.59.4: Last-resort recovery when state backups and trade_history both fail.
    Scans logs for "Position opened" events and subtracts "TRADE CLOSED" events
    to find positions still open on the exchange.

    Log formats:
        Open:  "Position opened (slot XXXX): ORDER_ID [N/M slots]"
        Prior: "Signal detected: LONG | Pattern: BU-BU-DN (LONG)"
        Close: "TRADE CLOSED | LONG DN-ST-MU | Entry: $70219.3 ..."

    Returns:
        List of (pattern_name, vol_mult) tuples. vol_mult defaults to 1.0 (not in logs).
    """
    import re
    import glob as glob_mod

    try:
        # Find log files, newest first
        log_pattern = os.path.join(LOG_DIR, f"{BOT_NAME}_*.log")
        log_files = sorted(glob_mod.glob(log_pattern), reverse=True)
        if not log_files:
            return []

        # Track opened and closed patterns per direction
        opened_patterns = []  # [(pattern, entry_price), ...]
        closed_counts = {}    # {(pattern, entry_price_str): count}

        # Regex patterns
        re_signal = re.compile(
            r'Signal detected: (LONG|SHORT) \| Pattern: ([A-Z]+-[A-Z]+-[A-Z]+)'
        )
        re_opened = re.compile(r'Position opened \(slot ([0-9a-f]+)\)')
        re_closed = re.compile(
            r'TRADE CLOSED \| (LONG|SHORT) ([A-Z/]+-?[A-Z]*-?[A-Z]*) \| Entry: \$([0-9.]+)'
        )
        re_entry_price = re.compile(r'Actual fill: \$([0-9.]+)')

        # Scan last 2 log files (covers cross-day scenarios)
        for log_file in log_files[:2]:
            try:
                with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()
            except (IOError, OSError):
                continue

            pending_signal = None  # (direction, pattern)
            pending_entry_price = None

            for line in lines:
                # Track signal → entry_price → opened sequence
                m_sig = re_signal.search(line)
                if m_sig:
                    pending_signal = (m_sig.group(1), m_sig.group(2))
                    pending_entry_price = None
                    continue

                m_price = re_entry_price.search(line)
                if m_price and pending_signal:
                    pending_entry_price = m_price.group(1)
                    continue

                m_open = re_opened.search(line)
                if m_open and pending_signal and pending_entry_price:
                    sig_dir, sig_pat = pending_signal
                    if sig_dir == direction:
                        opened_patterns.append((sig_pat, pending_entry_price))
                    pending_signal = None
                    pending_entry_price = None
                    continue

                # Track closures
                m_close = re_closed.search(line)
                if m_close:
                    close_dir = m_close.group(1)
                    close_pat = m_close.group(2)
                    close_price = m_close.group(3)
                    if close_dir == direction and close_pat != 'N/A':
                        key = (close_pat, close_price)
                        closed_counts[key] = closed_counts.get(key, 0) + 1

        # Subtract closed from opened to find still-open positions
        still_open = []
        for pat, price in opened_patterns:
            key = (pat, price)
            if closed_counts.get(key, 0) > 0:
                closed_counts[key] -= 1  # consume one closure
            else:
                still_open.append((pat, 1.0))  # vol_mult not in logs

        if still_open:
            logger.info(
                f"✅ Recovery: {len(still_open)} pattern(s) recovered from logs "
                f"for {direction}: {[s[0] for s in still_open]}"
            )

        return still_open

    except Exception as e:
        logger.debug(f"Pattern recovery from logs failed: {e}")
        return []


def recover_position_to_state(
    state: Dict[str, Any],
    config: Dict[str, Any],
    exchange_pos: Dict,
    direction: str,
    exchange: Optional[ccxt.bingx] = None,
    cache: Optional[APICache] = None,
    saved_tpsl_pairs: Optional[list] = None,
) -> None:
    """
    Recover position from exchange to local state.

    Args:
        state: Bot state dictionary
        config: Bot configuration
        exchange_pos: Exchange position dictionary
        direction: 'LONG' or 'SHORT'
        exchange: Optional exchange instance for placing TP/SL
        cache: Optional APICache instance
        saved_tpsl_pairs: Optional pre-saved per-slot (tp, sl) tuples from
            _snapshot_all_tpsl() — used when orders were cancelled before this call
    """
    strategy = config['strategy']
    entry_price = float(exchange_pos.get('entryPrice', 0))
    quantity = float(exchange_pos.get('contracts', 0))
    dir_mult = 1 if direction == 'LONG' else -1

    # v1.57.1: Duplicate recovery guard — skip if recovered slots for this direction exist
    # (Prevents same exchange position from being recovered multiple times)
    existing_recovered = [
        s for s in (state.get('positions') or {}).values()
        if s.get('direction') == direction and s.get('recovered', False)
    ]
    if existing_recovered:
        logger.warning(
            f"⚠️ Recovery skipped: {len(existing_recovered)} recovered {direction} slot(s) "
            f"already exist (entry=${existing_recovered[0].get('entry_price', 0):.1f}). "
            f"Not creating duplicate recovery slots."
        )
        return

    # Try to find pattern_name and vol_mult from existing slots for this direction
    old_pattern_name = None
    old_vol_mult = 1.0
    backup_patterns = []  # v1.59.4: multi-pattern list from backups
    for slot in (state.get('positions') or {}).values():
        if slot.get('direction') == direction:
            old_pattern_name = extract_pattern_name(slot.get('reason', '')) or slot.get('pattern_name')
            old_vol_mult = slot.get('vol_mult', 1.0)
            if old_pattern_name:
                break

    # v1.59.4: Try state backup files (.bak, .new, timestamped) for pattern recovery
    # BingX averages entry_price for same-direction positions, so price matching fails
    if not old_pattern_name:
        backup_patterns = _recover_patterns_from_state_backups(direction)
        if backup_patterns:
            old_pattern_name, old_vol_mult = backup_patterns[0]
            logger.info(f"Recovery source: state backup ({len(backup_patterns)} pattern(s))")

    # v1.59.4: Try log files for pattern recovery (open - closed = still active)
    # Also used to supplement backup_patterns with more patterns for multi-slot recovery
    log_patterns = _recover_patterns_from_logs(direction)
    if log_patterns:
        if not old_pattern_name:
            old_pattern_name, old_vol_mult = log_patterns[0]
            backup_patterns = log_patterns
            logger.info(f"Recovery source: log files ({len(log_patterns)} pattern(s))")
        elif len(log_patterns) > len(backup_patterns):
            # Log has more patterns (better for multi-slot assignment)
            backup_patterns = log_patterns
            logger.info(
                f"Recovery: supplementing with log patterns "
                f"({len(log_patterns)} > {len(backup_patterns)} from backup)"
            )

    # v1.55.0: Fallback — search recent trade_history for matching entry+direction
    # This handles false mass closures where slots were removed but position still exists
    if not old_pattern_name:
        old_pattern_name, old_vol_mult = _recover_pattern_from_history(
            entry_price, direction, config
        )

    if not old_pattern_name:
        logger.warning(
            f"⚠️ Recovery: pattern_name unavailable for {direction} — "
            f"per-pattern TP/SL adjustment will use defaults until manually restored"
        )

    # Determine TP/SL source (priority: saved snapshot > exchange orders > config)
    if saved_tpsl_pairs:
        # Use first pair as default fallback for slots beyond saved count
        default_tp, default_sl = saved_tpsl_pairs[0]
        if not default_sl:
            _, default_sl, _, _ = calculate_tp_sl(
                entry_price, dir_mult, strategy, vol_mult=old_vol_mult, pattern=old_pattern_name, config=config
            )
        tp_pct_adjusted = abs(default_tp / entry_price - 1) * 100 if entry_price > 0 else 1.0
        logger.info(
            f"Recovered {direction} position: entry=${entry_price:.1f} | "
            f"Using {len(saved_tpsl_pairs)} saved TP/SL pair(s)"
        )
    else:
        # Try to read TP/SL from existing exchange orders (preserves per-pattern values)
        tp_from_exchange, sl_from_exchange = _read_tpsl_from_exchange_orders(
            exchange, config.get('symbol', 'BTC-USDT'), direction
        )

        if tp_from_exchange and sl_from_exchange:
            default_tp = tp_from_exchange
            default_sl = sl_from_exchange
            tp_pct_adjusted = abs(default_tp / entry_price - 1) * 100 if entry_price > 0 else 1.0
            logger.info(
                f"Recovered {direction} position: entry=${entry_price:.1f} | "
                f"TP/SL from exchange orders: TP=${default_tp:.1f}, SL=${default_sl:.1f}"
            )
        else:
            # Fallback: calculate from config defaults (pass pattern for per-pattern lookup)
            default_tp, default_sl, tp_pct_adjusted, _ = calculate_tp_sl(
                entry_price, dir_mult, strategy, vol_mult=old_vol_mult, pattern=old_pattern_name, config=config
            )
            logger.info(
                f"Recovered {direction} position: entry=${entry_price:.1f} | "
                f"TP/SL from config{f' (pattern={old_pattern_name})' if old_pattern_name else ' defaults'}"
                f" (vol_mult={old_vol_mult:.4f}): "
                f"TP=${default_tp:.1f}, SL=${default_sl:.1f}"
            )

    # Determine how many recovery slots to create (N=1 for single-position mode)
    max_positions = config.get('max_positions', 1)
    n_slots = _calculate_recovery_slot_count(
        exchange, cache, config, quantity, entry_price, max_positions, state
    )
    per_slot_qty = round(quantity / n_slots, QUANTITY_ROUND_DECIMALS)

    positions = state.setdefault('positions', {})
    new_slot_ids = []

    for i in range(n_slots):
        # Last slot gets remainder to avoid rounding loss
        if i == n_slots - 1:
            slot_qty = round(quantity - per_slot_qty * (n_slots - 1), QUANTITY_ROUND_DECIMALS)
        else:
            slot_qty = per_slot_qty

        # v1.59.4: Per-slot pattern from backup (round-robin if fewer patterns than slots)
        if backup_patterns:
            slot_pattern, slot_vol_mult = backup_patterns[i % len(backup_patterns)]
        else:
            slot_pattern = old_pattern_name
            slot_vol_mult = old_vol_mult

        # Per-slot TP/SL from saved snapshot (if available)
        if saved_tpsl_pairs and i < len(saved_tpsl_pairs):
            slot_tp = saved_tpsl_pairs[i][0] or default_tp
            slot_sl = saved_tpsl_pairs[i][1] or default_sl
        elif backup_patterns and slot_pattern != old_pattern_name:
            # Recalculate TP/SL for this slot's specific pattern
            slot_tp, slot_sl, _, _ = calculate_tp_sl(
                entry_price, dir_mult, strategy,
                vol_mult=slot_vol_mult, pattern=slot_pattern, config=config
            )
        else:
            slot_tp = default_tp
            slot_sl = default_sl

        slot_reason = f"Recovered from exchange ({slot_pattern})" if slot_pattern else 'Recovered from exchange'
        slot_tp_pct = abs(slot_tp / entry_price - 1) * 100 if entry_price > 0 else 1.0
        scale_out_stages = setup_scale_out(strategy, entry_price, slot_qty, dir_mult, slot_tp_pct)
        slot_id = uuid.uuid4().hex[:8]
        recovered_slot = {
            'slot_id': slot_id,
            'direction': direction,
            'entry_price': entry_price,
            'quantity': slot_qty,
            'remaining_quantity': slot_qty,
            'tp_price': slot_tp,
            'sl_price': slot_sl,
            'vol_mult': slot_vol_mult,
            'scale_out_enabled': bool(scale_out_stages),
            'scale_out_stages': scale_out_stages,
            'entry_time': datetime.now().isoformat(),
            'reason': slot_reason,
            'pattern_name': slot_pattern or None,
            'recovered': True,
            'needs_tpsl': True,
            # v1.68.0: Set _sl_price_original at recovery to prevent vol_adapt corruption
            '_sl_price_original': slot_sl,
        }
        positions[slot_id] = recovered_slot
        new_slot_ids.append(slot_id)

    # One-Way mode: track direction; Hedge: no single-direction constraint
    if config.get('position_mode') != 'hedge':
        state['active_direction'] = direction
    state['has_position'] = True
    logger.info(f"Recovery: created {n_slots} slot(s) from {quantity:.4f} total qty")
    save_state(state)

    if exchange:
        # v1.59.4: Clean up stale orders for this direction before placing new ones.
        # After cascade SL closures, old TP/SL orders may remain if cancel failed
        # (e.g., network error). Cancel them now to prevent over-coverage.
        try:
            symbol = config.get('symbol', 'BTC-USDT')
            open_orders = exchange.fetch_open_orders(symbol)
            new_order_ids = set()  # will be populated below
            stale_dir_orders = []
            for o in open_orders:
                info = o.get('info', {})
                pos_side = info.get('positionSide', '')
                otype = str(info.get('type', ''))
                if pos_side == direction and ('TAKE_PROFIT' in otype or 'STOP' in otype):
                    stale_dir_orders.append(o)
            if stale_dir_orders:
                logger.info(
                    f"Recovery cleanup: cancelling {len(stale_dir_orders)} "
                    f"stale {direction} orders before placing new ones"
                )
                for o in stale_dir_orders:
                    try:
                        exchange.cancel_order(o['id'], symbol)
                        logger.info(f"🗑️ Cancelled stale {direction} order: {o['id']}")
                    except ccxt.OrderNotFound:
                        pass
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to cancel stale order {o['id']}: {e}")
        except Exception as e:
            logger.warning(f"⚠️ Recovery cleanup failed (non-fatal): {e}")

        for sid in new_slot_ids:
            slot = positions[sid]
            place_tp_sl_orders(exchange, state, config, position=slot)
            if slot.get('tp_order_id') or slot.get('sl_order_id') or slot.get('scale_out_stages'):
                slot['needs_tpsl'] = False
        save_state(state)


def _calculate_recovery_slot_count(
    exchange: Optional[ccxt.bingx],
    cache: Optional[APICache],
    config: Dict[str, Any],
    quantity: float,
    entry_price: float,
    max_positions: int,
    state: Dict[str, Any],
) -> int:
    """Calculate how many virtual slots to create during orphan recovery.

    Uses current equity to estimate expected per-slot quantity,
    then divides exchange quantity by that estimate.
    Falls back to 1 slot if calculation fails.
    """
    if max_positions <= 1 or not exchange or not cache:
        return 1

    existing_slots = len(state.get('positions') or {})
    available_slots = max(1, max_positions - existing_slots)

    try:
        balance = fetch_balance_cached(exchange, cache, force_refresh=True)
        total_equity = float(balance.get('USDT', {}).get('total', 0))

        if total_equity <= 0 or entry_price <= 0:
            return 1

        size_pct = config.get('position_size_pct', 95) / 100
        leverage = config.get('leverage', 3)
        per_slot_equity = total_equity * size_pct / max_positions
        expected_slot_qty = round(
            (per_slot_equity * leverage) / entry_price,
            QUANTITY_ROUND_DECIMALS,
        )

        if expected_slot_qty <= 0:
            return 1

        n_slots = max(1, min(available_slots, round(quantity / expected_slot_qty)))
        logger.info(
            f"Recovery slot calc: qty={quantity:.4f}, "
            f"expected_per_slot={expected_slot_qty:.4f}, "
            f"available={available_slots} → {n_slots} slot(s)"
        )
        return n_slots

    except Exception as e:
        logger.warning(f"Could not calculate recovery slot count, using 1: {e}")
        return 1


def _read_tpsl_from_exchange_orders(
    exchange: Optional[ccxt.bingx],
    symbol: str,
    direction: str,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Read TP and SL prices from existing exchange open orders.

    With multi-slot, multiple TP/SL orders may exist. Returns the widest
    (most protective) pair based on direction:
    - LONG: highest TP, lowest SL
    - SHORT: lowest TP, highest SL

    Args:
        exchange: Exchange instance (may be None)
        symbol: Trading symbol
        direction: 'LONG' or 'SHORT'

    Returns:
        Tuple of (tp_price, sl_price) — either may be None if not found
    """
    if not exchange:
        return None, None

    try:
        open_orders = exchange.fetch_open_orders(symbol)
        tp_prices = []
        sl_prices = []

        for order in open_orders:
            # In hedge mode, filter by positionSide to avoid mixing directions
            order_pos_side = (
                (order.get('info') or {}).get('positionSide', 'BOTH')
            ).upper()
            if order_pos_side != 'BOTH' and order_pos_side != direction:
                continue

            order_type = (order.get('type') or '').upper()
            # CCXT normalizes stopPrice; BingX also provides it in info
            stop_price = float(
                order.get('stopPrice')
                or (order.get('info') or {}).get('stopPrice')
                or 0
            )
            if stop_price <= 0:
                continue

            if 'TAKE_PROFIT' in order_type:
                tp_prices.append(stop_price)
            elif 'STOP' in order_type and 'TAKE' not in order_type:
                sl_prices.append(stop_price)

        # Select widest pair based on direction for maximum protection
        tp_price = None
        sl_price = None
        if tp_prices:
            tp_price = max(tp_prices) if direction == 'LONG' else min(tp_prices)
        if sl_prices:
            sl_price = min(sl_prices) if direction == 'LONG' else max(sl_prices)

        if tp_prices or sl_prices:
            tp_str = f"${tp_price:.1f}" if tp_price else "None"
            sl_str = f"${sl_price:.1f}" if sl_price else "None"
            logger.debug(
                f"Exchange orders found: {len(tp_prices)} TP, {len(sl_prices)} SL → "
                f"TP={tp_str}, SL={sl_str}"
            )

        return tp_price, sl_price

    except Exception as e:
        logger.debug(f"Could not read TP/SL from exchange orders: {e}")
        return None, None


def _snapshot_all_tpsl(
    exchange: Optional[ccxt.bingx],
    symbol: str,
    direction: str,
) -> list:
    """Snapshot per-slot TP/SL prices from exchange orders before cancellation.

    Called BEFORE Phase 1 order cancellation to preserve per-pattern TP/SL values.
    Returns list of (tp_price, sl_price) tuples, one per detected slot.
    Emergency SL is filtered out (identified by largest amount when SL count > TP count).

    Args:
        exchange: Exchange instance (may be None)
        symbol: Trading symbol
        direction: 'LONG' or 'SHORT'

    Returns:
        List of (tp_price, sl_price) tuples. Empty list if no TP orders found.
    """
    if not exchange:
        return []

    try:
        open_orders = exchange.fetch_open_orders(symbol)
        tp_entries = []  # (price, amount)
        sl_entries = []  # (price, amount)

        for order in open_orders:
            # In hedge mode, filter by positionSide to avoid mixing directions
            order_pos_side = (
                (order.get('info') or {}).get('positionSide', 'BOTH')
            ).upper()
            if order_pos_side != 'BOTH' and order_pos_side != direction:
                continue

            order_type = (order.get('type') or '').upper()
            stop_price = float(
                order.get('stopPrice')
                or (order.get('info') or {}).get('stopPrice')
                or 0
            )
            amount = float(order.get('amount', 0))
            if stop_price <= 0:
                continue

            if 'TAKE_PROFIT' in order_type:
                tp_entries.append((stop_price, amount))
            elif 'STOP' in order_type and 'TAKE' not in order_type:
                sl_entries.append((stop_price, amount))

        if not tp_entries:
            return []

        # Filter emergency SL: if more SLs than TPs, remove the one with largest amount
        per_slot_sls = list(sl_entries)
        if len(sl_entries) > len(tp_entries):
            sl_by_amount = sorted(sl_entries, key=lambda x: x[1], reverse=True)
            emergency = sl_by_amount[0]
            per_slot_sls = sl_by_amount[1:]
            logger.debug(
                f"Snapshot: filtered emergency SL price=${emergency[0]:.1f} "
                f"amount={emergency[1]:.4f}"
            )

        # Sort for consistent pairing by "risk profile width"
        # Pair widest TP with widest SL (both farther from entry)
        tp_prices = [p for p, _ in tp_entries]
        sl_prices = [p for p, _ in per_slot_sls]

        if direction == 'SHORT':
            # SHORT: lower TP = wider (farther below entry), higher SL = wider
            tp_prices.sort()            # ascending = widest first
            sl_prices.sort(reverse=True)  # descending = widest first
        else:
            # LONG: higher TP = wider (farther above entry), lower SL = wider
            tp_prices.sort(reverse=True)  # descending = widest first
            sl_prices.sort()              # ascending = widest first

        # Pair by index (widest-with-widest)
        n_pairs = min(len(tp_prices), len(sl_prices))
        pairs = [(tp_prices[i], sl_prices[i]) for i in range(n_pairs)]

        # Append unpaired TPs with None SL
        for i in range(n_pairs, len(tp_prices)):
            pairs.append((tp_prices[i], None))

        if pairs:
            logger.info(
                f"Snapshot: {len(tp_entries)} TP, {len(sl_entries)} SL orders "
                f"→ {len(pairs)} pair(s) saved before cancellation"
            )

        return pairs

    except Exception as e:
        logger.warning(f"Could not snapshot TP/SL from exchange for {direction}: {e}")
        return []


def recalculate_position_orders(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    new_quantity: float,
    position: Optional[Dict] = None,
) -> bool:
    """
    Recalculate scale-out stages and update TP/SL orders when position quantity changes.

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration
        new_quantity: New position quantity from exchange
        position: Specific slot to recalculate (if None, uses first slot)

    Returns:
        True if recalculation was successful
    """
    if position is None:
        positions = state.get('positions') or {}
        if not positions:
            return False
        position = next(iter(positions.values()))

    strategy = config['strategy']
    entry_price = position.get('entry_price', 0)
    direction = 1 if position['direction'] == 'LONG' else -1
    vol_mult = position.get('vol_mult', 1.0)

    # Cancel existing orders FIRST (before modifying scale_out_stages)
    try:
        cancel_remaining_orders(exchange, state, config, position=position)
    except Exception as e:
        logger.warning(f"Failed to cancel existing orders: {e}")

    # Update quantities
    position['quantity'] = new_quantity
    position['remaining_quantity'] = new_quantity

    # Recalculate TP/SL using shared function (supports per-pattern values)
    reason = position.get('reason', '')
    pattern_name = extract_pattern_name(reason) or None
    regime_tp_sl = position.get('regime_tp_sl')

    tp_price, sl_price, tp_pct_adjusted, _ = calculate_tp_sl(
        entry_price, direction, strategy, vol_mult, pattern_name, regime_tp_sl, config
    )

    position['tp_price'] = tp_price
    position['sl_price'] = sl_price

    # Recalculate scale-out stages (after cancelling old orders)
    scale_out_stages = setup_scale_out(
        strategy, entry_price, new_quantity, direction, tp_pct_adjusted
    )
    position['scale_out_stages'] = scale_out_stages
    position['scale_out_enabled'] = bool(scale_out_stages)

    # Update rotation fields
    position['rotation_enabled'] = False
    position['is_partial'] = False

    logger.info(f"🔧 Recalculated position: qty={new_quantity}, stages={len(scale_out_stages)}")

    # Place new orders
    try:
        position['needs_tpsl'] = True
        save_state(state)

        place_tp_sl_orders(exchange, state, config, position=position)
        if position.get('sl_order_id') or scale_out_stages:
            position['needs_tpsl'] = False
            save_state(state)
            logger.info("TP/SL orders updated after recalculation")
            return True
        else:
            logger.warning("TP/SL orders placed but SL not confirmed — needs_tpsl remains True")
            save_state(state)
            return False
    except Exception as e:
        logger.exception(f"Failed to update TP/SL orders: {e}")
        save_state(state)
        return False


def close_position_market(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    cache: APICache,
    exit_reason: str = 'MARKET',
    metrics: Optional[PerformanceMetrics] = None,
    position: Optional[Dict] = None,
) -> bool:
    """
    Close position with market order (used for early exit).

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration
        cache: APICache instance
        exit_reason: Reason for closing (e.g., 'EARLY_BD', 'EARLY_BU')
        metrics: Optional PerformanceMetrics
        position: Specific slot to close (if None, uses first slot)

    Returns:
        True if market close was successful
    """
    if position is None:
        positions = state.get('positions') or {}
        if not positions:
            logger.warning("No position to close")
            return False
        position = next(iter(positions.values()))

    symbol = config['symbol']
    direction = position.get('direction', '')
    quantity = position.get('remaining_quantity', position.get('quantity', 0))

    if quantity <= 0:
        logger.warning("Position quantity is zero")
        return False

    try:
        # Cancel existing TP/SL orders first
        cancel_remaining_orders(exchange, state, config, position=position)

        # Determine close side and positionSide
        close_side = 'sell' if direction == 'LONG' else 'buy'
        position_side = _get_position_side(config, direction)

        logger.info(f"🚨 Early exit: closing {direction} {quantity} {symbol} @ market ({exit_reason})")

        # Place market order to close
        t_close = time.time()
        order = exchange.create_order(
            symbol=symbol,
            type='market',
            side=close_side,
            amount=quantity,
            params={'positionSide': position_side}
        )

        if order:
            # Get actual fill price
            fill_price = float(order.get('average', 0)) or float(order.get('price', 0))

            if fill_price <= 0:
                # Fallback to ticker price
                ticker = fetch_ticker_cached(exchange, symbol, cache, force_refresh=True)
                fill_price = ticker['last']

            close_latency_ms = (time.time() - t_close) * 1000
            logger.info(f"✅ Early exit executed @ ${fill_price:.1f} [{close_latency_ms:.0f}ms]")

            # Record the closed position
            record_closed_position(
                exchange, state, config, fill_price, exit_reason, cache, metrics,
                position=position,
            )
            return True

    except ccxt.InsufficientFunds as e:
        logger.error(f"Insufficient funds for market close: {e}")
    except ccxt.InvalidOrder as e:
        logger.error(f"Invalid order for market close: {e}")
    except ccxt.NetworkError as e:
        logger.error(f"Network error during market close: {e}")
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error during market close: {e}")
    except Exception as e:
        logger.exception(f"Failed to close position: {e}")

    # TP/SL was cancelled but market close failed — re-place both to protect position
    try:
        logger.warning("⚠️ Re-placing TP/SL after failed market close")
        place_tp_sl_orders(exchange, state, config, position=position)
    except Exception as restore_e:
        logger.error(f"Failed to re-place TP/SL after market close failure: {restore_e}")

    return False


def _cancel_all_symbol_orders(exchange: ccxt.bingx, symbol: str) -> int:
    """Cancel all open orders for a symbol (clean slate for recovery).

    Returns number of orders cancelled.
    """
    try:
        open_orders = exchange.fetch_open_orders(symbol)
        cancelled = 0
        for order in open_orders:
            try:
                exchange.cancel_order(order['id'], symbol)
                cancelled += 1
            except (ccxt.OrderNotFound, ccxt.InvalidOrder):
                pass
            except Exception as e:
                logger.debug(f"Failed to cancel order {order.get('id')}: {e}")
        if cancelled:
            logger.info(f"🧹 Cancelled {cancelled} open orders for clean recovery")
        return cancelled
    except Exception as e:
        logger.warning(f"Failed to fetch/cancel orders for recovery: {e}")
        return 0


def _restore_none_pattern_slots(state: Dict[str, Any]) -> int:
    """Scan slots with pattern_name=None and try to restore from logs.

    v1.59.4: Prevents N/A cascade — once a recovery slot gets pattern=None,
    all subsequent closures record N/A in trade_history, which contaminates
    future recovery attempts. This function breaks the cycle by restoring
    patterns from log files (signal→open tracking).

    Called after crash recovery and can also be called periodically.

    Returns:
        Number of slots restored.
    """
    positions = state.get('positions') or {}
    none_dirs = set()
    for slot in positions.values():
        if not slot.get('pattern_name'):
            none_dirs.add(slot.get('direction'))

    if not none_dirs:
        return 0

    restored = 0
    for direction in none_dirs:
        log_patterns = _recover_patterns_from_logs(direction)
        if not log_patterns:
            continue

        # Assign patterns round-robin to None-pattern slots of this direction
        dir_none_slots = [
            s for s in positions.values()
            if s.get('direction') == direction and not s.get('pattern_name')
        ]
        for i, slot in enumerate(dir_none_slots):
            pat, vol = log_patterns[i % len(log_patterns)]
            slot['pattern_name'] = pat
            slot['vol_mult'] = vol
            old_reason = slot.get('reason', '')
            if 'Recovered from exchange' in old_reason and pat:
                slot['reason'] = f"Recovered from exchange ({pat})"
            restored += 1
            logger.info(
                f"✅ Restored pattern '{pat}' to {direction} slot {slot.get('slot_id', '?')[:8]} "
                f"(was None)"
            )

    if restored:
        logger.info(f"Pattern restoration: {restored} slot(s) updated from log analysis")
    return restored


def recover_from_crash(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    cache: APICache,
    circuit_breaker: Optional[CircuitBreaker] = None,
    metrics: Optional[PerformanceMetrics] = None,
) -> bool:
    """
    Comprehensive crash recovery - reconcile exchange with local state.

    Phase 1: Cancel all open orders (clean slate, prevents orphaned order conflicts)
    Phase 2: Ghost position detection
    Phase 3: Reconcile exchange vs local state (cases 1-3)
    Phase 4: Re-place TP/SL for all surviving slots + emergency SL

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration
        cache: APICache instance
        circuit_breaker: Optional CircuitBreaker
        metrics: Optional PerformanceMetrics

    Returns:
        True if recovery action was taken
    """
    # Import here to avoid circular dependency
    from .position_monitor import get_actual_exit_price

    symbol = config['symbol']
    logger.info("🔄 Running crash recovery check...")

    # Phase 0: Snapshot per-slot TP/SL from exchange BEFORE cancellation
    tpsl_snapshots = {}  # direction → list of (tp, sl) pairs
    for dir_label in ('LONG', 'SHORT'):
        pairs = _snapshot_all_tpsl(exchange, symbol, dir_label)
        if pairs:
            tpsl_snapshots[dir_label] = pairs

    # Phase 1: Cancel all open orders (prevents orphaned TP/SL conflicts)
    _cancel_all_symbol_orders(exchange, symbol)
    # Clear stale order IDs from all slots (orders are now cancelled)
    for slot in (state.get('positions') or {}).values():
        slot['tp_order_id'] = None
        slot['sl_order_id'] = None
    state['emergency_sl_order_id'] = None
    state['emergency_sl_orders'] = {}

    # Phase 2: Ghost position detection (log warnings, don't block)
    detect_ghost_positions(exchange, state, config, cache, circuit_breaker, metrics)

    try:
        exchange_positions = fetch_positions_cached(exchange, symbol, cache, force_refresh=True,
                                                     circuit_breaker=circuit_breaker, metrics=metrics)

        # Build exchange map by direction
        exchange_map = {}  # 'long'/'short' → exchange position dict
        for pos in exchange_positions:
            contracts = float(pos.get('contracts', 0))
            if contracts > 0:
                exchange_map[pos.get('side')] = pos

        bot_slots = state.get('positions') or {}
        recovery_needed = False

        # Phase 3: Check each direction
        for dir_label, dir_key in [('LONG', 'long'), ('SHORT', 'short')]:
            dir_slots = {sid: s for sid, s in bot_slots.items() if s.get('direction') == dir_label}
            exchange_pos = exchange_map.get(dir_key)

            # Case 1: Exchange has position, bot has no slots for this direction → orphan
            if exchange_pos and not dir_slots:
                logger.warning(f"🔧 Recovery: Found orphan {dir_label} position on exchange")
                recover_position_to_state(
                    state, config, exchange_pos, dir_label, exchange, cache,
                    saved_tpsl_pairs=tpsl_snapshots.get(dir_label),
                )
                recovery_needed = True
                continue

            # Case 2: Bot has slots, exchange has no position for this direction → closed
            if dir_slots and not exchange_pos:
                logger.warning(f"🔧 Recovery: {len(dir_slots)} {dir_label} slot(s) not found on exchange")
                for slot in list(dir_slots.values()):
                    actual_exit = get_actual_exit_price(exchange, state, config, position=slot)
                    if actual_exit:
                        record_closed_position(exchange, state, config, actual_exit['price'],
                                              actual_exit['reason'], cache, metrics, position=slot)
                    else:
                        try:
                            ticker = fetch_ticker_cached(exchange, config['symbol'], cache, force_refresh=True)
                            fallback_price = ticker['last']
                        except Exception:
                            fallback_price = slot['entry_price']
                        record_closed_position(exchange, state, config, fallback_price,
                                              'CRASH_RECOVERY', cache, metrics, position=slot)
                recovery_needed = True
                continue

            # Case 3: Both exist — check quantity mismatch
            if exchange_pos and dir_slots:
                ex_qty = float(exchange_pos.get('contracts', 0))
                local_qty_sum = sum(s.get('quantity', 0) for s in dir_slots.values())

                if abs(ex_qty - local_qty_sum) > QTY_TOLERANCE:
                    logger.warning(
                        f"🔧 Recovery: {dir_label} qty mismatch "
                        f"(exchange={ex_qty:.4f}, local_sum={local_qty_sum:.4f})"
                    )
                    if ex_qty < local_qty_sum:
                        # Some slots closed externally — remove oldest FIFO
                        sorted_slots = sorted(dir_slots.items(), key=lambda x: x[1].get('entry_time', ''))
                        remaining = local_qty_sum
                        for sid, slot in sorted_slots:
                            if abs(remaining - ex_qty) <= QTY_TOLERANCE:
                                break
                            slot_qty = slot.get('quantity', 0)
                            if remaining - slot_qty >= ex_qty - QTY_TOLERANCE:
                                actual_exit = get_actual_exit_price(exchange, state, config, position=slot)
                                if actual_exit:
                                    record_closed_position(exchange, state, config, actual_exit['price'],
                                                          actual_exit['reason'], cache, metrics, position=slot)
                                else:
                                    try:
                                        ticker = fetch_ticker_cached(exchange, config['symbol'], cache, force_refresh=True)
                                        fallback_price = ticker['last']
                                    except Exception:
                                        fallback_price = slot.get('entry_price', 0)
                                    record_closed_position(exchange, state, config, fallback_price,
                                                          'CRASH_RECOVERY', cache, metrics, position=slot)
                                remaining -= slot_qty
                            else:
                                break
                    else:
                        # Exchange has more qty — create new slot if room, else absorb
                        diff = round(ex_qty - local_qty_sum, QUANTITY_ROUND_DECIMALS)
                        max_pos = config.get('max_positions', 1)
                        available = max_pos - len(dir_slots)

                        if available > 0 and diff > 0:
                            # Use TP/SL from existing slots (same direction)
                            ref_slot = next(iter(dir_slots.values()))
                            ep = float(exchange_pos.get('entryPrice', ref_slot.get('entry_price', 0)))
                            tp_p = ref_slot.get('tp_price')
                            sl_p = ref_slot.get('sl_price')
                            if not tp_p or not sl_p:
                                dm = 1 if dir_label == 'LONG' else -1
                                ref_pat = (
                                    extract_pattern_name(ref_slot.get('reason', ''))
                                    or ref_slot.get('pattern_name')
                                )
                                tp_p, sl_p, _, _ = calculate_tp_sl(
                                    ep, dm, config['strategy'],
                                    vol_mult=1.0, pattern=ref_pat, config=config,
                                )
                            new_sid = uuid.uuid4().hex[:8]
                            ref_pattern = (
                                extract_pattern_name(ref_slot.get('reason', ''))
                                or ref_slot.get('pattern_name')
                            )
                            new_slot = {
                                'slot_id': new_sid,
                                'direction': dir_label,
                                'entry_price': ep,
                                'quantity': diff,
                                'remaining_quantity': diff,
                                'tp_price': tp_p,
                                'sl_price': sl_p,
                                'vol_mult': 1.0,
                                'scale_out_enabled': False,
                                'scale_out_stages': [],
                                'entry_time': datetime.now().isoformat(),
                                'reason': f'Recovered from exchange ({ref_pattern})' if ref_pattern else 'Recovered from exchange',
                                'pattern_name': ref_pattern,
                                'recovered': True,
                                'needs_tpsl': True,
                            }
                            state.setdefault('positions', {})[new_sid] = new_slot
                            logger.info(f"Created recovery slot {new_sid} for +{diff:.4f} excess qty")
                        else:
                            first_slot = next(iter(dir_slots.values()))
                            orig_rem = first_slot.get('remaining_quantity', first_slot['quantity'])
                            first_slot['quantity'] = round(first_slot['quantity'] + diff, QUANTITY_ROUND_DECIMALS)
                            first_slot['remaining_quantity'] = round(orig_rem + diff, QUANTITY_ROUND_DECIMALS)
                            logger.info(f"Absorbed +{diff} qty into slot {first_slot.get('slot_id')}")
                    recovery_needed = True

        # Phase 4: Re-place TP/SL for all surviving slots (orders were cancelled in Phase 1)
        remaining_slots = state.get('positions') or {}
        if remaining_slots:
            for slot in remaining_slots.values():
                if not slot.get('tp_order_id') or not slot.get('sl_order_id'):
                    place_tp_sl_orders(exchange, state, config, position=slot)
            update_emergency_sl(exchange, state, config)

        # Phase 5 (v1.59.4): Restore pattern_name for None-pattern slots
        # After recovery, some slots may have pattern=None (BingX price averaging
        # defeats price-based matching). Try log-based recovery to fill gaps.
        _restore_none_pattern_slots(state)

        save_state(state)

        if not recovery_needed:
            logger.info("✅ Crash recovery check passed - state is consistent")

        return recovery_needed

    except ccxt.NetworkError as e:
        logger.error(f"Crash recovery failed (network error): {e}")
        return False
    except ccxt.ExchangeError as e:
        logger.error(f"Crash recovery failed (exchange error): {e}")
        return False
    except Exception as e:
        logger.exception(f"Crash recovery failed: {e}")
        return False


def _update_confidence_log_outcome(
    entry_time: Optional[str],
    outcome: str,
    pnl_pct: float
) -> None:
    """
    Update the most recent confidence log entry with trade outcome.

    Reads only the last few KB of the file to find the target row,
    then rewrites only if an empty-outcome row is found.
    """
    try:
        csv_path = CONFIDENCE_LOG_FILE

        if not os.path.exists(csv_path):
            return

        outcome_value = f"{outcome}:{pnl_pct:+.2f}%"

        # Read only the tail of the file to find the last empty-outcome row
        file_size = os.path.getsize(csv_path)
        read_size = min(4096, file_size)  # Last 4KB is sufficient

        with open(csv_path, 'r+', encoding='utf-8') as f:
            # Seek to near end of file
            if file_size > read_size:
                f.seek(file_size - read_size)
                # Skip partial first line
                f.readline()
                tail_start = f.tell()
            else:
                tail_start = 0
                f.seek(0)

            tail_lines = f.readlines()

            if not tail_lines:
                return

            # Search from end for row with trailing comma
            for i in range(len(tail_lines) - 1, -1, -1):
                line = tail_lines[i].rstrip('\n').rstrip('\r')
                if line.endswith(','):
                    tail_lines[i] = line + outcome_value + '\n'
                    # Rewrite only the tail portion
                    f.seek(tail_start)
                    f.writelines(tail_lines)
                    f.truncate()
                    return

    except Exception as e:
        logger.warning(f"Failed to update confidence log outcome: {e}")