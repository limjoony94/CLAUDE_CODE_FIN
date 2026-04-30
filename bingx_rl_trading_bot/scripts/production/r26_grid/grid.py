"""R26 Grid Bot — Order management & state.

Maintains:
- Active grid (init_mid, levels, fills)
- Open positions (entry, TP order id, side)
- State persistence to JSON

Wraps CCXT operations: place limit, place TP after fill, force close all.
"""
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger('r26_grid.grid')


@dataclass
class GridLevel:
    """One grid level (buy or sell limit)."""
    side: str           # 'buy' or 'sell'
    level_idx: int       # 0..levels_each_side-1
    price: float          # planned limit price
    notional_usd: float
    order_id: Optional[str] = None    # exchange order id once placed
    filled: bool = False
    fill_price: Optional[float] = None
    fill_ts_ms: Optional[int] = None


@dataclass
class OpenPosition:
    """Position opened from a grid fill."""
    side: str           # 'long' or 'short'
    entry_price: float
    notional_usd: float
    qty_btc: float
    grid_level_idx: int  # which level filled
    grid_level_side: str  # 'buy' or 'sell' on grid
    tp_price: float
    tp_order_id: Optional[str] = None
    open_ts_ms: int = 0


@dataclass
class GridState:
    """Active grid + open positions."""
    active: bool = False
    init_mid: float = 0.0
    init_ts_ms: int = 0
    init_idx: int = 0      # bar index when grid setup
    buy_levels: list = field(default_factory=list)    # list[GridLevel]
    sell_levels: list = field(default_factory=list)   # list[GridLevel]
    open_positions: list = field(default_factory=list)  # list[OpenPosition]
    last_update_ts_ms: int = 0


class StateManager:
    """Persist GridState to disk."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> GridState:
        if not self.path.exists():
            return GridState()
        try:
            with open(self.path, 'r') as f:
                data = json.load(f)
            state = GridState(
                active=data.get('active', False),
                init_mid=data.get('init_mid', 0.0),
                init_ts_ms=data.get('init_ts_ms', 0),
                init_idx=data.get('init_idx', 0),
                buy_levels=[GridLevel(**lv) for lv in data.get('buy_levels', [])],
                sell_levels=[GridLevel(**lv) for lv in data.get('sell_levels', [])],
                open_positions=[OpenPosition(**p) for p in data.get('open_positions', [])],
                last_update_ts_ms=data.get('last_update_ts_ms', 0),
            )
            logger.info(f"State loaded: active={state.active}, "
                        f"open_positions={len(state.open_positions)}")
            return state
        except Exception as e:
            logger.error(f"State load failed: {e}. Starting clean.")
            return GridState()

    def save(self, state: GridState):
        """Atomic save with OneDrive-aware retry (C1 BUG#58 precedent).

        OneDrive can lock files during sync, causing tmp.replace to fail with
        WinError 5. Retries with backoff; falls back to direct write if all fail.
        """
        state.last_update_ts_ms = int(time.time() * 1000)
        data = {
            'active': state.active,
            'init_mid': state.init_mid,
            'init_ts_ms': state.init_ts_ms,
            'init_idx': state.init_idx,
            'buy_levels': [asdict(lv) for lv in state.buy_levels],
            'sell_levels': [asdict(lv) for lv in state.sell_levels],
            'open_positions': [asdict(p) for p in state.open_positions],
            'last_update_ts_ms': state.last_update_ts_ms,
        }
        # Try atomic save with retries (OneDrive lock recovery)
        for attempt, delay in enumerate([0, 0.5, 1.0, 2.0]):
            if delay > 0:
                time.sleep(delay)
            try:
                tmp = self.path.with_suffix('.tmp')
                with open(tmp, 'w') as f:
                    json.dump(data, f, indent=2)
                tmp.replace(self.path)
                if attempt > 0:
                    logger.debug(f"State save succeeded on retry {attempt}")
                return
            except (PermissionError, OSError) as e:
                if attempt < 3:
                    continue
                logger.warning(f"State atomic save failed after retries: {e}; "
                                f"falling back to direct write")
        # Direct write fallback (less atomic, but functional)
        try:
            with open(self.path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"State direct write also failed: {e}")


class GridManager:
    """Manage grid lifecycle: setup, fill detection, TP placement, force close."""

    def __init__(self, exchange, symbol: str, state: GridState,
                  state_manager: StateManager, spacing_pct: float,
                  notional_callback=None):
        """
        notional_callback: optional function returning current per_level_notional.
            If set, called on each TP fill to recompute grid size from latest balance
            (per-TP compound). Otherwise grid uses fixed initial notional.
        """
        self.ex = exchange
        self.symbol = symbol
        self.state = state
        self.sm = state_manager
        self.spacing = spacing_pct / 100
        self.notional_callback = notional_callback

    # -------- Grid setup --------
    def setup_grid(self, init_mid: float, ts_ms: int, levels_each_side: int,
                    per_level_notional_usd: float):
        """Initialize grid: compute levels, place limit orders."""
        from .indicators import compute_grid_levels
        buys, sells = compute_grid_levels(init_mid, self.spacing * 100, levels_each_side)

        self.state.active = True
        self.state.init_mid = init_mid
        self.state.init_ts_ms = ts_ms
        self.state.buy_levels = []
        self.state.sell_levels = []
        self.state.open_positions = []

        # Compute qty from notional
        for i, p in enumerate(buys):
            qty = per_level_notional_usd / p
            level = GridLevel(side='buy', level_idx=i, price=p,
                              notional_usd=per_level_notional_usd)
            try:
                order = self.ex.create_limit_buy_order(
                    self.symbol, qty, p,
                    {'positionSide': 'BOTH'}
                )
                level.order_id = order.get('id')
                logger.info(f"Buy LIMIT placed: lv={i}, price={p:.2f}, qty={qty:.6f}, "
                             f"order_id={level.order_id}")
            except Exception as e:
                logger.error(f"Buy LIMIT placement failed: lv={i}, p={p}: {e}")
            self.state.buy_levels.append(level)

        for i, p in enumerate(sells):
            qty = per_level_notional_usd / p
            level = GridLevel(side='sell', level_idx=i, price=p,
                              notional_usd=per_level_notional_usd)
            try:
                order = self.ex.create_limit_sell_order(
                    self.symbol, qty, p,
                    {'positionSide': 'BOTH'}
                )
                level.order_id = order.get('id')
                logger.info(f"Sell LIMIT placed: lv={i}, price={p:.2f}, qty={qty:.6f}, "
                             f"order_id={level.order_id}")
            except Exception as e:
                logger.error(f"Sell LIMIT placement failed: lv={i}, p={p}: {e}")
            self.state.sell_levels.append(level)

        self.sm.save(self.state)
        logger.info(f"Grid initialized at mid={init_mid:.2f} with "
                     f"{len(self.state.buy_levels)} buys + "
                     f"{len(self.state.sell_levels)} sells")

    # -------- Fill detection --------
    def check_fills(self, mark_price: float):
        """Poll exchange for filled grid orders. On fill: open position + place TP limit."""
        for level in (self.state.buy_levels + self.state.sell_levels):
            if level.filled or level.order_id is None:
                continue
            try:
                order = self.ex.fetch_order(level.order_id, self.symbol)
                status = order.get('status')
                if status == 'closed':  # fully filled
                    level.filled = True
                    level.fill_price = float(order.get('average') or order.get('price') or level.price)
                    level.fill_ts_ms = int(order.get('timestamp') or time.time() * 1000)
                    self._on_fill(level)
                elif status == 'canceled':
                    logger.warning(f"Grid order {level.order_id} canceled externally; "
                                    f"marking unfilled, will not retry")
                    level.order_id = None  # skip in future polls
            except Exception as e:
                logger.warning(f"fetch_order {level.order_id} failed: {e}")

        self.sm.save(self.state)

    def _on_fill(self, level: GridLevel):
        """Grid level filled → open position + place TP limit on opposite side.

        BT-LIVE parity: TP price computed from level.price (the limit), not fill_price.
        For LIMIT orders these should be equal, but explicit level.price ensures
        exact BT match even if exchange records partial-fill price differences.
        """
        # Position side: buy fill = LONG, sell fill = SHORT
        pos_side = 'long' if level.side == 'buy' else 'short'
        # TP price computed from grid level price (BT parity)
        if pos_side == 'long':
            tp_price = level.price * (1 + self.spacing)
        else:
            tp_price = level.price * (1 - self.spacing)

        # Quantity from notional / level.price (BT parity — entry price = level price)
        qty_btc = level.notional_usd / level.price

        # Place TP limit
        tp_order_id = None
        try:
            if pos_side == 'long':
                # Close LONG = SELL limit at tp_price (reduceOnly)
                order = self.ex.create_limit_sell_order(
                    self.symbol, qty_btc, tp_price,
                    {'positionSide': 'BOTH', 'reduceOnly': True}
                )
            else:
                order = self.ex.create_limit_buy_order(
                    self.symbol, qty_btc, tp_price,
                    {'positionSide': 'BOTH', 'reduceOnly': True}
                )
            tp_order_id = order.get('id')
            logger.info(f"TP placed: side={pos_side}, qty={qty_btc:.6f}, "
                         f"entry={level.fill_price:.2f}, tp={tp_price:.2f}, "
                         f"tp_order_id={tp_order_id}")
        except Exception as e:
            logger.error(f"TP placement failed: {e}")

        pos = OpenPosition(
            side=pos_side, entry_price=level.price,  # use level.price (BT parity)
            notional_usd=level.notional_usd, qty_btc=qty_btc,
            grid_level_idx=level.level_idx, grid_level_side=level.side,
            tp_price=tp_price, tp_order_id=tp_order_id,
            open_ts_ms=level.fill_ts_ms,
        )
        self.state.open_positions.append(pos)

    # -------- TP fill detection --------
    def check_tp_fills(self):
        """Poll TP orders for fills → remove from open positions, place new grid level."""
        remaining = []
        for pos in self.state.open_positions:
            if pos.tp_order_id is None:
                remaining.append(pos)
                continue
            try:
                order = self.ex.fetch_order(pos.tp_order_id, self.symbol)
                status = order.get('status')
                if status == 'closed':
                    fill_price = float(order.get('average') or order.get('price') or pos.tp_price)
                    realized_pnl_pct = (fill_price - pos.entry_price) / pos.entry_price * 100
                    if pos.side == 'short':
                        realized_pnl_pct = -realized_pnl_pct
                    logger.info(f"TP filled: side={pos.side}, entry={pos.entry_price:.2f}, "
                                 f"exit={fill_price:.2f}, pnl={realized_pnl_pct:+.4f}%")
                    # Re-place the grid level (reset to pre-fill state)
                    self._replace_grid_level(pos)
                    # Don't add back to open_positions
                else:
                    remaining.append(pos)
            except Exception as e:
                logger.warning(f"TP fetch_order {pos.tp_order_id} failed: {e}")
                remaining.append(pos)
        self.state.open_positions = remaining
        self.sm.save(self.state)

    def _replace_grid_level(self, pos: OpenPosition):
        """After TP fill, restore the grid level for next cycle.

        If notional_callback is set, recomputes per-level notional from current
        balance (per-TP compound). Otherwise uses original level notional.
        """
        levels = self.state.buy_levels if pos.grid_level_side == 'buy' else self.state.sell_levels
        if pos.grid_level_idx >= len(levels):
            return
        level = levels[pos.grid_level_idx]

        # Per-TP compound: recompute notional from latest balance.
        # On API failure, KEEP existing level.notional_usd (consistency with other levels)
        # rather than falling back to fixed config (which would mismatch).
        if self.notional_callback is not None:
            try:
                new_notional = self.notional_callback()
                if new_notional > 0:
                    old = level.notional_usd
                    level.notional_usd = new_notional
                    if abs(new_notional - old) > 0.01:
                        logger.info(f"Compound update lv {pos.grid_level_idx}: "
                                     f"${old:.2f} → ${new_notional:.2f}")
                else:
                    logger.warning(f"Compound recompute returned non-positive value; "
                                    f"keeping existing notional ${level.notional_usd:.2f}")
            except Exception as e:
                logger.warning(f"Notional recompute failed: {e}; "
                                f"keeping existing notional ${level.notional_usd:.2f}")

        try:
            qty = level.notional_usd / level.price
            if level.side == 'buy':
                order = self.ex.create_limit_buy_order(
                    self.symbol, qty, level.price,
                    {'positionSide': 'BOTH'}
                )
            else:
                order = self.ex.create_limit_sell_order(
                    self.symbol, qty, level.price,
                    {'positionSide': 'BOTH'}
                )
            level.order_id = order.get('id')
            level.filled = False
            level.fill_price = None
            level.fill_ts_ms = None
            logger.info(f"Grid level re-placed: side={level.side}, lv={level.level_idx}, "
                         f"price={level.price:.2f}, qty={qty:.6f}, "
                         f"notional=${level.notional_usd:.2f}, order_id={level.order_id}")
        except Exception as e:
            logger.error(f"Grid level replacement failed: {e}")

    # -------- Force close all (trend exit, halt) --------
    def force_close_all(self, reason: str = 'TREND_EXIT'):
        """Cancel all open orders + market close all open positions (taker)."""
        # Cancel all grid orders (both filled-tracking but not-yet-tp and unfilled)
        for level in (self.state.buy_levels + self.state.sell_levels):
            if level.order_id and not level.filled:
                try:
                    self.ex.cancel_order(level.order_id, self.symbol)
                    logger.info(f"Cancelled grid order: {level.order_id} ({level.side} lv {level.level_idx})")
                except Exception as e:
                    logger.warning(f"Cancel order {level.order_id} failed: {e}")
                level.order_id = None

        # Cancel TP orders + market close positions
        for pos in self.state.open_positions:
            if pos.tp_order_id:
                try:
                    self.ex.cancel_order(pos.tp_order_id, self.symbol)
                except Exception as e:
                    logger.warning(f"Cancel TP {pos.tp_order_id} failed: {e}")

            # Market close
            try:
                if pos.side == 'long':
                    order = self.ex.create_market_sell_order(
                        self.symbol, pos.qty_btc,
                        {'positionSide': 'BOTH', 'reduceOnly': True}
                    )
                else:
                    order = self.ex.create_market_buy_order(
                        self.symbol, pos.qty_btc,
                        {'positionSide': 'BOTH', 'reduceOnly': True}
                    )
                logger.info(f"Force close ({reason}): side={pos.side}, qty={pos.qty_btc:.6f}, "
                             f"order_id={order.get('id')}")
            except Exception as e:
                logger.error(f"Force close failed for pos: {e}")

        # Reset grid state
        self.state.active = False
        self.state.init_mid = 0
        self.state.buy_levels = []
        self.state.sell_levels = []
        self.state.open_positions = []
        self.sm.save(self.state)
        logger.info(f"Grid force-closed and reset (reason: {reason})")
