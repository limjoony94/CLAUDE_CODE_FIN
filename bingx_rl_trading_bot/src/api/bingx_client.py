"""BingX API 클라이언트 (CCXT 기반)"""

from typing import Dict, List, Optional, Any
from loguru import logger
from bingx import BingxSync

from .exceptions import (
    BingXAPIError,
    BingXAuthError,
    BingXNetworkError,
    BingXRateLimitError,
    BingXOrderError,
    BingXInsufficientBalanceError
)


class BingXClient:
    """BingX 거래소 API 클라이언트 (CCXT 기반)"""

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        testnet: bool = True,
        timeout: int = 30
    ):
        """
        Args:
            api_key: BingX API 키
            secret_key: BingX Secret 키
            testnet: 테스트넷 사용 여부
            timeout: 요청 타임아웃 (초)
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        self.timeout = timeout

        # Initialize CCXT BingX exchange
        self.exchange = BingxSync({
            'apiKey': api_key,
            'secret': secret_key,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',  # Perpetual swap
                'test': testnet,  # Enable testnet mode
                'recvWindow': 60000  # 60 seconds tolerance for timestamp (default is 5000ms)
            },
            'timeout': timeout * 1000  # ccxt uses milliseconds
        })

        # Set sandbox mode for testnet FIRST (before any API calls)
        if testnet:
            self.exchange.set_sandbox_mode(True)

        # Synchronize with BingX server time to avoid timestamp errors
        # (Must be AFTER sandbox mode is set)
        self._sync_server_time()

        logger.info(f"BingX Client (CCXT) initialized ({'Testnet' if testnet else 'Mainnet'})")

    def _sync_server_time(self):
        """Synchronize with BingX server time by overriding milliseconds generation"""
        try:
            # Use CCXT's built-in load_time_difference method
            # This makes a public API call to get server time
            self.exchange.load_time_difference()

            time_diff = self.exchange.options.get('timeDifference', 0)

            if time_diff != 0:
                logger.info(f"✅ Server time synchronized (offset: {time_diff}ms)")

                # Manually override the milliseconds method to add time difference
                # IMPORTANT: CCXT calculates offset as (local - server), so we need to
                # SUBTRACT the offset to move local time FORWARD to match server time
                original_milliseconds = self.exchange.milliseconds
                def adjusted_milliseconds():
                    return original_milliseconds() - time_diff
                self.exchange.milliseconds = adjusted_milliseconds

                logger.info(f"✅ Milliseconds method overridden (local time adjusted by {-time_diff}ms)")
            else:
                logger.warning(f"⚠️ Server time sync returned 0 offset")

        except Exception as e:
            logger.warning(f"⚠️ Could not sync server time: {e}")
            logger.warning(f"   Continuing with local time (may cause timestamp errors)")

    def _handle_ccxt_error(self, e: Exception):
        """Handle CCXT exceptions and convert to our exceptions"""
        error_msg = str(e)

        if 'authentication' in error_msg.lower() or 'unauthorized' in error_msg.lower():
            raise BingXAuthError(error_msg)
        elif 'rate limit' in error_msg.lower():
            raise BingXRateLimitError(error_msg)
        elif 'insufficient' in error_msg.lower() or 'balance' in error_msg.lower():
            raise BingXInsufficientBalanceError(error_msg)
        elif 'network' in error_msg.lower() or 'timeout' in error_msg.lower():
            raise BingXNetworkError(error_msg)
        else:
            raise BingXAPIError(error_msg)

    # ========== 시장 데이터 API ==========

    def _convert_symbol(self, symbol: str) -> str:
        """Convert BingX symbol format to CCXT format for perpetual swaps

        Args:
            symbol: BingX format (BTC-USDT) or already converted (BTC/USDT:USDT)

        Returns:
            CCXT format (BTC/USDT:USDT)
        """
        # If already in correct format, return as-is
        if ':' in symbol:
            return symbol

        # BTC-USDT -> BTC/USDT:USDT (perpetual swap format)
        base_symbol = symbol.replace('-', '/')
        return f"{base_symbol}:USDT"

    def get_klines(
        self,
        symbol: str = "BTC-USDT",
        interval: str = "5m",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        K선(캔들) 데이터 조회

        Args:
            symbol: 거래 쌍
            interval: 시간 간격 (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            limit: 반환 개수 (최대 1000)
            start_time: 시작 타임스탬프 (ms)
            end_time: 종료 타임스탬프 (ms)

        Returns:
            캔들 데이터 리스트
        """
        try:
            # Convert symbol format: BTC-USDT -> BTC/USDT:USDT
            ccxt_symbol = self._convert_symbol(symbol)

            # Fetch OHLCV data
            params = {}
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time

            # 🔍 DEBUG: Log the exact parameters being sent
            import logging as log
            log.info(f"🔍 DEBUG fetch_ohlcv: symbol={ccxt_symbol}, timeframe={interval}, limit={limit}, params={params}")

            ohlcv = self.exchange.fetch_ohlcv(
                symbol=ccxt_symbol,
                timeframe=interval,
                limit=limit,
                params=params
            )

            # Convert CCXT format to BingX format
            # CCXT: [timestamp, open, high, low, close, volume]
            # BingX: {time, open, high, low, close, volume}
            klines = []
            for candle in ohlcv:
                klines.append({
                    'time': candle[0],
                    'open': str(candle[1]),
                    'high': str(candle[2]),
                    'low': str(candle[3]),
                    'close': str(candle[4]),
                    'volume': str(candle[5])
                })

            return klines

        except Exception as e:
            self._handle_ccxt_error(e)

    def get_ticker(self, symbol: str = "BTC-USDT") -> Dict[str, Any]:
        """
        24시간 티커 정보 조회

        Args:
            symbol: 거래 쌍

        Returns:
            티커 정보
        """
        try:
            ccxt_symbol = self._convert_symbol(symbol)
            ticker = self.exchange.fetch_ticker(ccxt_symbol)

            # Convert to BingX format
            return {
                'symbol': symbol,
                'lastPrice': str(ticker['last']),
                'priceChange': str(ticker['change'] or 0),
                'priceChangePercent': str(ticker['percentage'] or 0),
                'highPrice': str(ticker['high']),
                'lowPrice': str(ticker['low']),
                'volume': str(ticker['baseVolume']),
                'quoteVolume': str(ticker['quoteVolume'])
            }

        except Exception as e:
            self._handle_ccxt_error(e)

    def get_orderbook(self, symbol: str = "BTC-USDT", limit: int = 20) -> Dict[str, Any]:
        """
        호가창 정보 조회

        Args:
            symbol: 거래 쌍
            limit: 호가 개수 (5, 10, 20, 50, 100)

        Returns:
            호가창 데이터
        """
        try:
            ccxt_symbol = self._convert_symbol(symbol)
            orderbook = self.exchange.fetch_order_book(ccxt_symbol, limit=limit)

            return {
                'bids': [[str(price), str(amount)] for price, amount in orderbook['bids']],
                'asks': [[str(price), str(amount)] for price, amount in orderbook['asks']],
                'timestamp': orderbook['timestamp']
            }

        except Exception as e:
            self._handle_ccxt_error(e)

    def get_recent_trades(self, symbol: str = "BTC-USDT", limit: int = 100) -> List[Dict[str, Any]]:
        """
        최근 체결 내역 조회

        Args:
            symbol: 거래 쌍
            limit: 반환 개수

        Returns:
            체결 내역 리스트
        """
        try:
            ccxt_symbol = self._convert_symbol(symbol)
            trades = self.exchange.fetch_trades(ccxt_symbol, limit=limit)

            return [{
                'id': str(trade['id']),
                'price': str(trade['price']),
                'qty': str(trade['amount']),
                'time': trade['timestamp'],
                'isBuyerMaker': trade['side'] == 'sell'
            } for trade in trades]

        except Exception as e:
            self._handle_ccxt_error(e)

    # ========== 계정 API (인증 필요) ==========

    def get_balance(self) -> Dict[str, Any]:
        """
        계정 잔고 조회

        Returns:
            잔고 정보 (BingX API의 실제 equity 포함)
        """
        try:
            balance = self.exchange.fetch_balance()

            # Testnet uses VST (Virtual Stable Token), mainnet uses USDT
            asset_key = 'VST' if self.testnet else 'USDT'

            # Get BingX-specific data from info field
            info = balance.get('info', {})
            data = info.get('data', {})

            # BingX provides both 'balance' (realized) and 'equity' (realized + unrealized)
            bingx_balance = data.get('balance', {}) if isinstance(data, dict) else {}

            # ✅ FIXED 2025-10-25: Return both wallet balance and equity
            # - balance: Wallet balance (realized P&L only)
            # - equity: Total equity (realized + unrealized P&L)
            # - unrealizedProfit: Unrealized P&L from positions
            return {
                'balance': {
                    'asset': 'USDT',  # Always report as USDT for compatibility
                    'balance': str(bingx_balance.get('balance', balance[asset_key]['total']) if bingx_balance else balance[asset_key]['total']),
                    'equity': str(bingx_balance.get('equity', balance[asset_key]['total']) if bingx_balance else balance[asset_key]['total']),
                    'unrealizedProfit': str(bingx_balance.get('unrealizedProfit', 0) if bingx_balance else 0),
                    'availableMargin': str(bingx_balance.get('availableMargin', balance[asset_key]['free']) if bingx_balance else balance[asset_key]['free'])
                }
            }

        except Exception as e:
            self._handle_ccxt_error(e)

    def get_positions(self, symbol: str = "BTC-USDT") -> List[Dict[str, Any]]:
        """
        현재 포지션 조회

        Args:
            symbol: 거래 쌍

        Returns:
            포지션 리스트
        """
        try:
            ccxt_symbol = self._convert_symbol(symbol)
            positions = self.exchange.fetch_positions([ccxt_symbol])

            # Convert to BingX format
            result = []
            for pos in positions:
                if pos['contracts'] != 0:  # Only return positions with non-zero size
                    result.append({
                        'symbol': symbol,
                        'positionSide': 'LONG' if pos['side'] == 'long' else 'SHORT',
                        'positionAmt': str(pos['contracts']),
                        'entryPrice': str(pos['entryPrice'] or 0),
                        'unrealizedProfit': str(pos['unrealizedPnl'] or 0),
                        'leverage': str(pos['leverage'] or 1)
                    })

            return result

        except Exception as e:
            self._handle_ccxt_error(e)

    def get_position_close_details(self, position_id: str = None, order_id: str = None, symbol: str = "BTC-USDT") -> Optional[Dict[str, Any]]:
        """
        청산된 포지션의 정확한 exit price와 P&L 조회 (Position History API)

        Args:
            position_id: 포지션 ID (우선순위 1 - RECOMMENDED)
            order_id: 주문 ID (우선순위 2 - fallback, but won't match)
            symbol: 거래 쌍

        Returns:
            {
                'exit_price': float,      # 실제 청산가
                'realized_pnl': float,    # 실현 손익 (수수료 포함)
                'net_profit': float,      # 순손익 (수수료 제외)
                'close_time': int,        # 청산 시간 (timestamp)
                'quantity': float         # 청산된 수량
            }

        Note:
            - 최근 1개월 이내 청산된 포지션만 조회 가능
            - Position ID로 매칭 (RECOMMENDED)
            - Order ID는 Position History API에 없어서 매칭 불가
        """
        try:
            import time
            ccxt_symbol = self._convert_symbol(symbol)

            # Validate input
            if not position_id and not order_id:
                logger.warning(f"⚠️ get_position_close_details called without position_id or order_id")
                return None

            # Calculate timestamp range (최근 1개월)
            end_ts = int(time.time() * 1000)  # Current time in ms
            start_ts = end_ts - (30 * 24 * 60 * 60 * 1000)  # 30 days ago

            # Position History 조회 with required timestamps
            history = self.exchange.fetch_position_history(
                symbol=ccxt_symbol,
                since=start_ts,
                limit=100,
                params={
                    'startTs': start_ts,
                    'endTs': end_ts
                }
            )

            # Match by Position ID (PRIORITY 1 - Reliable)
            if position_id:
                for pos in history:
                    pos_id = str(pos.get('id', ''))

                    if pos_id == str(position_id):
                        logger.info(f"✅ Found closed position in history: {position_id}")

                        # CCXT normalize된 데이터에서 추출
                        return {
                            'exit_price': float(pos.get('info', {}).get('avgClosePrice', 0)),
                            'realized_pnl': float(pos.get('info', {}).get('realisedProfit', 0)),
                            'net_profit': float(pos.get('info', {}).get('netProfit', 0)),
                            'close_time': int(pos.get('timestamp', 0)),
                            'quantity': abs(float(pos.get('info', {}).get('closePositionAmt', 0)))
                        }

                logger.warning(f"⚠️ Position {position_id} not found in history (may be >1 month old)")
                return None

            # Fallback to Order ID (PRIORITY 2 - Won't work, but try anyway)
            elif order_id:
                logger.warning(f"⚠️ Using order_id ({order_id}) - Position History API doesn't have orderId field!")
                logger.warning(f"   This will likely fail. Please provide position_id instead.")

                # Try to match (will fail, but log the attempt)
                for pos in history:
                    pos_id = str(pos.get('id', ''))

                    if pos_id == str(order_id):
                        logger.info(f"✅ Found closed position in history: {order_id}")

                        return {
                            'exit_price': float(pos.get('info', {}).get('avgClosePrice', 0)),
                            'realized_pnl': float(pos.get('info', {}).get('realisedProfit', 0)),
                            'net_profit': float(pos.get('info', {}).get('netProfit', 0)),
                            'close_time': int(pos.get('timestamp', 0)),
                            'quantity': abs(float(pos.get('info', {}).get('closePositionAmt', 0)))
                        }

                logger.warning(f"⚠️ Position {order_id} not found in history (expected - order_id ≠ position_id)")
                return None

        except Exception as e:
            logger.warning(f"⚠️ Could not fetch position history: {e}")
            return None

    def set_leverage(self, symbol: str, side: str, leverage: int) -> Dict[str, Any]:
        """
        레버리지 설정

        Args:
            symbol: 거래 쌍
            side: 포지션 방향 (LONG, SHORT)
            leverage: 레버리지 배수

        Returns:
            설정 결과
        """
        try:
            ccxt_symbol = self._convert_symbol(symbol)

            # CCXT set_leverage
            result = self.exchange.set_leverage(
                leverage=leverage,
                symbol=ccxt_symbol,
                params={'side': side}
            )

            logger.info(f"Leverage set to {leverage}x for {side} {symbol}")
            return result

        except Exception as e:
            self._handle_ccxt_error(e)

    # ========== 주문 API (인증 필요) ==========

    def create_order(
        self,
        symbol: str,
        side: str,
        position_side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC"
    ) -> Dict[str, Any]:
        """
        주문 생성

        Args:
            symbol: 거래 쌍
            side: 주문 방향 (BUY, SELL)
            position_side: 포지션 방향 (LONG, SHORT)
            order_type: 주문 타입 (MARKET, LIMIT, STOP, STOP_MARKET, TAKE_PROFIT, TAKE_PROFIT_MARKET)
            quantity: 수량
            price: 지정가 가격 (LIMIT 주문 시 필수)
            stop_price: 스탑 가격 (STOP 주문 시 필수)
            time_in_force: 주문 유효 기간 (GTC, IOC, FOK)

        Returns:
            주문 결과
        """
        try:
            ccxt_symbol = self._convert_symbol(symbol)

            # Prepare order parameters
            params = {
                'positionSide': position_side,
                'timeInForce': time_in_force
            }

            if stop_price:
                params['stopPrice'] = stop_price

            # 🔍 DEBUG: Log parameters being sent to BingX
            logger.debug(f"create_order called with: side={side}, position_side={position_side}, params={params}")

            # Create order using CCXT
            result = self.exchange.create_order(
                symbol=ccxt_symbol,
                type=order_type.lower(),
                side=side.lower(),
                amount=quantity,
                price=price,
                params=params
            )

            logger.info(f"Order created: {side} {quantity} {symbol} @ {price or 'MARKET'}")
            return result

        except Exception as e:
            error_msg = str(e)
            if 'insufficient' in error_msg.lower() or 'balance' in error_msg.lower():
                logger.error("Insufficient balance for order")
                raise BingXInsufficientBalanceError(error_msg)
            else:
                logger.error(f"Order creation failed: {error_msg}")
                self._handle_ccxt_error(e)

    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        주문 취소

        Args:
            symbol: 거래 쌍
            order_id: 주문 ID

        Returns:
            취소 결과
        """
        try:
            ccxt_symbol = self._convert_symbol(symbol)
            result = self.exchange.cancel_order(order_id, ccxt_symbol)
            logger.info(f"Order cancelled: {order_id}")
            return result

        except Exception as e:
            self._handle_ccxt_error(e)

    def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """
        모든 주문 취소

        Args:
            symbol: 거래 쌍

        Returns:
            취소 결과
        """
        try:
            ccxt_symbol = self._convert_symbol(symbol)
            result = self.exchange.cancel_all_orders(ccxt_symbol)
            logger.info(f"All orders cancelled for {symbol}")
            return result

        except Exception as e:
            self._handle_ccxt_error(e)

    def get_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        주문 조회

        Args:
            symbol: 거래 쌍
            order_id: 주문 ID

        Returns:
            주문 정보
        """
        try:
            ccxt_symbol = self._convert_symbol(symbol)
            return self.exchange.fetch_order(order_id, ccxt_symbol)

        except Exception as e:
            self._handle_ccxt_error(e)

    def get_open_orders(self, symbol: str = "BTC-USDT") -> List[Dict[str, Any]]:
        """
        미체결 주문 조회

        Args:
            symbol: 거래 쌍

        Returns:
            미체결 주문 리스트
        """
        try:
            ccxt_symbol = self._convert_symbol(symbol)
            return self.exchange.fetch_open_orders(ccxt_symbol)

        except Exception as e:
            self._handle_ccxt_error(e)

    def close_position(
        self,
        symbol: str,
        position_side: str,
        quantity: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        포지션 청산

        Args:
            symbol: 거래 쌍
            position_side: 포지션 방향 (LONG, SHORT) - for matching position
            quantity: 청산 수량 (None이면 전체 청산)

        Returns:
            청산 결과
        """
        # 현재 포지션 조회
        positions = self.get_positions(symbol)

        for pos in positions:
            if pos['positionSide'] == position_side:
                pos_qty = abs(float(pos['positionAmt']))
                close_qty = quantity if quantity else pos_qty

                # 청산 주문 방향 결정
                side = 'SELL' if position_side == 'LONG' else 'BUY'

                # 시장가 청산 주문
                # Note: In One-Way mode, positionSide must be "BOTH" for closing orders
                return self.create_order(
                    symbol=symbol,
                    side=side,
                    position_side='BOTH',  # ✅ Use "BOTH" for One-Way mode closing
                    order_type='MARKET',
                    quantity=close_qty
                )

        logger.warning(f"No {position_side} position found for {symbol}")
        return {}

    def enter_position_with_protection(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        leverage: int = 4,
        balance_sl_pct: float = 0.06,
        current_balance: float = None,
        position_size_pct: float = None
    ) -> Dict[str, Any]:
        """
        포지션 진입 + 동시에 Stop Loss 보호 주문 설정 (Balance-Based SL)

        Args:
            symbol: 거래 쌍 (e.g., "BTC-USDT")
            side: 포지션 방향 ("LONG" or "SHORT")
            quantity: 진입 수량 (BTC)
            entry_price: 진입 가격 (참고용, 실제는 market order)
            leverage: 레버리지 배수 (default: 4x)
            balance_sl_pct: Total balance 기준 Stop Loss (default: 0.06 = -6%)
            current_balance: 현재 잔액 (optional, for logging)
            position_size_pct: 포지션 크기 비율 (required for balance-based SL)

        Returns:
            Dict containing:
                - entry_order: 진입 주문 결과
                - stop_loss_order: Stop Loss 주문 결과
                - stop_loss_price: Stop Loss 트리거 가격
                - price_sl_pct: 가격 기준 SL 비율

        Note:
            - Balance-Based SL: Consistent risk across all position sizes
            - Formula: price_sl_pct = balance_sl_pct / (position_size_pct × leverage)
            - Example (50% position, 4x leverage, -6% balance SL):
              price_sl_pct = 0.06 / (0.50 × 4) = 0.03 = 3% price change
            - Smaller positions get wider SL, larger positions get tighter SL
            - Take Profit은 ML Exit Model이 처리 (고정 TP 없음)
        """
        try:
            # Calculate price-based SL from balance-based SL
            # Formula: price_sl_pct = balance_sl_pct / (position_size_pct × leverage)
            # Example (50% position, 4x leverage): 0.06 / (0.50 × 4) = 0.03 = 3%
            if position_size_pct is None or position_size_pct <= 0:
                raise ValueError("position_size_pct is required for balance-based SL calculation")

            price_sl_pct = abs(balance_sl_pct) / (position_size_pct * leverage)
            logger.info(f"🛡️ Balance-Based SL Calculation:")
            logger.info(f"   Target: {abs(balance_sl_pct)*100:.1f}% total balance loss")
            logger.info(f"   Position Size: {position_size_pct*100:.1f}%")
            logger.info(f"   Leverage: {leverage}x")
            logger.info(f"   → Price SL: {price_sl_pct*100:.2f}%")

            logger.info(f"🛡️ Entering {side} position with Stop Loss protection:")
            logger.info(f"   Quantity: {quantity:.6f} BTC")
            logger.info(f"   Stop Loss: {price_sl_pct*100:.2f}% (Exchange-Level)")
            logger.info(f"   Exit Strategy: ML Exit Model + Max Hold")

            # 1. 포지션 진입 (Market Order)
            order_side = "BUY" if side == "LONG" else "SELL"
            entry_order = self.create_order(
                symbol=symbol,
                side=order_side,
                position_side="BOTH",  # One-Way mode
                order_type="MARKET",
                quantity=quantity
            )

            logger.info(f"✅ Entry order executed: {entry_order.get('id', 'N/A')}")

            # Get actual entry price from order result (or use fallback)
            actual_entry_price = entry_order.get('price', entry_price)

            # 2. Stop Loss 주문 (STOP_MARKET - 긴급 손절)
            if side == "LONG":
                stop_loss_price = actual_entry_price * (1 - price_sl_pct)
                stop_order_side = "SELL"
            else:  # SHORT
                stop_loss_price = actual_entry_price * (1 + price_sl_pct)
                stop_order_side = "BUY"

            stop_loss_order = self.create_order(
                symbol=symbol,
                side=stop_order_side,
                position_side="BOTH",
                order_type="STOP_MARKET",
                quantity=quantity,
                stop_price=stop_loss_price
            )

            logger.info(f"✅ Stop Loss set: ${stop_loss_price:,.2f} (order: {stop_loss_order.get('id', 'N/A')})")
            logger.info(f"🛡️ Exchange-level protection active (24/7 monitoring)")

            # 3. Fetch Position ID from exchange (CRITICAL for reconciliation)
            # Wait briefly for position to be created
            import time
            time.sleep(0.5)

            position_id = None
            try:
                ccxt_symbol = self._convert_symbol(symbol)
                positions = self.exchange.fetch_positions([ccxt_symbol])

                # Match by quantity (most reliable identifier immediately after entry)
                for pos in positions:
                    pos_contracts = abs(float(pos.get('contracts', 0)))
                    if abs(pos_contracts - quantity) < 0.0001:  # Tolerance for floating point
                        position_id = pos.get('id')
                        logger.info(f"✅ Position ID captured: {position_id}")
                        break

                if not position_id:
                    logger.warning(f"⚠️ Could not capture position_id (will use order_id as fallback)")

            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch position_id: {e}")

            return {
                'entry_order': entry_order,
                'stop_loss_order': stop_loss_order,
                'stop_loss_price': stop_loss_price,
                'price_sl_pct': price_sl_pct,
                'position_id': position_id  # NEW: Critical for reconciliation
            }

        except Exception as e:
            logger.error(f"❌ Failed to enter position with protection: {e}")
            # If entry succeeded but protection orders failed, try to close position
            if 'entry_order' in locals():
                logger.warning(f"⚠️ Entry succeeded but protection failed - attempting emergency close")
                try:
                    self.close_position(symbol=symbol, position_side=side, quantity=quantity)
                    logger.info(f"✅ Emergency close successful")
                except Exception as close_error:
                    logger.error(f"❌ Emergency close failed: {close_error}")
                    logger.error(f"⚠️ MANUAL INTERVENTION REQUIRED - Position open without protection!")
            raise

    def cancel_position_orders(
        self,
        symbol: str,
        order_ids: List[str]
    ) -> Dict[str, Any]:
        """
        포지션 관련 미체결 주문 취소 (Stop Loss, Take Profit 등)

        Args:
            symbol: 거래 쌍
            order_ids: 취소할 주문 ID 리스트

        Returns:
            Dict containing:
                - cancelled: 성공적으로 취소된 주문 ID 리스트
                - failed: 취소 실패한 주문 ID 리스트
        """
        cancelled = []
        failed = []

        for order_id in order_ids:
            try:
                self.cancel_order(symbol=symbol, order_id=order_id)
                cancelled.append(order_id)
                logger.info(f"✅ Cancelled order: {order_id}")
            except Exception as e:
                failed.append(order_id)
                logger.warning(f"⚠️ Failed to cancel order {order_id}: {e}")

        logger.info(f"📊 Order cancellation: {len(cancelled)} succeeded, {len(failed)} failed")

        return {
            'cancelled': cancelled,
            'failed': failed
        }

    # ========== 유틸리티 메서드 ==========

    def ping(self) -> bool:
        """
        API 연결 테스트

        Returns:
            연결 성공 여부
        """
        try:
            self.exchange.fetch_time()
            logger.info("Ping successful")
            return True
        except Exception as e:
            logger.error(f"Ping failed: {str(e)}")
            return False

    def get_exchange_info(self, symbol: str = "BTC-USDT") -> Dict[str, Any]:
        """
        거래소 정보 조회

        Args:
            symbol: 거래 쌍

        Returns:
            거래소 정보
        """
        try:
            ccxt_symbol = self._convert_symbol(symbol)
            markets = self.exchange.fetch_markets()

            # Find matching market
            for market in markets:
                if market['symbol'] == ccxt_symbol:
                    return {
                        'symbol': symbol,
                        'contractSize': market.get('contractSize', 1),
                        'pricePrecision': market.get('precision', {}).get('price', 2),
                        'quantityPrecision': market.get('precision', {}).get('amount', 3)
                    }

            return {}

        except Exception as e:
            self._handle_ccxt_error(e)
