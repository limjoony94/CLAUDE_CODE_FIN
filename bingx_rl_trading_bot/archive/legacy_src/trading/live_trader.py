"""실시간 거래 실행"""

import time
from typing import Dict, Any

import numpy as np
from loguru import logger

from ..api.bingx_client import BingXClient
from ..api.exceptions import BingXAPIError
from ..data.data_collector import DataCollector
from ..indicators.technical_indicators import TechnicalIndicators
from ..risk.risk_manager import RiskManager


class LiveTrader:
    """
    실시간 거래 실행 엔진
    """

    def __init__(
        self,
        client: BingXClient,
        agent,
        config: Dict[str, Any],
        dry_run: bool = True
    ):
        """
        Args:
            client: BingX API 클라이언트
            agent: 강화학습 에이전트
            config: 설정
            dry_run: 모의 거래 모드
        """
        self.client = client
        self.agent = agent
        self.config = config
        self.dry_run = dry_run

        # 모듈 초기화
        self.data_collector = DataCollector(
            client=client,
            symbol=config['exchange']['symbol'],
            interval=config['trading']['timeframe']
        )

        self.indicator_calculator = TechnicalIndicators(config['indicators'])
        self.risk_manager = RiskManager(config['risk'])

        # 거래 상태
        self.is_running = False
        self.current_position = 0.0
        self.balance = 0.0

        logger.info(f"Live Trader initialized ({'DRY RUN' if dry_run else 'LIVE'} mode)")

    def start(self) -> None:
        """거래 시작"""
        logger.info("Starting live trading...")

        # API 연결 테스트
        if not self.client.ping():
            logger.error("Failed to connect to exchange")
            return

        # 레버리지 설정
        leverage = self.config['trading']['leverage']
        symbol = self.config['exchange']['symbol']

        try:
            self.client.set_leverage(symbol, 'LONG', leverage)
            self.client.set_leverage(symbol, 'SHORT', leverage)
            logger.info(f"Leverage set to {leverage}x")
        except BingXAPIError as e:
            logger.error(f"Failed to set leverage: {e.message}")
            return

        # 초기 잔고 조회
        try:
            balance_info = self.client.get_balance()
            self.balance = float(balance_info['balance']['balance'])
            self.risk_manager.initialize(self.balance)
            logger.info(f"Initial balance: {self.balance:.2f} USDT")
        except BingXAPIError as e:
            logger.error(f"Failed to get balance: {e.message}")
            return

        # 거래 루프 시작
        self.is_running = True
        self._trading_loop()

    def stop(self) -> None:
        """거래 중지"""
        logger.info("Stopping live trading...")
        self.is_running = False

        # 열린 포지션 청산 (선택)
        if not self.dry_run:
            self._close_all_positions()

    def _trading_loop(self) -> None:
        """거래 루프"""
        symbol = self.config['exchange']['symbol']
        interval_seconds = 300  # 5분

        while self.is_running:
            try:
                # 1. 최신 데이터 수집
                df = self.data_collector.collect_recent_data(limit=100)

                if df.empty:
                    logger.warning("No data received, skipping...")
                    time.sleep(10)
                    continue

                # 2. 기술적 지표 계산
                df = self.indicator_calculator.calculate_all_indicators(df)

                if df.empty:
                    logger.warning("Indicator calculation failed, skipping...")
                    time.sleep(10)
                    continue

                # 3. 관측 생성 (마지막 캔들)
                observation = self._create_observation(df)

                # 4. AI 예측
                action, _ = self.agent.predict(observation, deterministic=True)

                # 5. 행동 실행
                self._execute_action(action, df.iloc[-1]['close'])

                # 6. 리스크 체크
                current_balance = self._get_current_balance()
                risk_check = self.risk_manager.check_risk_limits(
                    current_balance,
                    self.current_position,
                    df.iloc[-1]['close']
                )

                if not risk_check['can_trade']:
                    logger.warning("Risk limits exceeded, stopping trading")
                    self.stop()
                    break

                # 7. 대기
                logger.info(f"Waiting {interval_seconds}s for next candle...")
                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                logger.info("Trading interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(60)

        logger.info("Trading loop ended")

    def _create_observation(self, df) -> np.ndarray:
        """
        관측 생성

        Args:
            df: 데이터프레임

        Returns:
            관측 배열
        """
        # 마지막 행의 특성 추출
        last_row = df.iloc[-1]

        feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'ema_9', 'ema_21', 'ema_50',
            'rsi', 'bb_upper', 'bb_middle', 'bb_lower',
            'macd', 'macd_signal', 'adx', 'atr'
        ]

        market_features = []
        for col in feature_columns:
            if col in last_row:
                market_features.append(float(last_row[col]))
            else:
                market_features.append(0.0)

        # 계정 정보
        current_price = float(last_row['close'])
        position_normalized = self.current_position / 0.1  # max_position_size
        balance_normalized = self.balance / 10000.0  # initial_balance
        unrealized_pnl = self._calculate_unrealized_pnl(current_price)
        pnl_normalized = unrealized_pnl / 10000.0

        account_features = [
            position_normalized,
            balance_normalized,
            pnl_normalized,
            current_price / 10000.0
        ]

        observation = np.array(market_features + account_features, dtype=np.float32)
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

        return observation

    def _execute_action(self, action: np.ndarray, current_price: float) -> None:
        """
        행동 실행

        Args:
            action: [포지션 크기, 손절 비율, 익절 비율]
            current_price: 현재 가격
        """
        target_position, stop_loss_pct, take_profit_pct = action
        target_position = np.clip(target_position, -1.0, 1.0)
        max_position_size = self.config['trading']['max_position_size']
        target_position_size = target_position * max_position_size

        # 포지션 변화
        position_change = target_position_size - self.current_position

        # 최소 거래 크기 체크
        if abs(position_change) < 0.001:
            logger.debug("Position change too small, skipping trade")
            return

        symbol = self.config['exchange']['symbol']

        # 리스크 검증
        if not self.risk_manager.validate_order(
            'MARKET',
            position_change,
            current_price,
            self.balance
        ):
            logger.warning("Order failed risk validation")
            return

        # 주문 실행
        if self.dry_run:
            logger.info(f"[DRY RUN] Would execute: {position_change:.4f} BTC at {current_price:.2f}")
            self.current_position += position_change
        else:
            try:
                # 실제 주문
                side = 'BUY' if position_change > 0 else 'SELL'
                position_side = 'LONG' if target_position > 0 else 'SHORT'

                order = self.client.create_order(
                    symbol=symbol,
                    side=side,
                    position_side=position_side,
                    order_type='MARKET',
                    quantity=abs(position_change)
                )

                self.current_position = target_position_size
                logger.info(f"Order executed: {side} {abs(position_change):.4f} BTC")

                # 손절/익절 설정 (옵션)
                if self.config['risk']['use_stop_loss'] and abs(self.current_position) > 0.001:
                    self._set_stop_orders(current_price, stop_loss_pct, take_profit_pct)

            except BingXAPIError as e:
                logger.error(f"Order execution failed: {e.message}")

    def _set_stop_orders(
        self,
        entry_price: float,
        stop_loss_pct: float,
        take_profit_pct: float
    ) -> None:
        """
        손절/익절 주문 설정

        Args:
            entry_price: 진입 가격
            stop_loss_pct: 손절 비율
            take_profit_pct: 익절 비율
        """
        if self.dry_run or abs(self.current_position) < 0.001:
            return

        symbol = self.config['exchange']['symbol']

        try:
            # 손절 주문
            if stop_loss_pct > 0:
                stop_price = entry_price * (1 - stop_loss_pct) if self.current_position > 0 \
                             else entry_price * (1 + stop_loss_pct)

                self.client.create_order(
                    symbol=symbol,
                    side='SELL' if self.current_position > 0 else 'BUY',
                    position_side='LONG' if self.current_position > 0 else 'SHORT',
                    order_type='STOP_MARKET',
                    quantity=abs(self.current_position),
                    stop_price=stop_price
                )

                logger.info(f"Stop loss set at {stop_price:.2f}")

            # 익절 주문
            if take_profit_pct > 0:
                take_price = entry_price * (1 + take_profit_pct) if self.current_position > 0 \
                             else entry_price * (1 - take_profit_pct)

                self.client.create_order(
                    symbol=symbol,
                    side='SELL' if self.current_position > 0 else 'BUY',
                    position_side='LONG' if self.current_position > 0 else 'SHORT',
                    order_type='TAKE_PROFIT_MARKET',
                    quantity=abs(self.current_position),
                    stop_price=take_price
                )

                logger.info(f"Take profit set at {take_price:.2f}")

        except BingXAPIError as e:
            logger.error(f"Failed to set stop orders: {e.message}")

    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """미실현 손익 계산"""
        # 간단한 계산 (실제로는 진입가격 추적 필요)
        return 0.0

    def _get_current_balance(self) -> float:
        """현재 잔고 조회"""
        if self.dry_run:
            return self.balance

        try:
            balance_info = self.client.get_balance()
            return float(balance_info['balance']['balance'])
        except BingXAPIError as e:
            logger.error(f"Failed to get balance: {e.message}")
            return self.balance

    def _close_all_positions(self) -> None:
        """모든 포지션 청산"""
        if abs(self.current_position) < 0.001:
            return

        symbol = self.config['exchange']['symbol']

        try:
            position_side = 'LONG' if self.current_position > 0 else 'SHORT'
            self.client.close_position(symbol, position_side)
            logger.info("All positions closed")
            self.current_position = 0.0
        except BingXAPIError as e:
            logger.error(f"Failed to close positions: {e.message}")
