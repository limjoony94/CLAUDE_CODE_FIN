"""리스크 관리 시스템"""

from typing import Dict, Any

import numpy as np
from loguru import logger


class RiskManager:
    """
    거래 리스크 관리
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 리스크 설정
        """
        self.max_drawdown = config.get('max_drawdown', 0.15)
        self.max_daily_loss = config.get('max_daily_loss', 0.05)
        self.max_position_risk = config.get('max_position_risk', 0.02)

        self.initial_balance = 0.0
        self.peak_balance = 0.0
        self.daily_start_balance = 0.0
        self.trades_today = 0

        self.circuit_breaker_active = False

        logger.info(f"Risk Manager initialized - Max DD: {self.max_drawdown*100}%, "
                   f"Max Daily Loss: {self.max_daily_loss*100}%")

    def initialize(self, initial_balance: float) -> None:
        """
        초기화

        Args:
            initial_balance: 초기 잔고
        """
        self.initial_balance = initial_balance
        self.peak_balance = initial_balance
        self.daily_start_balance = initial_balance
        self.circuit_breaker_active = False

    def check_risk_limits(
        self,
        current_balance: float,
        position_size: float,
        price: float
    ) -> Dict[str, Any]:
        """
        리스크 한도 체크

        Args:
            current_balance: 현재 잔고
            position_size: 포지션 크기
            price: 현재 가격

        Returns:
            리스크 체크 결과
        """
        # Peak 업데이트
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance

        # 낙폭 계산
        current_drawdown = (self.peak_balance - current_balance) / self.peak_balance

        # 일일 손실 계산
        daily_loss = (self.daily_start_balance - current_balance) / self.daily_start_balance

        # 포지션 리스크 계산
        position_value = abs(position_size) * price
        position_risk = position_value / current_balance if current_balance > 0 else 1.0

        # 리스크 한도 초과 체크
        risk_exceeded = {
            'max_drawdown_exceeded': current_drawdown > self.max_drawdown,
            'max_daily_loss_exceeded': daily_loss > self.max_daily_loss,
            'max_position_risk_exceeded': position_risk > self.max_position_risk
        }

        # 서킷 브레이커 활성화
        if any(risk_exceeded.values()):
            self.circuit_breaker_active = True
            logger.warning(f"Risk limits exceeded! Circuit breaker activated. "
                          f"DD: {current_drawdown*100:.2f}%, Daily Loss: {daily_loss*100:.2f}%")

        return {
            'current_drawdown': current_drawdown,
            'daily_loss': daily_loss,
            'position_risk': position_risk,
            'risk_exceeded': risk_exceeded,
            'circuit_breaker_active': self.circuit_breaker_active,
            'can_trade': not self.circuit_breaker_active
        }

    def calculate_position_size(
        self,
        balance: float,
        risk_per_trade: float,
        stop_loss_distance: float,
        leverage: int = 10
    ) -> float:
        """
        적정 포지션 크기 계산

        Args:
            balance: 현재 잔고
            risk_per_trade: 거래당 리스크 (비율)
            stop_loss_distance: 손절까지 거리 (비율)
            leverage: 레버리지

        Returns:
            포지션 크기 (BTC)
        """
        if stop_loss_distance <= 0:
            return 0.0

        # 리스크 금액
        risk_amount = balance * risk_per_trade

        # 포지션 크기 계산
        position_size = risk_amount / stop_loss_distance / leverage

        return position_size

    def validate_order(
        self,
        order_type: str,
        position_size: float,
        price: float,
        balance: float
    ) -> bool:
        """
        주문 검증

        Args:
            order_type: 주문 타입
            position_size: 포지션 크기
            price: 가격
            balance: 잔고

        Returns:
            주문 가능 여부
        """
        # 서킷 브레이커 체크
        if self.circuit_breaker_active:
            logger.warning("Circuit breaker active - order rejected")
            return False

        # 잔고 체크
        required_margin = abs(position_size) * price
        if required_margin > balance:
            logger.warning(f"Insufficient balance - Required: {required_margin:.2f}, "
                          f"Available: {balance:.2f}")
            return False

        # 포지션 크기 체크
        position_value = abs(position_size) * price
        position_risk = position_value / balance if balance > 0 else 1.0

        if position_risk > self.max_position_risk:
            logger.warning(f"Position risk too high - {position_risk*100:.2f}% "
                          f"(Max: {self.max_position_risk*100:.2f}%)")
            return False

        return True

    def reset_daily_stats(self, current_balance: float) -> None:
        """
        일일 통계 리셋

        Args:
            current_balance: 현재 잔고
        """
        self.daily_start_balance = current_balance
        self.trades_today = 0
        self.circuit_breaker_active = False
        logger.info("Daily stats reset")

    def get_risk_metrics(self, current_balance: float) -> Dict[str, float]:
        """
        리스크 지표 조회

        Args:
            current_balance: 현재 잔고

        Returns:
            리스크 지표
        """
        current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
        daily_loss = (self.daily_start_balance - current_balance) / self.daily_start_balance

        return {
            'current_drawdown': current_drawdown,
            'daily_loss': daily_loss,
            'peak_balance': self.peak_balance,
            'drawdown_limit': self.max_drawdown,
            'daily_loss_limit': self.max_daily_loss,
            'circuit_breaker_active': self.circuit_breaker_active
        }
