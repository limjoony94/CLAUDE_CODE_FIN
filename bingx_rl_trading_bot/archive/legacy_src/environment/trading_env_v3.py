"""최종 개선 환경 V3 - 순수 수익률 중심"""

from typing import Dict, Any, Tuple, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from loguru import logger


class TradingEnvironmentV3(gym.Env):
    """
    순수 수익률 중심 거래 환경
    - 보상 = 포트폴리오 수익률만 사용
    - 명확한 학습 신호
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        leverage: int = 10,
        transaction_fee: float = 0.0004,
        slippage: float = 0.0001,
        max_position_size: float = 0.1,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # 특성 선택
        self.feature_columns = self._select_features()
        n_features = len(self.feature_columns) + 3

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )

        # 행동: 포지션 크기만
        self.action_space = spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )

        # 상태 변수
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0

        logger.info(f"Trading Environment V3 (Pure PnL) initialized with {len(self.df)} steps")

    def _select_features(self) -> list:
        """핵심 특성만 선택"""
        core_features = [
            'close', 'volume',
            'ema_9', 'ema_21', 'ema_50',
            'rsi',
            'macd', 'macd_signal',
            'bb_upper', 'bb_middle', 'bb_lower',
            'atr'
        ]
        return [f for f in core_features if f in self.df.columns]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0

        return self._get_observation(), self._get_info()

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        current_price = self._get_current_price()

        # 거래 전 포트폴리오 가치
        prev_portfolio = self._calculate_portfolio_value()

        # 포지션 조정
        target_position = np.clip(float(action[0]), -1.0, 1.0)
        target_position_size = target_position * self.max_position_size

        # 거래 실행
        self._execute_trade(target_position_size - self.position, current_price)

        # 다음 스텝
        self.current_step += 1

        # 손절/익절 체크
        if self.current_step < len(self.df):
            next_price = self._get_current_price()
            self._check_stop_conditions(next_price)

        # 순수 수익률 보상
        new_portfolio = self._calculate_portfolio_value()
        reward = self._calculate_pure_return_reward(prev_portfolio, new_portfolio)

        # 종료 조건
        terminated = self.balance <= 0 or self.current_step >= len(self.df) - 1
        truncated = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self) -> np.ndarray:
        row = self.df.iloc[self.current_step]

        # 시장 특성
        market_features = [float(row[col]) for col in self.feature_columns]

        # 계정 특성
        position_norm = self.position / self.max_position_size
        balance_ratio = self.balance / self.initial_balance
        pnl_ratio = self.total_pnl / self.initial_balance

        observation = np.array(market_features + [position_norm, balance_ratio, pnl_ratio], dtype=np.float32)
        return np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

    def _get_info(self) -> Dict[str, Any]:
        current_price = self._get_current_price()

        return {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': self._calculate_portfolio_value(),
            'total_pnl': self.total_pnl,
            'trade_count': self.trade_count,
            'win_rate': self.win_count / max(self.trade_count, 1),
            'current_price': current_price,
            'unrealized_pnl': self._calculate_unrealized_pnl(current_price)
        }

    def _get_current_price(self) -> float:
        if self.current_step >= len(self.df):
            return self.df.iloc[-1]['close']
        return self.df.iloc[self.current_step]['close']

    def _execute_trade(self, position_change: float, price: float) -> float:
        if abs(position_change) < 0.001:
            return 0.0

        execution_price = price * (1 + self.slippage * np.sign(position_change))
        trade_value = abs(position_change) * execution_price * self.leverage
        fee = trade_value * self.transaction_fee

        realized_pnl = 0.0

        # 기존 포지션 청산
        if abs(self.position) > 0.001 and np.sign(position_change) != np.sign(self.position):
            close_size = min(abs(position_change), abs(self.position))
            price_diff = (execution_price - self.entry_price) * np.sign(self.position)
            realized_pnl = close_size * price_diff * self.leverage - fee

            self.balance += realized_pnl
            self.total_pnl += realized_pnl

            self.trade_count += 1
            if realized_pnl > 0:
                self.win_count += 1

        # 새 포지션 진입
        remaining_change = position_change
        if abs(self.position) > 0.001 and np.sign(position_change) != np.sign(self.position):
            remaining_change = position_change + self.position

        if abs(remaining_change) > 0.001:
            # 올바른 증거금 계산 (레버리지 적용)
            required_margin = abs(remaining_change) * execution_price / self.leverage

            if required_margin <= self.balance:
                self.position += remaining_change
                self.entry_price = execution_price
                self.balance -= fee

        return realized_pnl

    def _check_stop_conditions(self, current_price: float) -> None:
        if abs(self.position) < 0.001:
            return

        if self.position > 0:  # 롱
            stop_loss = self.entry_price * (1 - self.stop_loss_pct)
            take_profit = self.entry_price * (1 + self.take_profit_pct)

            if current_price <= stop_loss or current_price >= take_profit:
                self._execute_trade(-self.position, current_price)

        elif self.position < 0:  # 숏
            stop_loss = self.entry_price * (1 + self.stop_loss_pct)
            take_profit = self.entry_price * (1 - self.take_profit_pct)

            if current_price >= stop_loss or current_price <= take_profit:
                self._execute_trade(-self.position, current_price)

    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        if abs(self.position) < 0.001:
            return 0.0

        price_diff = (current_price - self.entry_price) * np.sign(self.position)
        return abs(self.position) * price_diff * self.leverage

    def _calculate_portfolio_value(self) -> float:
        current_price = self._get_current_price()
        unrealized_pnl = self._calculate_unrealized_pnl(current_price)
        return self.balance + unrealized_pnl

    def _calculate_pure_return_reward(self, prev_value: float, new_value: float) -> float:
        """
        순수 수익률 기반 보상

        보상 = 포트폴리오 수익률 × 10000
        → Mean Reward 100 = 1% 수익률
        """
        if prev_value <= 0:
            return -1000.0  # 파산

        # 순수 포트폴리오 수익률
        portfolio_return = (new_value - prev_value) / prev_value

        # 10000배 증폭 (학습 신호 강화)
        reward = portfolio_return * 10000

        # 파산 강력 페널티
        if self.balance <= 0:
            reward -= 1000.0

        return reward

    def render(self, mode='human'):
        if mode == 'human':
            portfolio = self._calculate_portfolio_value()
            print(f"Step: {self.current_step}, "
                  f"Portfolio: ${portfolio:.2f}, "
                  f"Position: {self.position:.4f}, "
                  f"PnL: ${self.total_pnl:.2f}")

    def close(self):
        pass
