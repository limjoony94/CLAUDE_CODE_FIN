"""V5: 최종 개선 - 순수 수익 + 강력한 제약"""

from typing import Dict, Any, Tuple, Optional
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from loguru import logger


class TradingEnvironmentV5(gym.Env):
    """
    최종 개선:
    1. 홀딩 보너스 완전 제거
    2. 강력한 거래 빈도 제약 (최소 10 스텝)
    3. 손절/익절 완화 (2%/3%)
    4. 순수 포트폴리오 수익률만 보상
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        leverage: int = 3,
        transaction_fee: float = 0.0004,
        slippage: float = 0.0001,
        max_position_size: float = 0.03,
        stop_loss_pct: float = 0.02,  # 1% → 2%
        take_profit_pct: float = 0.03,  # 2% → 3%
        min_hold_steps: int = 10  # 3 → 10 스텝
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
        self.min_hold_steps = min_hold_steps

        # 특성
        self.feature_columns = self._select_features()
        n_features = len(self.feature_columns) + 5

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )

        # 행동: 포지션만
        self.action_space = spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )

        # 상태
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.entry_step = 0
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0

        logger.info(f"Trading Environment V5 (Pure PnL, Strict Constraints) - {len(self.df)} steps")

    def _select_features(self) -> list:
        return [
            'close', 'volume',
            'ema_9', 'ema_21', 'ema_50',
            'rsi',
            'macd', 'macd_signal',
            'bb_upper', 'bb_middle', 'bb_lower',
            'atr'
        ]

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
        self.entry_step = 0
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0

        return self._get_observation(), self._get_info()

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        current_price = self._get_current_price()
        prev_portfolio = self._calculate_portfolio_value()

        # 포지션 조정
        target_position = np.clip(float(action[0]), -1.0, 1.0)
        target_position_size = target_position * self.max_position_size
        position_change = target_position_size - self.position

        # 거래 실행
        trade_pnl = self._execute_trade(position_change, current_price)

        # 스텝 이동
        self.current_step += 1

        # 손절/익절
        if self.current_step < len(self.df):
            next_price = self._get_current_price()
            self._check_stop_conditions(next_price)

        # V5 보상: 순수 수익률 + 강력한 제약만
        new_portfolio = self._calculate_portfolio_value()
        reward = self._calculate_reward_v5(
            prev_portfolio,
            new_portfolio,
            position_change
        )

        terminated = self.balance <= 0 or self.current_step >= len(self.df) - 1
        truncated = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_observation(self) -> np.ndarray:
        row = self.df.iloc[self.current_step]

        # 시장 특성
        market_features = [float(row[col]) for col in self.feature_columns]

        # 시장 체제
        market_regime = self._detect_market_regime()

        # 계정
        position_norm = self.position / self.max_position_size
        balance_ratio = self.balance / self.initial_balance
        pnl_ratio = self.total_pnl / self.initial_balance

        observation = np.array(
            market_features + list(market_regime) + [position_norm, balance_ratio, pnl_ratio],
            dtype=np.float32
        )

        return np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

    def _detect_market_regime(self) -> Tuple[float, float]:
        if self.current_step < 50:
            return (0.0, 0.0)

        start_idx = max(0, self.current_step - 50)
        recent = self.df.iloc[start_idx:self.current_step + 1]

        # 추세 강도
        if recent['close'].iloc[0] > 0:
            price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
            trend_strength = np.clip(price_change * 20, -1.0, 1.0)
        else:
            trend_strength = 0.0

        # 변동성
        volatility = recent['close'].pct_change().std()
        avg_volatility = 0.002
        volatility_regime = np.clip(volatility / avg_volatility, 0.0, 2.0) - 1.0

        return (trend_strength, volatility_regime)

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

        # 청산
        if abs(self.position) > 0.001 and np.sign(position_change) != np.sign(self.position):
            close_size = min(abs(position_change), abs(self.position))
            price_diff = (execution_price - self.entry_price) * np.sign(self.position)
            realized_pnl = close_size * price_diff * self.leverage - fee

            self.balance += realized_pnl
            self.total_pnl += realized_pnl

            self.trade_count += 1
            if realized_pnl > 0:
                self.win_count += 1

        # 진입
        remaining_change = position_change
        if abs(self.position) > 0.001 and np.sign(position_change) != np.sign(self.position):
            remaining_change = position_change + self.position

        if abs(remaining_change) > 0.001:
            required_margin = abs(remaining_change) * execution_price / self.leverage

            if required_margin <= self.balance:
                self.position += remaining_change
                self.entry_price = execution_price
                self.entry_step = self.current_step
                self.balance -= fee

        return realized_pnl

    def _check_stop_conditions(self, current_price: float) -> None:
        if abs(self.position) < 0.001:
            return

        if self.position > 0:
            stop_loss = self.entry_price * (1 - self.stop_loss_pct)
            take_profit = self.entry_price * (1 + self.take_profit_pct)

            if current_price <= stop_loss or current_price >= take_profit:
                self._execute_trade(-self.position, current_price)

        elif self.position < 0:
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

    def _calculate_reward_v5(
        self,
        prev_value: float,
        new_value: float,
        position_change: float
    ) -> float:
        """
        V5 보상: 순수 포트폴리오 수익률 + 강력한 거래 빈도 제약만

        핵심:
        - 홀딩 보너스 완전 제거
        - 포트폴리오 수익률만 반영
        - 과도한 거래 강력 페널티
        """
        if prev_value <= 0:
            return -1000.0

        # 1. 순수 포트폴리오 수익률 (유일한 주 보상)
        portfolio_return = (new_value - prev_value) / prev_value
        reward = portfolio_return * 10000

        # 2. 강력한 거래 빈도 제약
        if abs(position_change) > 0.001:
            hold_duration = self.current_step - self.entry_step

            if hold_duration < self.min_hold_steps:
                # 최소 홀딩 기간 미달 시 강력 페널티
                frequency_penalty = -100.0  # -5 → -100으로 강화
                reward += frequency_penalty

        # 3. 파산 페널티
        if self.balance <= 0:
            reward -= 1000.0

        return reward

    def render(self, mode='human'):
        if mode == 'human':
            portfolio = self._calculate_portfolio_value()
            print(f"Step: {self.current_step}, Portfolio: ${portfolio:.2f}, "
                  f"Position: {self.position:.4f}, PnL: ${self.total_pnl:.2f}")

    def close(self):
        pass
