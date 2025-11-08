"""V6: 적응형 보상 스케일 + 행동 제약 - 근본 원인 해결"""

from typing import Dict, Any, Tuple, Optional
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from loguru import logger


class TradingEnvironmentV6(gym.Env):
    """
    V6 핵심 개선사항:
    1. 적응형 보상 스케일 (volatility 기반, Mode Collapse 해결)
    2. 행동 공간에서 거래 빈도 물리적 제약 (페널티 아님)
    3. 손절/익절 완화 (3%/5%)
    4. 순수 포트폴리오 수익률만 보상 (페널티 완전 제거)

    근본 원인 해결:
    - 보상 스케일 불균형 → 적응형 스케일로 해결
    - Mode Collapse → 거래 물리적 강제로 다양성 확보
    - 보상 왜곡 → 페널티 제거, 순수 수익률만
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
        stop_loss_pct: float = 0.03,  # 2% → 3%
        take_profit_pct: float = 0.05,  # 3% → 5%
        min_hold_steps: int = 10,
        volatility_window: int = 50  # 적응형 스케일용
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
        self.volatility_window = volatility_window

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
        self.last_trade_step = 0  # 마지막 거래 시점
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0

        # 적응형 스케일용 통계
        self.recent_returns = []
        self.current_volatility = 0.002  # 초기값

        logger.info(f"Trading Environment V6 (Adaptive Reward Scale) - {len(self.df)} steps")
        logger.info(f"  - Adaptive reward scaling based on {volatility_window}-step volatility")
        logger.info(f"  - Physical action constraint (min_hold: {min_hold_steps})")
        logger.info(f"  - Pure PnL reward (NO penalties)")
        logger.info(f"  - Stop loss: {stop_loss_pct*100}%, Take profit: {take_profit_pct*100}%")

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
        self.last_trade_step = 0
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0

        self.recent_returns = []
        self.current_volatility = 0.002

        return self._get_observation(), self._get_info()

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        current_price = self._get_current_price()
        prev_portfolio = self._calculate_portfolio_value()

        # 포지션 목표
        target_position = np.clip(float(action[0]), -1.0, 1.0)
        target_position_size = target_position * self.max_position_size

        # 🔑 핵심: 최소 홀딩 기간 물리적 강제
        if self.current_step - self.last_trade_step < self.min_hold_steps:
            # 거래 불가 → 현재 포지션 유지
            target_position_size = self.position

        # 거래 실행
        position_change = target_position_size - self.position
        trade_pnl = self._execute_trade(position_change, current_price)

        # 거래 발생 시 기록
        if abs(position_change) > 0.001:
            self.last_trade_step = self.current_step

        # 스텝 이동
        self.current_step += 1

        # 손절/익절
        if self.current_step < len(self.df):
            next_price = self._get_current_price()
            self._check_stop_conditions(next_price)

        # 🔑 V6 보상: 적응형 스케일 + 순수 수익률
        new_portfolio = self._calculate_portfolio_value()
        reward = self._calculate_reward_v6(prev_portfolio, new_portfolio)

        # 포트폴리오 수익률 기록 (적응형 스케일용)
        portfolio_return = (new_portfolio - prev_portfolio) / prev_portfolio if prev_portfolio > 0 else 0.0
        self._update_volatility(portfolio_return)

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
        """시장 체제 감지"""
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

        # 변동성 체제
        volatility = recent['close'].pct_change().std()
        avg_volatility = 0.002
        volatility_regime = np.clip(volatility / avg_volatility, 0.0, 2.0) - 1.0

        return (trend_strength, volatility_regime)

    def _update_volatility(self, portfolio_return: float) -> None:
        """적응형 스케일용 변동성 업데이트"""
        self.recent_returns.append(portfolio_return)

        # 최근 window만 유지
        if len(self.recent_returns) > self.volatility_window:
            self.recent_returns.pop(0)

        # 변동성 계산
        if len(self.recent_returns) >= 10:
            self.current_volatility = max(np.std(self.recent_returns), 0.0001)  # 최소값 방지
        else:
            self.current_volatility = 0.002  # 초기 기본값

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
            'unrealized_pnl': self._calculate_unrealized_pnl(current_price),
            'current_volatility': self.current_volatility,  # 디버깅용
            'last_trade_step': self.last_trade_step
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

    def _calculate_reward_v6(
        self,
        prev_value: float,
        new_value: float
    ) -> float:
        """
        V6 보상 함수: 적응형 스케일 + 순수 수익률

        핵심 개선:
        1. 적응형 스케일: volatility 기반으로 동적 조정
        2. 페널티 완전 제거: 행동 제약으로 대체
        3. 순수 수익률만 반영

        수학적 근거:
        - reward_scale = 100 / volatility
        - volatility가 낮으면 (0.001) → scale 100,000 (민감)
        - volatility가 높으면 (0.01) → scale 10,000 (안정)
        - 평균 volatility (0.002) → scale 50,000
        """
        if prev_value <= 0:
            return -1000.0  # 파산만 페널티

        # 포트폴리오 수익률
        portfolio_return = (new_value - prev_value) / prev_value

        # 🔑 적응형 보상 스케일
        # volatility가 낮을수록 큰 스케일 (작은 수익도 크게 보상)
        # volatility가 높을수록 작은 스케일 (큰 변동을 정규화)
        reward_scale = 100.0 / max(self.current_volatility, 0.0001)

        # 스케일 범위 제한 (안정성)
        reward_scale = np.clip(reward_scale, 5000.0, 100000.0)

        reward = portfolio_return * reward_scale

        # 파산 페널티만 유지
        if self.balance <= 0:
            reward -= 1000.0

        return reward

    def render(self, mode='human'):
        if mode == 'human':
            portfolio = self._calculate_portfolio_value()
            print(f"Step: {self.current_step}, Portfolio: ${portfolio:.2f}, "
                  f"Position: {self.position:.4f}, PnL: ${self.total_pnl:.2f}, "
                  f"Volatility: {self.current_volatility:.6f}")

    def close(self):
        pass
