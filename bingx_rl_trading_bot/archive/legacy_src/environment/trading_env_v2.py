"""개선된 Gymnasium 거래 환경 v2"""

from typing import Dict, Any, Tuple, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from loguru import logger


class TradingEnvironmentV2(gym.Env):
    """
    개선된 비트코인 선물 거래 환경
    - 단순화된 행동 공간 (포지션만)
    - 강화된 보상 신호
    - 개선된 리스크 관리
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
        stop_loss_pct: float = 0.02,  # 고정 2% 손절
        take_profit_pct: float = 0.04,  # 고정 4% 익절
        reward_scaling: float = 100.0  # 보상 증폭
    ):
        """
        Args:
            df: OHLCV + 지표 데이터프레임
            initial_balance: 초기 잔고 (USDT)
            leverage: 레버리지 배수
            transaction_fee: 거래 수수료 비율
            slippage: 슬리피지 비율
            max_position_size: 최대 포지션 크기 (BTC)
            stop_loss_pct: 고정 손절 비율
            take_profit_pct: 고정 익절 비율
            reward_scaling: 보상 스케일링
        """
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.reward_scaling = reward_scaling

        # 상태 공간: 단순화된 특성
        self.feature_columns = self._select_features()
        n_features = len(self.feature_columns) + 3  # +3 for position, balance_ratio, pnl_ratio

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )

        # 행동 공간: 단순화 - 포지션 크기만
        # -1 (풀 숏) ~ 0 (노포지션) ~ 1 (풀 롱)
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
        self.max_portfolio_value = initial_balance

        # 에피소드 기록
        self.episode_returns = []

        logger.info(f"Trading Environment V2 initialized with {len(self.df)} steps")

    def _select_features(self) -> list:
        """관측 특성 선택 - 핵심 특성만"""
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
        """환경 리셋"""
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.max_portfolio_value = self.initial_balance
        self.episode_returns = []

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """환경 스텝 실행"""
        current_price = self._get_current_price()

        # 행동 실행 (포지션 조정)
        target_position = float(action[0])
        target_position = np.clip(target_position, -1.0, 1.0)
        target_position_size = target_position * self.max_position_size

        # 거래 전 포트폴리오 가치
        prev_portfolio = self._calculate_portfolio_value()

        # 거래 실행
        trade_pnl = self._execute_trade(target_position_size - self.position, current_price)

        # 다음 스텝으로 이동
        self.current_step += 1

        # 다음 스텝 가격으로 손절/익절 체크
        if self.current_step < len(self.df):
            next_price = self._get_current_price()
            self._check_stop_conditions(next_price)

        # 보상 계산 (개선된 보상 함수)
        new_portfolio = self._calculate_portfolio_value()
        reward = self._calculate_reward_v2(prev_portfolio, new_portfolio, trade_pnl)

        # 종료 조건
        terminated = self.balance <= 0 or self.current_step >= len(self.df) - 1
        truncated = False

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """현재 관측 생성"""
        row = self.df.iloc[self.current_step]

        # 시장 특성
        market_features = []
        for col in self.feature_columns:
            value = float(row[col])
            market_features.append(value)

        # 계정 특성 (정규화)
        position_norm = self.position / self.max_position_size
        balance_ratio = self.balance / self.initial_balance
        pnl_ratio = self.total_pnl / self.initial_balance

        account_features = [position_norm, balance_ratio, pnl_ratio]

        observation = np.array(market_features + account_features, dtype=np.float32)
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

        return observation

    def _get_info(self) -> Dict[str, Any]:
        """정보 딕셔너리"""
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
        """현재 가격"""
        if self.current_step >= len(self.df):
            return self.df.iloc[-1]['close']
        return self.df.iloc[self.current_step]['close']

    def _execute_trade(self, position_change: float, price: float) -> float:
        """거래 실행"""
        if abs(position_change) < 0.001:
            return 0.0

        # 슬리피지 적용
        execution_price = price * (1 + self.slippage * np.sign(position_change))

        # 거래 비용
        trade_value = abs(position_change) * execution_price * self.leverage
        fee = trade_value * self.transaction_fee

        realized_pnl = 0.0

        # 기존 포지션 청산
        if abs(self.position) > 0.001:
            if np.sign(position_change) != np.sign(self.position):
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
            # 레버리지 고려한 올바른 증거금 계산
            required_margin = abs(remaining_change) * execution_price / self.leverage
            if required_margin <= self.balance:
                self.position += remaining_change
                self.entry_price = execution_price
                self.balance -= fee
            else:
                logger.debug(f"Insufficient margin: need ${required_margin:.2f}, have ${self.balance:.2f}")

        return realized_pnl

    def _check_stop_conditions(self, current_price: float) -> None:
        """손절/익절 조건 확인 (고정 비율)"""
        if abs(self.position) < 0.001:
            return

        # 롱 포지션
        if self.position > 0:
            stop_loss = self.entry_price * (1 - self.stop_loss_pct)
            take_profit = self.entry_price * (1 + self.take_profit_pct)

            if current_price <= stop_loss or current_price >= take_profit:
                self._execute_trade(-self.position, current_price)

        # 숏 포지션
        elif self.position < 0:
            stop_loss = self.entry_price * (1 + self.stop_loss_pct)
            take_profit = self.entry_price * (1 - self.take_profit_pct)

            if current_price >= stop_loss or current_price <= take_profit:
                self._execute_trade(-self.position, current_price)

    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """미실현 손익"""
        if abs(self.position) < 0.001:
            return 0.0

        price_diff = (current_price - self.entry_price) * np.sign(self.position)
        return abs(self.position) * price_diff * self.leverage

    def _calculate_portfolio_value(self) -> float:
        """포트폴리오 가치"""
        current_price = self._get_current_price()
        unrealized_pnl = self._calculate_unrealized_pnl(current_price)
        return self.balance + unrealized_pnl

    def _calculate_reward_v2(
        self,
        prev_value: float,
        new_value: float,
        trade_pnl: float
    ) -> float:
        """
        개선된 보상 함수

        - 포트폴리오 수익률 기반
        - 거래 손익 보너스/페널티
        - 위험 조정 보상
        """
        # 1. 포트폴리오 수익률 (주 보상) - 이미 증폭됨
        if prev_value > 0:
            portfolio_return = (new_value - prev_value) / prev_value
        else:
            portfolio_return = 0.0
        portfolio_reward = portfolio_return * 1000  # 증폭

        # 2. 거래 손익 보상 (보조 보상)
        trade_reward = 0.0
        if abs(trade_pnl) > 0.001:
            trade_reward = (trade_pnl / self.initial_balance) * 100

            # 승리 보너스/패배 페널티
            if trade_pnl > 0:
                trade_reward += 1.0  # 승리 보너스
            else:
                trade_reward -= 1.0  # 패배 페널티

        # 3. 파산 페널티 (완화)
        bankruptcy_penalty = 0.0
        if self.balance <= 0:
            bankruptcy_penalty = -10.0  # 기존 -100에서 완화

        # 4. 위험 페널티 (완화)
        risk_penalty = 0.0
        if abs(self.position) > 0.9 * self.max_position_size:
            risk_penalty = -0.1  # 극단적 포지션만 페널티

        # 총 보상 (스케일링 제거 - 이미 충분히 증폭됨)
        total_reward = portfolio_reward + trade_reward + bankruptcy_penalty + risk_penalty

        # 신고점 보너스
        if new_value > self.max_portfolio_value:
            self.max_portfolio_value = new_value
            total_reward += 2.0  # 신고점 보너스

        return total_reward

    def render(self, mode='human'):
        """환경 렌더링"""
        if mode == 'human':
            portfolio = self._calculate_portfolio_value()
            print(f"Step: {self.current_step}, "
                  f"Portfolio: ${portfolio:.2f}, "
                  f"Position: {self.position:.4f}, "
                  f"PnL: ${self.total_pnl:.2f}")

    def close(self):
        """환경 종료"""
        pass
