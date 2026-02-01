"""Gymnasium 거래 환경"""

from typing import Dict, Any, Tuple, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from loguru import logger


class TradingEnvironment(gym.Env):
    """
    비트코인 선물 거래를 위한 강화학습 환경
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
        reward_scaling: float = 1.0
    ):
        """
        Args:
            df: OHLCV + 지표 데이터프레임
            initial_balance: 초기 잔고 (USDT)
            leverage: 레버리지 배수
            transaction_fee: 거래 수수료 비율
            slippage: 슬리피지 비율
            max_position_size: 최대 포지션 크기 (BTC)
            reward_scaling: 보상 스케일링
        """
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.transaction_fee = transaction_fee
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.reward_scaling = reward_scaling

        # 상태 공간: OHLCV + 지표 + 계정 정보
        self.feature_columns = self._select_features()
        n_features = len(self.feature_columns) + 4  # +4 for position, balance, pnl, price

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features,),
            dtype=np.float32
        )

        # 행동 공간: [포지션 크기, 손절 비율, 익절 비율]
        # 포지션 크기: -1 (풀 숏) ~ 0 (노포지션) ~ 1 (풀 롱)
        # 손절/익절: 0 ~ 1 (가격 대비 비율)
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 0.5, 0.5]),
            dtype=np.float32
        )

        # 상태 변수
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0  # 현재 포지션 (BTC, 양수: 롱, 음수: 숏)
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0

        # 에피소드 통계
        self.episode_stats = {
            'total_reward': 0,
            'total_pnl': 0,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }

        logger.info(f"Trading environment initialized with {len(self.df)} steps")

    def _select_features(self) -> list:
        """
        관측 특성 선택

        Returns:
            특성 컬럼 리스트
        """
        # 기본 특성
        base_features = ['open', 'high', 'low', 'close', 'volume']

        # 지표 특성
        indicator_features = []
        for col in self.df.columns:
            if col not in base_features and col != 'timestamp':
                # bool 타입 제외
                if self.df[col].dtype != bool:
                    indicator_features.append(col)

        features = base_features + indicator_features
        return [f for f in features if f in self.df.columns]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        환경 리셋

        Args:
            seed: 랜덤 시드
            options: 추가 옵션

        Returns:
            (초기 관측, 정보 딕셔너리)
        """
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0

        self.episode_stats = {
            'total_reward': 0,
            'total_pnl': 0,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        환경 스텝 실행

        Args:
            action: [포지션 크기, 손절 비율, 익절 비율]

        Returns:
            (관측, 보상, 종료, 잘림, 정보)
        """
        # 현재 가격
        current_price = self._get_current_price()

        # 이전 계정 가치
        prev_portfolio_value = self._calculate_portfolio_value()

        # 포지션 청산 체크 (손절/익절)
        closed_pnl = self._check_stop_conditions(current_price)

        # 행동 실행
        target_position, stop_loss_pct, take_profit_pct = action
        target_position = np.clip(target_position, -1.0, 1.0)
        target_position_size = target_position * self.max_position_size

        # 포지션 조정
        position_change = target_position_size - self.position
        trade_pnl = self._execute_trade(position_change, current_price)

        # 손절/익절 설정
        if abs(self.position) > 0.001:
            self._set_stop_conditions(current_price, stop_loss_pct, take_profit_pct)

        # 다음 스텝으로 이동
        self.current_step += 1

        # 보상 계산
        new_portfolio_value = self._calculate_portfolio_value()
        reward = self._calculate_reward(prev_portfolio_value, new_portfolio_value, trade_pnl)

        # 에피소드 종료 조건
        terminated = self.balance <= 0 or self.current_step >= len(self.df) - 1
        truncated = False

        # 관측 및 정보
        observation = self._get_observation()
        info = self._get_info()

        # 통계 업데이트
        self.episode_stats['total_reward'] += reward

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        현재 관측 생성

        Returns:
            관측 배열
        """
        # 시장 데이터
        row = self.df.iloc[self.current_step]
        market_features = row[self.feature_columns].values.astype(np.float32)

        # 정규화를 위한 현재 가격
        current_price = float(row['close'])

        # 계정 정보
        position_normalized = self.position / self.max_position_size  # -1 ~ 1
        balance_normalized = self.balance / self.initial_balance  # 상대 잔고
        unrealized_pnl = self._calculate_unrealized_pnl(current_price)
        pnl_normalized = unrealized_pnl / self.initial_balance  # 상대 손익

        account_features = np.array([
            position_normalized,
            balance_normalized,
            pnl_normalized,
            current_price / 10000.0  # 가격 스케일링
        ], dtype=np.float32)

        # 전체 관측
        observation = np.concatenate([market_features, account_features])

        # NaN 처리
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

        return observation

    def _get_info(self) -> Dict[str, Any]:
        """
        정보 딕셔너리 생성

        Returns:
            정보 딕셔너리
        """
        current_price = self._get_current_price()

        return {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': self._calculate_portfolio_value(),
            'unrealized_pnl': self._calculate_unrealized_pnl(current_price),
            'total_pnl': self.total_pnl,
            'trade_count': self.trade_count,
            'win_rate': self.win_count / max(self.trade_count, 1),
            'current_price': current_price
        }

    def _get_current_price(self) -> float:
        """현재 가격 조회"""
        if self.current_step >= len(self.df):
            return self.df.iloc[-1]['close']
        return self.df.iloc[self.current_step]['close']

    def _execute_trade(self, position_change: float, price: float) -> float:
        """
        거래 실행

        Args:
            position_change: 포지션 변화량
            price: 거래 가격

        Returns:
            실현 손익
        """
        if abs(position_change) < 0.0001:
            return 0.0

        # 슬리피지 적용
        execution_price = price * (1 + self.slippage * np.sign(position_change))

        # 거래 비용
        trade_value = abs(position_change) * execution_price * self.leverage
        fee = trade_value * self.transaction_fee

        # 포지션 청산 시 손익 계산
        realized_pnl = 0.0
        if abs(self.position) > 0.0001:
            # 기존 포지션과 반대 방향 거래 = 청산
            if np.sign(position_change) != np.sign(self.position):
                close_size = min(abs(position_change), abs(self.position))
                price_diff = (execution_price - self.entry_price) * np.sign(self.position)
                realized_pnl = close_size * price_diff * self.leverage - fee

                self.balance += realized_pnl
                self.total_pnl += realized_pnl

                # 승/패 카운트
                self.trade_count += 1
                if realized_pnl > 0:
                    self.win_count += 1

        # 새 포지션 진입
        remaining_change = position_change
        if abs(self.position) > 0.0001 and np.sign(position_change) != np.sign(self.position):
            remaining_change = position_change + self.position

        if abs(remaining_change) > 0.0001:
            # 필요 증거금
            required_margin = abs(remaining_change) * execution_price
            if required_margin <= self.balance:
                self.position += remaining_change
                self.entry_price = execution_price
                self.balance -= fee
            else:
                logger.warning("Insufficient balance for trade")

        return realized_pnl

    def _set_stop_conditions(
        self,
        current_price: float,
        stop_loss_pct: float,
        take_profit_pct: float
    ) -> None:
        """
        손절/익절 설정

        Args:
            current_price: 현재 가격
            stop_loss_pct: 손절 비율
            take_profit_pct: 익절 비율
        """
        if self.position > 0:  # 롱 포지션
            self.stop_loss = current_price * (1 - stop_loss_pct)
            self.take_profit = current_price * (1 + take_profit_pct)
        elif self.position < 0:  # 숏 포지션
            self.stop_loss = current_price * (1 + stop_loss_pct)
            self.take_profit = current_price * (1 - take_profit_pct)

    def _check_stop_conditions(self, current_price: float) -> float:
        """
        손절/익절 조건 확인

        Args:
            current_price: 현재 가격

        Returns:
            청산 손익
        """
        if abs(self.position) < 0.0001:
            return 0.0

        closed_pnl = 0.0

        # 롱 포지션
        if self.position > 0:
            if current_price <= self.stop_loss or current_price >= self.take_profit:
                closed_pnl = self._execute_trade(-self.position, current_price)
                logger.debug(f"Long position closed at {current_price}, PnL: {closed_pnl:.2f}")

        # 숏 포지션
        elif self.position < 0:
            if current_price >= self.stop_loss or current_price <= self.take_profit:
                closed_pnl = self._execute_trade(-self.position, current_price)
                logger.debug(f"Short position closed at {current_price}, PnL: {closed_pnl:.2f}")

        return closed_pnl

    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """
        미실현 손익 계산

        Args:
            current_price: 현재 가격

        Returns:
            미실현 손익
        """
        if abs(self.position) < 0.0001:
            return 0.0

        price_diff = (current_price - self.entry_price) * np.sign(self.position)
        unrealized_pnl = abs(self.position) * price_diff * self.leverage

        return unrealized_pnl

    def _calculate_portfolio_value(self) -> float:
        """
        포트폴리오 가치 계산

        Returns:
            총 계정 가치
        """
        current_price = self._get_current_price()
        unrealized_pnl = self._calculate_unrealized_pnl(current_price)
        portfolio_value = self.balance + unrealized_pnl

        return portfolio_value

    def _calculate_reward(
        self,
        prev_value: float,
        new_value: float,
        trade_pnl: float
    ) -> float:
        """
        보상 계산

        Args:
            prev_value: 이전 포트폴리오 가치
            new_value: 현재 포트폴리오 가치
            trade_pnl: 거래 손익

        Returns:
            보상 값
        """
        # 포트폴리오 가치 변화 기반 보상
        portfolio_return = (new_value - prev_value) / prev_value

        # 거래 손익 기반 보상
        trade_reward = trade_pnl / self.initial_balance if abs(trade_pnl) > 0.0001 else 0.0

        # 리스크 조정 보상 (변동성 페널티)
        volatility_penalty = 0.0
        if abs(self.position) > 0.5 * self.max_position_size:
            volatility_penalty = -0.01

        # 총 보상
        total_reward = (portfolio_return + trade_reward + volatility_penalty) * self.reward_scaling

        # 파산 페널티
        if self.balance <= 0:
            total_reward -= 10.0

        return total_reward

    def render(self, mode='human'):
        """환경 렌더링 (선택)"""
        if mode == 'human':
            print(f"Step: {self.current_step}, "
                  f"Balance: {self.balance:.2f}, "
                  f"Position: {self.position:.4f}, "
                  f"PnL: {self.total_pnl:.2f}")

    def close(self):
        """환경 종료"""
        pass
