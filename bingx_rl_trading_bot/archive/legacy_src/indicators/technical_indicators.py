"""기술적 지표 계산"""

from typing import Dict, Any

import numpy as np
import pandas as pd
from loguru import logger


class TechnicalIndicators:
    """
    스캘핑 전략에 최적화된 기술적 지표
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Args:
            config: 지표 설정 딕셔너리
        """
        self.config = config or self._default_config()

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """기본 지표 설정"""
        return {
            'ema': {'periods': [9, 21, 50]},
            'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
            'bollinger_bands': {'period': 20, 'std': 2},
            'stochastic': {'k_period': 14, 'd_period': 3},
            'adx': {'period': 14},
            'atr': {'period': 14},
            'vwap': {'enabled': True}
        }

    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 기술적 지표 계산

        Args:
            df: OHLCV 데이터프레임

        Returns:
            지표가 추가된 데이터프레임
        """
        df = df.copy()

        # EMA
        df = self.calculate_ema(df)

        # RSI
        df = self.calculate_rsi(df)

        # Bollinger Bands
        df = self.calculate_bollinger_bands(df)

        # Stochastic
        df = self.calculate_stochastic(df)

        # ADX
        df = self.calculate_adx(df)

        # ATR
        df = self.calculate_atr(df)

        # VWAP
        if self.config['vwap']['enabled']:
            df = self.calculate_vwap(df)

        # MACD (추가)
        df = self.calculate_macd(df)

        # 결측치 제거
        df = df.dropna()

        logger.info(f"Calculated all indicators for {len(df)} rows")
        return df

    def calculate_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        지수 이동평균 (EMA)

        Args:
            df: 데이터프레임

        Returns:
            EMA가 추가된 데이터프레임
        """
        df = df.copy()
        periods = self.config['ema']['periods']

        for period in periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        return df

    def calculate_rsi(self, df: pd.DataFrame, period: int = None) -> pd.DataFrame:
        """
        상대강도지수 (RSI)

        Args:
            df: 데이터프레임
            period: RSI 기간

        Returns:
            RSI가 추가된 데이터프레임
        """
        df = df.copy()
        if period is None:
            period = self.config['rsi']['period']

        # 가격 변화
        delta = df['close'].diff()

        # 상승/하락 분리
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # 평균 상승/하락
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # RSI 계산
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 과매수/과매도 시그널
        df['rsi_overbought'] = df['rsi'] > self.config['rsi']['overbought']
        df['rsi_oversold'] = df['rsi'] < self.config['rsi']['oversold']

        return df

    def calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        볼린저 밴드

        Args:
            df: 데이터프레임

        Returns:
            볼린저 밴드가 추가된 데이터프레임
        """
        df = df.copy()
        period = self.config['bollinger_bands']['period']
        std_multiplier = self.config['bollinger_bands']['std']

        # 중심선 (SMA)
        df['bb_middle'] = df['close'].rolling(window=period).mean()

        # 표준편차
        std = df['close'].rolling(window=period).std()

        # 상단/하단 밴드
        df['bb_upper'] = df['bb_middle'] + (std * std_multiplier)
        df['bb_lower'] = df['bb_middle'] - (std * std_multiplier)

        # 밴드 폭
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # 가격 위치 (%B)
        df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        return df

    def calculate_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        스토캐스틱 오실레이터

        Args:
            df: 데이터프레임

        Returns:
            스토캐스틱이 추가된 데이터프레임
        """
        df = df.copy()
        k_period = self.config['stochastic']['k_period']
        d_period = self.config['stochastic']['d_period']

        # %K 계산
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()

        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)

        # %D 계산 (K의 이동평균)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()

        return df

    def calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        평균방향지수 (ADX)

        Args:
            df: 데이터프레임

        Returns:
            ADX가 추가된 데이터프레임
        """
        df = df.copy()
        period = self.config['adx']['period']

        # True Range 계산
        df['tr'] = self._calculate_true_range(df)

        # +DM, -DM 계산
        df['high_diff'] = df['high'].diff()
        df['low_diff'] = -df['low'].diff()

        df['plus_dm'] = np.where(
            (df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0),
            df['high_diff'],
            0
        )

        df['minus_dm'] = np.where(
            (df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0),
            df['low_diff'],
            0
        )

        # 평활화
        df['tr_smooth'] = df['tr'].rolling(window=period).mean()
        df['plus_dm_smooth'] = df['plus_dm'].rolling(window=period).mean()
        df['minus_dm_smooth'] = df['minus_dm'].rolling(window=period).mean()

        # +DI, -DI 계산
        df['plus_di'] = 100 * df['plus_dm_smooth'] / df['tr_smooth']
        df['minus_di'] = 100 * df['minus_dm_smooth'] / df['tr_smooth']

        # DX 계산
        df['dx'] = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])

        # ADX 계산
        df['adx'] = df['dx'].rolling(window=period).mean()

        # 임시 컬럼 제거
        df = df.drop(columns=[
            'tr', 'high_diff', 'low_diff', 'plus_dm', 'minus_dm',
            'tr_smooth', 'plus_dm_smooth', 'minus_dm_smooth', 'dx'
        ])

        return df

    def calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        평균 실제 범위 (ATR)

        Args:
            df: 데이터프레임

        Returns:
            ATR이 추가된 데이터프레임
        """
        df = df.copy()
        period = self.config['atr']['period']

        # True Range
        tr = self._calculate_true_range(df)

        # ATR (True Range의 이동평균)
        df['atr'] = tr.rolling(window=period).mean()

        # ATR 퍼센트 (변동성 측정)
        df['atr_percent'] = (df['atr'] / df['close']) * 100

        return df

    def calculate_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        거래량 가중 평균가 (VWAP)

        Args:
            df: 데이터프레임

        Returns:
            VWAP이 추가된 데이터프레임
        """
        df = df.copy()

        # 전형가격 (Typical Price)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        # VWAP 계산
        df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()

        # VWAP과 가격 차이
        df['vwap_diff'] = ((df['close'] - df['vwap']) / df['vwap']) * 100

        df = df.drop(columns=['typical_price'])

        return df

    def calculate_macd(
        self,
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> pd.DataFrame:
        """
        MACD (Moving Average Convergence Divergence)

        Args:
            df: 데이터프레임
            fast_period: 빠른 EMA 기간
            slow_period: 느린 EMA 기간
            signal_period: 시그널 기간

        Returns:
            MACD가 추가된 데이터프레임
        """
        df = df.copy()

        # EMA 계산
        ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()

        # MACD 라인
        df['macd'] = ema_fast - ema_slow

        # 시그널 라인
        df['macd_signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()

        # 히스토그램
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        return df

    def _calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """
        True Range 계산

        Args:
            df: 데이터프레임

        Returns:
            True Range 시리즈
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        return true_range

    def calculate_sequential_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sequential/Context Features 추가 - 추세 및 문맥 인식

        사용자 통찰: "모델이 가장 최근 캔들의 지표만 보고 추세를 모른다"
        해결: 변화, 추세, 패턴 피처 추가

        Args:
            df: 기본 지표가 계산된 데이터프레임

        Returns:
            Sequential features가 추가된 데이터프레임
        """
        df = df.copy()

        # 1. Trend Context Features
        # RSI 변화
        if 'rsi' in df.columns:
            df['rsi_change_5'] = df['rsi'].diff(5)
            df['rsi_change_20'] = df['rsi'].diff(20)

        # 가격 vs 이동평균 비율
        if 'bb_middle' in df.columns:  # SMA20
            df['price_vs_sma20'] = (df['close'] - df['bb_middle']) / df['bb_middle']

        if 'ema_50' in df.columns:
            df['price_vs_ema50'] = (df['close'] - df['ema_50']) / df['ema_50']

        # 이동평균 배열
        if 'ema_9' in df.columns and 'ema_21' in df.columns:
            df['ema9_vs_ema21'] = (df['ema_9'] - df['ema_21']) / df['ema_21']

        # 2. Momentum Indicators
        # Volume 변화
        df['volume_change_5'] = df['volume'] / df['volume'].rolling(5).mean()
        df['volume_change_20'] = df['volume'] / df['volume'].rolling(20).mean()

        # Volatility 변화 (ATR)
        if 'atr' in df.columns:
            df['atr_change'] = df['atr'] / df['atr'].shift(10)

        # MACD Histogram 변화
        if 'macd_histogram' in df.columns:
            df['macd_hist_change'] = df['macd_histogram'].diff(5)
            df['macd_cross'] = ((df['macd'] > df['macd_signal']) &
                               (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)

        # 3. Pattern Features
        # 연속 상승/하락 캔들
        df['up_candle'] = (df['close'] > df['open']).astype(int)
        df['down_candle'] = (df['close'] < df['open']).astype(int)

        df['consecutive_up'] = df['up_candle'].rolling(5).sum()
        df['consecutive_down'] = df['down_candle'].rolling(5).sum()

        # Higher Highs / Lower Lows
        df['higher_high'] = ((df['high'] > df['high'].shift(1)) &
                            (df['high'].shift(1) > df['high'].shift(2))).astype(int)
        df['lower_low'] = ((df['low'] < df['low'].shift(1)) &
                          (df['low'].shift(1) < df['low'].shift(2))).astype(int)

        # 4. Sequence Statistics
        # Price 변동성 (rolling std)
        df['price_std_10'] = df['close'].pct_change().rolling(10).std()
        df['price_std_50'] = df['close'].pct_change().rolling(50).std()

        # Return autocorrelation (lag-1)
        returns = df['close'].pct_change()
        df['return_autocorr'] = returns.rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
        )

        # 5. Multi-Timeframe Context
        # 1시간 평균과 비교 (12 candles = 1 hour)
        df['price_vs_1h_avg'] = df['close'] / df['close'].rolling(12).mean()

        # 4시간 평균과 비교 (48 candles = 4 hours)
        df['price_vs_4h_avg'] = df['close'] / df['close'].rolling(48).mean()

        # Trend alignment (모든 MA가 같은 방향)
        if 'ema_9' in df.columns and 'ema_21' in df.columns and 'ema_50' in df.columns:
            df['trend_alignment'] = (
                ((df['ema_9'] > df['ema_21']) & (df['ema_21'] > df['ema_50'])).astype(int) -
                ((df['ema_9'] < df['ema_21']) & (df['ema_21'] < df['ema_50'])).astype(int)
            )

        # 임시 컬럼 제거
        df = df.drop(columns=['up_candle', 'down_candle'], errors='ignore')

        logger.info(f"Added sequential features. Total features: {len(df.columns)}")

        return df

    def get_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        지표 기반 거래 시그널 생성

        Args:
            df: 지표가 계산된 데이터프레임

        Returns:
            시그널이 추가된 데이터프레임
        """
        df = df.copy()

        # 초기화
        df['signal_strength'] = 0.0
        df['trend_signal'] = 0  # 1: 상승, -1: 하락, 0: 중립

        # EMA 추세 시그널
        if 'ema_9' in df.columns and 'ema_21' in df.columns:
            df['ema_bullish'] = df['ema_9'] > df['ema_21']
            df['ema_bearish'] = df['ema_9'] < df['ema_21']
            df.loc[df['ema_bullish'], 'signal_strength'] += 0.2
            df.loc[df['ema_bearish'], 'signal_strength'] -= 0.2

        # RSI 시그널
        if 'rsi' in df.columns:
            df.loc[df['rsi_oversold'], 'signal_strength'] += 0.3
            df.loc[df['rsi_overbought'], 'signal_strength'] -= 0.3

        # MACD 시그널
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df['macd_bullish'] = df['macd'] > df['macd_signal']
            df['macd_bearish'] = df['macd'] < df['macd_signal']
            df.loc[df['macd_bullish'], 'signal_strength'] += 0.2
            df.loc[df['macd_bearish'], 'signal_strength'] -= 0.2

        # ADX 추세 강도
        if 'adx' in df.columns:
            df['strong_trend'] = df['adx'] > 25
            df.loc[df['strong_trend'], 'signal_strength'] *= 1.2

        # 종합 추세 시그널
        df.loc[df['signal_strength'] > 0.3, 'trend_signal'] = 1
        df.loc[df['signal_strength'] < -0.3, 'trend_signal'] = -1

        return df
