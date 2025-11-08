"""근본 원인 심층 분석"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.indicators.technical_indicators import TechnicalIndicators
from src.data.data_processor import DataProcessor
from loguru import logger


def analyze_market_characteristics():
    """시장 특성 분석"""

    print("="*80)
    print("근본 원인 심층 분석")
    print("="*80)

    # 데이터 로드
    df = pd.read_csv('data/historical/BTCUSDT_5m_max.csv', parse_dates=['timestamp'])

    # 지표 계산
    indicator_calc = TechnicalIndicators()
    df = indicator_calc.calculate_all_indicators(df)

    # 전처리
    processor = DataProcessor()
    df = processor.prepare_data(df, fit=True)

    # 분할
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]

    print(f"\n📊 데이터 분할")
    print(f"훈련: {len(train_df):,} ({train_df['timestamp'].min()} ~ {train_df['timestamp'].max()})")
    print(f"검증: {len(val_df):,} ({val_df['timestamp'].min()} ~ {val_df['timestamp'].max()})")
    print(f"테스트: {len(test_df):,} ({test_df['timestamp'].min()} ~ {test_df['timestamp'].max()})")

    print("\n" + "="*80)
    print("1. 시장 변동성 분석")
    print("="*80)

    for name, data in [('훈련', train_df), ('검증', val_df), ('테스트', test_df)]:
        returns = data['close'].pct_change()
        volatility = returns.std() * np.sqrt(288)  # 일일 변동성 (5분×288=1일)
        trend = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]

        print(f"\n{name} 세트:")
        print(f"  변동성 (일일): {volatility*100:.2f}%")
        print(f"  전체 추세: {trend*100:+.2f}%")
        print(f"  가격 범위: ${data['close'].min():,.0f} ~ ${data['close'].max():,.0f}")
        print(f"  평균 가격: ${data['close'].mean():,.0f}")

    print("\n" + "="*80)
    print("2. 기술적 지표 분포 차이")
    print("="*80)

    indicators = ['rsi', 'macd', 'bb_percent']

    for ind in indicators:
        if ind in df.columns:
            print(f"\n{ind.upper()}:")
            print(f"  훈련: 평균={train_df[ind].mean():.2f}, 표준편차={train_df[ind].std():.2f}")
            print(f"  검증: 평균={val_df[ind].mean():.2f}, 표준편차={val_df[ind].std():.2f}")
            print(f"  테스트: 평균={test_df[ind].mean():.2f}, 표준편차={test_df[ind].std():.2f}")

    print("\n" + "="*80)
    print("3. 거래 기회 분석")
    print("="*80)

    for name, data in [('훈련', train_df), ('검증', val_df), ('테스트', test_df)]:
        # 강한 추세 비율
        data['strong_trend'] = 0
        if 'adx' in data.columns:
            data['strong_trend'] = (data['adx'] > 25).astype(int)

        # RSI 과매수/과매도
        overbought = (data['rsi'] > 70).sum()
        oversold = (data['rsi'] < 30).sum()

        print(f"\n{name} 세트:")
        print(f"  강한 추세: {data['strong_trend'].sum()}/{len(data)} ({data['strong_trend'].mean()*100:.1f}%)")
        print(f"  RSI 과매수: {overbought} ({overbought/len(data)*100:.1f}%)")
        print(f"  RSI 과매도: {oversold} ({oversold/len(data)*100:.1f}%)")

    print("\n" + "="*80)
    print("4. 데이터 충분성 분석")
    print("="*80)

    # 에피소드당 필요 데이터
    episode_length = len(train_df)
    episodes_per_rollout = 2048 / episode_length
    total_rollouts = 5000000 / 2048

    print(f"\n에피소드 길이: {episode_length:,} 스텝")
    print(f"롤아웃당 에피소드: {episodes_per_rollout:.2f}")
    print(f"총 롤아웃 수: {total_rollouts:.0f}")
    print(f"총 에피소드 수: {total_rollouts * episodes_per_rollout:.0f}")
    print(f"\n→ 같은 12,064개 데이터를 {total_rollouts * episodes_per_rollout:.0f}번 반복 학습")
    print(f"→ 과적합 위험: 매우 높음")

    print("\n" + "="*80)
    print("5. 수수료 영향 분석")
    print("="*80)

    # 수수료 시뮬레이션
    transaction_fee = 0.0004
    slippage = 0.0001
    total_cost = transaction_fee + slippage

    avg_price = df['close'].mean()
    position_size = 0.03  # BTC
    leverage = 3

    trade_value = position_size * avg_price * leverage
    cost_per_trade = trade_value * total_cost

    # 손익분기 움직임
    breakeven_move = total_cost * 100

    print(f"\n거래당 비용:")
    print(f"  포지션 가치: ${trade_value:,.2f}")
    print(f"  수수료+슬리피지: ${cost_per_trade:.2f} ({total_cost*100:.03f}%)")
    print(f"  손익분기 가격 변동: {breakeven_move:.03f}%")
    print(f"\n→ 5분 평균 변동: {df['close'].pct_change().abs().mean()*100:.03f}%")
    print(f"→ 거래 빈도가 높으면 수수료에 잠식됨")

    print("\n" + "="*80)
    print("6. 과적합 지표")
    print("="*80)

    validation_score = 58.87
    test_score = -105.22
    generalization_gap = validation_score - test_score
    overfitting_ratio = abs(generalization_gap / validation_score) if validation_score != 0 else float('inf')

    print(f"\n검증 성능: {validation_score:+.2f}")
    print(f"테스트 성능: {test_score:+.2f}")
    print(f"일반화 격차: {generalization_gap:.2f}")
    print(f"과적합 비율: {overfitting_ratio:.1f}x")
    print(f"\n판정: {'심각한 과적합' if overfitting_ratio > 2 else '과적합' if overfitting_ratio > 1 else '정상'}")

    print("\n" + "="*80)
    print("7. 데이터 다양성 분석")
    print("="*80)

    # 가격 변화 패턴
    for name, data in [('훈련', train_df), ('검증', val_df), ('테스트', test_df)]:
        up_days = (data['close'].diff() > 0).sum()
        down_days = (data['close'].diff() < 0).sum()

        price_change = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100

        print(f"\n{name} 세트:")
        print(f"  상승 캔들: {up_days} ({up_days/(up_days+down_days)*100:.1f}%)")
        print(f"  하락 캔들: {down_days} ({down_days/(up_days+down_days)*100:.1f}%)")
        print(f"  누적 변화: {price_change:+.2f}%")

    print("\n" + "="*80)
    print("핵심 발견 요약")
    print("="*80)

    print("\n🔍 근본 원인:")
    print("1. 데이터 부족 - 60일은 다양한 시장 상황 커버 불가")
    print("2. 과도한 반복 - 같은 데이터 수천 번 반복 → 암기")
    print("3. 시장 체제 변화 - 각 기간의 시장 특성이 다름")
    print("4. 수수료 압박 - 잦은 거래 시 수수료가 수익 잠식")
    print("5. 검증 세트 과적합 - Best Model 선택 기준이 9일 데이터")


if __name__ == "__main__":
    analyze_market_characteristics()
