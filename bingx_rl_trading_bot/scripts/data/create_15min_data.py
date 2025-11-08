"""5분봉 데이터를 15분봉으로 리샘플링"""

import pandas as pd
from pathlib import Path
from loguru import logger


def main():
    logger.info("="*70)
    logger.info("5분봉 → 15분봉 리샘플링")
    logger.info("="*70)

    # 5분봉 데이터 로드
    input_file = Path('data/historical/BTCUSDT_5m_max.csv')
    output_file = Path('data/historical/BTCUSDT_15m.csv')

    logger.info(f"Loading: {input_file}")
    df_5m = pd.read_csv(input_file, parse_dates=['timestamp'])

    logger.info(f"5분봉 데이터: {len(df_5m):,} 캔들")
    logger.info(f"기간: {df_5m['timestamp'].min()} ~ {df_5m['timestamp'].max()}")

    # timestamp를 인덱스로 설정
    df_5m = df_5m.set_index('timestamp')

    # 15분봉으로 리샘플링
    logger.info("\n리샘플링 중...")

    df_15m = df_5m.resample('15T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    # NaN 제거 (시장 휴장 등)
    df_15m = df_15m.dropna()

    # 인덱스를 컬럼으로 복원
    df_15m = df_15m.reset_index()

    logger.info(f"\n15분봉 데이터: {len(df_15m):,} 캔들")
    logger.info(f"기간: {df_15m['timestamp'].min()} ~ {df_15m['timestamp'].max()}")

    # 통계 비교
    logger.info(f"\n=== 통계 비교 ===")
    logger.info(f"5분봉:")
    logger.info(f"  캔들 수: {len(df_5m):,}")
    logger.info(f"  평균 거래량: {df_5m['volume'].mean():,.2f}")

    logger.info(f"\n15분봉:")
    logger.info(f"  캔들 수: {len(df_15m):,}")
    logger.info(f"  평균 거래량: {df_15m['volume'].mean():,.2f}")

    # 저장
    logger.info(f"\n저장 중: {output_file}")
    df_15m.to_csv(output_file, index=False)

    logger.info(f"✅ 완료!")
    logger.info(f"\n사용 방법:")
    logger.info(f"  df = pd.read_csv('data/historical/BTCUSDT_15m.csv', parse_dates=['timestamp'])")

    logger.info("\n="*70)


if __name__ == "__main__":
    main()
