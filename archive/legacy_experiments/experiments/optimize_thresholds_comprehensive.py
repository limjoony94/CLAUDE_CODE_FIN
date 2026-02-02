"""
파라미터 최적화: SHORT Threshold & LONG TP 백테스팅
목표: 페이퍼 트레이딩 대신 빠른 백테스팅으로 최적값 찾기
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from loguru import logger
from scipy import stats

# 로깅 설정
logger.remove()
logger.add(sys.stdout,
          format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>",
          level="INFO")

class ThresholdOptimizer:
    def __init__(self):
        # 모델 로드
        self.load_models()

        # 데이터 로드
        self.load_data()

        # 결과 저장
        self.results = []

    def load_models(self):
        """모델 로드"""
        logger.info("📦 모델 로딩 중...")

        # LONG 모델 (Phase 4)
        long_model_path = "models/xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
        if os.path.exists(long_model_path):
            self.long_model = joblib.load(long_model_path)
            logger.info("✅ LONG 모델 로드 완료")
        else:
            raise FileNotFoundError(f"LONG 모델 없음: {long_model_path}")

        # SHORT 모델 (최적화된 버전 사용)
        short_model_path = "models/xgboost_v4_short_optimized_20251010_235955.pkl"
        if not os.path.exists(short_model_path):
            # Fallback to 3-class model
            short_model_path = "models/xgboost_v4_phase4_3class_lookahead3_thresh3.pkl"

        if os.path.exists(short_model_path):
            self.short_model = joblib.load(short_model_path)
            logger.info(f"✅ SHORT 모델 로드 완료: {os.path.basename(short_model_path)}")

            # 모델 타입 확인
            if hasattr(self.short_model, 'classes_'):
                self.short_is_3class = len(self.short_model.classes_) == 3
            else:
                self.short_is_3class = False
        else:
            raise FileNotFoundError(f"SHORT 모델 없음: {short_model_path}")

    def load_data(self):
        """데이터 로드"""
        logger.info("📊 데이터 로딩 중...")

        data_path = "data/historical/BTCUSDT_5m_max.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"데이터 없음: {data_path}")

        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # 최근 30일 데이터만 사용 (페이퍼 트레이딩 기간과 유사)
        recent_days = 30
        cutoff_date = df['timestamp'].max() - pd.Timedelta(days=recent_days)
        self.df = df[df['timestamp'] >= cutoff_date].reset_index(drop=True)

        logger.info(f"✅ 데이터 로드 완료: {len(self.df)} rows ({recent_days}일)")
        logger.info(f"📅 기간: {self.df['timestamp'].min()} ~ {self.df['timestamp'].max()}")

    def add_features(self, df):
        """기술적 지표 추가 (Phase 4 features)"""
        from src.indicators.technical_indicators import add_technical_indicators

        df = add_technical_indicators(df)

        # NaN 제거
        df = df.dropna()

        return df

    def backtest_long(self, threshold=0.7, tp_pct=1.5, sl_pct=1.0, max_hold_hours=4):
        """LONG 백테스팅"""
        df = self.add_features(self.df.copy())

        # LONG 예측
        feature_cols = [col for col in df.columns if col not in
                       ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        X = df[feature_cols]

        long_probs = self.long_model.predict_proba(X)[:, 1]
        df['long_prob'] = long_probs

        # 거래 시뮬레이션
        trades = []
        position = None
        initial_capital = 10000
        capital = initial_capital

        for i in range(len(df)):
            current_row = df.iloc[i]

            # 포지션 보유 중
            if position is not None:
                entry_time, entry_price, entry_idx = position
                current_price = current_row['close']
                hours_held = (current_row['timestamp'] - entry_time).total_seconds() / 3600

                pnl_pct = (current_price - entry_price) / entry_price * 100

                # 청산 조건
                exit_reason = None
                if pnl_pct >= tp_pct:
                    exit_reason = "TP"
                elif pnl_pct <= -sl_pct:
                    exit_reason = "SL"
                elif hours_held >= max_hold_hours:
                    exit_reason = "Max Hold"

                if exit_reason:
                    # 거래 완료
                    capital = capital * (1 + pnl_pct / 100)

                    trades.append({
                        'direction': 'LONG',
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_time': current_row['timestamp'],
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'hours_held': hours_held
                    })

                    position = None

            # 새 포지션 진입
            elif current_row['long_prob'] >= threshold:
                position = (current_row['timestamp'], current_row['close'], i)

        # 통계 계산
        if len(trades) == 0:
            return None

        trades_df = pd.DataFrame(trades)
        win_rate = (trades_df['pnl_pct'] > 0).mean() * 100
        avg_pnl = trades_df['pnl_pct'].mean()
        total_return = (capital - initial_capital) / initial_capital * 100

        return {
            'threshold': threshold,
            'tp_pct': tp_pct,
            'sl_pct': sl_pct,
            'max_hold_hours': max_hold_hours,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_return': total_return,
            'final_capital': capital,
            'trades': trades_df
        }

    def backtest_short(self, threshold=0.4, tp_pct=3.0, sl_pct=1.5, max_hold_hours=4):
        """SHORT 백테스팅"""
        df = self.add_features(self.df.copy())

        # SHORT 예측
        feature_cols = [col for col in df.columns if col not in
                       ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        X = df[feature_cols]

        # 모델 타입에 따라 확률 추출
        if self.short_is_3class:
            short_probs = self.short_model.predict_proba(X)[:, 2]  # Class 2 = SHORT
        else:
            # Binary model인 경우
            short_probs = self.short_model.predict_proba(X)[:, 1]

        df['short_prob'] = short_probs

        # 거래 시뮬레이션
        trades = []
        position = None
        initial_capital = 10000
        capital = initial_capital

        for i in range(len(df)):
            current_row = df.iloc[i]

            # 포지션 보유 중
            if position is not None:
                entry_time, entry_price, entry_idx = position
                current_price = current_row['close']
                hours_held = (current_row['timestamp'] - entry_time).total_seconds() / 3600

                # SHORT이므로 가격 하락이 수익
                pnl_pct = (entry_price - current_price) / entry_price * 100

                # 청산 조건
                exit_reason = None
                if pnl_pct >= tp_pct:
                    exit_reason = "TP"
                elif pnl_pct <= -sl_pct:
                    exit_reason = "SL"
                elif hours_held >= max_hold_hours:
                    exit_reason = "Max Hold"

                if exit_reason:
                    # 거래 완료
                    capital = capital * (1 + pnl_pct / 100)

                    trades.append({
                        'direction': 'SHORT',
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_time': current_row['timestamp'],
                        'exit_price': current_price,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'hours_held': hours_held
                    })

                    position = None

            # 새 포지션 진입
            elif current_row['short_prob'] >= threshold:
                position = (current_row['timestamp'], current_row['close'], i)

        # 통계 계산
        if len(trades) == 0:
            return None

        trades_df = pd.DataFrame(trades)
        win_rate = (trades_df['pnl_pct'] > 0).mean() * 100
        avg_pnl = trades_df['pnl_pct'].mean()
        total_return = (capital - initial_capital) / initial_capital * 100

        return {
            'threshold': threshold,
            'tp_pct': tp_pct,
            'sl_pct': sl_pct,
            'max_hold_hours': max_hold_hours,
            'num_trades': len(trades),
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_return': total_return,
            'final_capital': capital,
            'trades': trades_df
        }

    def optimize_short_threshold(self):
        """SHORT threshold 최적화"""
        logger.info("\n" + "="*80)
        logger.info("🔴 SHORT THRESHOLD 최적화")
        logger.info("="*80)

        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        results = []

        for thresh in thresholds:
            logger.info(f"\n📊 테스트 중: Threshold {thresh}")

            result = self.backtest_short(threshold=thresh)

            if result:
                results.append(result)

                logger.info(f"   거래 수: {result['num_trades']}")
                logger.info(f"   승률: {result['win_rate']:.1f}%")
                logger.info(f"   평균 P&L: {result['avg_pnl']:.2f}%")
                logger.info(f"   총 수익률: {result['total_return']:.2f}%")
            else:
                logger.warning(f"   ⚠️ 거래 없음")

        # 최적값 선택 (승률 * 총수익률 최대화)
        if results:
            results_df = pd.DataFrame(results)
            results_df['score'] = results_df['win_rate'] * results_df['total_return']
            best = results_df.loc[results_df['score'].idxmax()]

            logger.info(f"\n🏆 최적 SHORT Threshold: {best['threshold']}")
            logger.info(f"   승률: {best['win_rate']:.1f}%")
            logger.info(f"   총 수익률: {best['total_return']:.2f}%")
            logger.info(f"   거래 수: {int(best['num_trades'])}")

            return results_df, best
        else:
            return None, None

    def optimize_long_tp(self):
        """LONG TP 최적화"""
        logger.info("\n" + "="*80)
        logger.info("🟢 LONG TP 최적화")
        logger.info("="*80)

        tp_values = [1.0, 1.5, 2.0, 2.5]
        results = []

        for tp in tp_values:
            logger.info(f"\n📊 테스트 중: TP {tp}%")

            result = self.backtest_long(tp_pct=tp)

            if result:
                results.append(result)

                logger.info(f"   거래 수: {result['num_trades']}")
                logger.info(f"   승률: {result['win_rate']:.1f}%")
                logger.info(f"   평균 P&L: {result['avg_pnl']:.2f}%")
                logger.info(f"   총 수익률: {result['total_return']:.2f}%")
            else:
                logger.warning(f"   ⚠️ 거래 없음")

        # 최적값 선택
        if results:
            results_df = pd.DataFrame(results)
            results_df['score'] = results_df['win_rate'] * results_df['total_return']
            best = results_df.loc[results_df['score'].idxmax()]

            logger.info(f"\n🏆 최적 LONG TP: {best['tp_pct']:.1f}%")
            logger.info(f"   승률: {best['win_rate']:.1f}%")
            logger.info(f"   총 수익률: {best['total_return']:.2f}%")
            logger.info(f"   거래 수: {int(best['num_trades'])}")

            return results_df, best
        else:
            return None, None

    def run_optimization(self):
        """전체 최적화 실행"""
        logger.info("\n🚀 파라미터 최적화 시작")
        logger.info(f"📅 데이터 기간: {self.df['timestamp'].min()} ~ {self.df['timestamp'].max()}")
        logger.info(f"📊 데이터 포인트: {len(self.df)}")

        # SHORT threshold 최적화
        short_results, best_short = self.optimize_short_threshold()

        # LONG TP 최적화
        long_results, best_long = self.optimize_long_tp()

        # 최종 요약
        logger.info("\n" + "="*80)
        logger.info("🎯 최적화 완료 - 최종 권장값")
        logger.info("="*80)

        if best_short is not None:
            logger.info(f"\n🔴 SHORT:")
            logger.info(f"   Threshold: {best_short['threshold']} (현재: 0.4)")
            logger.info(f"   예상 승률: {best_short['win_rate']:.1f}% (페이퍼: ~30%)")
            logger.info(f"   예상 수익률: {best_short['total_return']:.2f}%/30일")

        if best_long is not None:
            logger.info(f"\n🟢 LONG:")
            logger.info(f"   TP: {best_long['tp_pct']:.1f}% (현재: 1.5%)")
            logger.info(f"   예상 승률: {best_long['win_rate']:.1f}% (페이퍼: ~33%)")
            logger.info(f"   예상 수익률: {best_long['total_return']:.2f}%/30일")

        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if short_results is not None:
            short_results.to_csv(f"claudedocs/optimization_short_{timestamp}.csv", index=False)
            logger.info(f"\n💾 SHORT 결과 저장: claudedocs/optimization_short_{timestamp}.csv")

        if long_results is not None:
            long_results.to_csv(f"claudedocs/optimization_long_{timestamp}.csv", index=False)
            logger.info(f"💾 LONG 결과 저장: claudedocs/optimization_long_{timestamp}.csv")

        return {
            'short': {'results': short_results, 'best': best_short},
            'long': {'results': long_results, 'best': best_long}
        }


if __name__ == "__main__":
    optimizer = ThresholdOptimizer()
    results = optimizer.run_optimization()

    logger.info("\n✅ 최적화 완료!")
    logger.info("\n📋 다음 단계:")
    logger.info("   1. 위 권장값을 프로덕션 봇에 적용")
    logger.info("   2. 소액 실전 테스트 ($100-200)")
    logger.info("   3. n=20-30 데이터 수집")
    logger.info("   4. 검증 완료 후 본격 운영")
