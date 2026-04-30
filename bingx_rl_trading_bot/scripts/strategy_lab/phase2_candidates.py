"""Phase 2 — Apply pipeline to 6 advisor-identified empty cells + auto-rank.

Each candidate is a structured proposal with:
  - mechanism × substrate × execution
  - hypothesized avg_gross / freq / regime dependence
  - user infrastructure constraints

Output:
  - Verdict per candidate (GO / CAUTION / BLOCK)
  - Top L1-L2 eligible ranked
  - PDCA Plan template for top candidate(s)
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from cross_checker import StrategyFeature
from reality_filter import StrategyProposal
from pipeline import evaluate_proposal, report_short


# ============================================================================
# 6 Empty cells (advisor-identified) + 4 additional creative combinations
# ============================================================================

CANDIDATES = [
    # ─────────────────────────────────────────────────────────
    # 1. Funding carry × dispersion-weighted multi-coin × weekly rebal
    # ─────────────────────────────────────────────────────────
    StrategyProposal(
        name='C1_dispersion_carry',
        feature=StrategyFeature(
            mechanism='funding_carry',
            substrate='multi_coin_crypto_8',
            execution='spot_perp_hedge',
            friction_profile='BingX_retail_maker',
            freq_tier='<1/day',
            timeframe_tier='daily',
            capital_tier='500-1500',
            directionality='pure_carry',
        ),
        # R13 equal-weight 2.85% APY, but dispersion-weighted top-2 widest funding
        # might recover R5 single-coin 3.28% performance with diversification benefit
        expected_avg_gross_pct_per_trade=0.05,  # weekly rebalance, ~50bp gross/week
        expected_trades_per_day=0.3,             # 2 coins × weekly = ~0.3/day
        capital_usd=500,
        user_infrastructure=['BingX'],
        bt_live_mappable=True,
        expected_regime_dependence=0.4,
        description='Top-2 widest 7d-funding-rate coins, weekly rebal, hedge with spot. Avoids R13 equal-weight dilution.',
    ),

    # ─────────────────────────────────────────────────────────
    # 2. Funding-z-score spread × intraday (BTC vs ETH)
    # ─────────────────────────────────────────────────────────
    StrategyProposal(
        name='C2_funding_zscore_spread',
        feature=StrategyFeature(
            mechanism='arbitrage',
            substrate='multi_coin_crypto_8',
            execution='spot_perp_hedge',
            friction_profile='BingX_retail_maker',
            freq_tier='<1/day',
            timeframe_tier='1h',
            capital_tier='500-1500',
            directionality='market_neutral',
        ),
        # When BTC funding rate diverges from ETH funding (z-score >2), enter the spread
        expected_avg_gross_pct_per_trade=0.08,
        expected_trades_per_day=0.5,
        capital_usd=500,
        user_infrastructure=['BingX'],
        bt_live_mappable=True,
        expected_regime_dependence=0.3,
        description='Long high-funding coin perp + short low-funding coin perp when z-score divergence > 2. Pair-trade harvesting funding gap.',
    ),

    # ─────────────────────────────────────────────────────────
    # 3. Cross-sectional momentum × hourly + sentiment-gated entry
    # ─────────────────────────────────────────────────────────
    StrategyProposal(
        name='C3_xs_momentum_sentiment_gated',
        feature=StrategyFeature(
            mechanism='cross_sectional',
            substrate='multi_coin_crypto_8',
            execution='maker_full_cycle',
            friction_profile='BingX_retail_maker',
            freq_tier='<1/day',
            timeframe_tier='1h',
            capital_tier='500-1500',
            directionality='market_neutral',
        ),
        # PathB-R1 was borderline (WF 3/5, bootstrap 48%). Sentiment overlay (only enter
        # when bullish overlay confirms) might lift to L2 stable.
        expected_avg_gross_pct_per_trade=0.10,
        expected_trades_per_day=0.4,
        capital_usd=500,
        user_infrastructure=['BingX'],
        bt_live_mappable=True,
        expected_regime_dependence=0.5,
        description='XS momentum (top-2 / bottom-2 8-coin) gated by aggregate sentiment (LSI / fear-greed / funding z-score). Skip entries in regime conflicts. Builds on PathB-R1.',
    ),

    # ─────────────────────────────────────────────────────────
    # 4. Volatility-state regime ensemble (carry + grid + MR rotation)
    # ─────────────────────────────────────────────────────────
    StrategyProposal(
        name='C4_regime_ensemble',
        feature=StrategyFeature(
            mechanism='session_time_filter',  # closest existing category
            substrate='BTC_perp_OHLCV_1h',
            execution='maker_taker_mixed',
            friction_profile='BingX_retail_maker',
            freq_tier='1-10/day',
            timeframe_tier='1h',
            capital_tier='500-1500',
            directionality='directional',  # mixed regime
        ),
        # Each sub-strategy marginal alone, ensemble might cover all regimes
        expected_avg_gross_pct_per_trade=0.06,
        expected_trades_per_day=2.0,
        capital_usd=500,
        user_infrastructure=['BingX'],
        bt_live_mappable=False,  # complex orchestration → Gate 1 difficult
        expected_regime_dependence=0.3,  # ensemble reduces single regime dependence
        description='Regime detector → activates carry (high-funding) / grid (low-vol ranging) / MR (mid-vol) sub-strategies. R26 grid bug NOT replicated (re-arm modeled).',
    ),

    # ─────────────────────────────────────────────────────────
    # 5. Basis × momentum filter × BingX (CFB 2025 paper)
    # ─────────────────────────────────────────────────────────
    StrategyProposal(
        name='C5_basis_momentum_filter',
        feature=StrategyFeature(
            mechanism='funding_carry',
            substrate='BTC_spot_perp_pair',
            execution='spot_perp_hedge',
            friction_profile='BingX_retail_maker',
            freq_tier='<1/day',
            timeframe_tier='daily',
            capital_tier='500-1500',
            directionality='pure_carry',
        ),
        # CFB 2025: basis arbitrage gated by sentiment/momentum delivers higher Sharpe
        # than naive basis. Only enter when momentum > X AND basis > threshold.
        expected_avg_gross_pct_per_trade=0.07,
        expected_trades_per_day=0.2,
        capital_usd=500,
        user_infrastructure=['BingX'],
        bt_live_mappable=True,
        expected_regime_dependence=0.3,
        description='R5 single-coin carry gated by 30d momentum filter. Skip carry during downtrends (typical funding compression / basis collapse periods).',
    ),

    # ─────────────────────────────────────────────────────────
    # 6. Multi-coin grid × ATR-volatility-conditional × cluster
    # ─────────────────────────────────────────────────────────
    StrategyProposal(
        name='C6_multi_grid_volatility_conditional',
        feature=StrategyFeature(
            mechanism='grid_oscillation',
            substrate='multi_coin_crypto_8',
            execution='maker_taker_mixed',
            friction_profile='BingX_retail_maker',
            freq_tier='1-10/day',
            timeframe_tier='1h',
            capital_tier='500-1500',
            directionality='market_neutral',
        ),
        # R26 lesson: grid in trending → catastrophic. ATR-based regime filter that ONLY
        # activates grid in low-vol ranging. Multi-coin diversifies regime correlation.
        expected_avg_gross_pct_per_trade=0.05,
        expected_trades_per_day=4.0,
        capital_usd=500,
        user_infrastructure=['BingX'],
        bt_live_mappable=False,  # multi-coin grid orchestration complex
        expected_regime_dependence=0.6,
        description='R26 variant w/ multi-coin (BTC/ETH/SOL) grid pool, regime filter requires ALL pairs in low-vol ranging. Re-arm AND halt properly modeled.',
    ),

    # ─────────────────────────────────────────────────────────
    # 7-10: Additional creative combinations (extending advisor list)
    # ─────────────────────────────────────────────────────────
    StrategyProposal(
        name='C7_funding_taker_skim',  # 시도 안 된 cell: funding × taker only × intraday
        feature=StrategyFeature(
            mechanism='funding_carry',
            substrate='BTC_funding_only',
            execution='spot_perp_hedge',
            friction_profile='BingX_retail_maker',
            freq_tier='<1/day',
            timeframe_tier='1h',
            capital_tier='500-1500',
            directionality='pure_carry',
        ),
        # Just-in-time funding capture: enter spot+perp 1h before funding tick, exit after.
        expected_avg_gross_pct_per_trade=0.012,  # 0.01%/8h × 0.85 capture rate
        expected_trades_per_day=3.0,
        capital_usd=500,
        user_infrastructure=['BingX'],
        bt_live_mappable=True,
        expected_regime_dependence=0.2,
        description='Just-in-time funding skim: open spot+perp 1h before funding payment, hold through tick, close. Eliminates 7h holding cost vs continuous carry.',
    ),

    StrategyProposal(
        name='C8_daily_xs_pure_carry_combo',
        feature=StrategyFeature(
            mechanism='cross_sectional',
            substrate='multi_coin_crypto_8',
            execution='spot_perp_hedge',
            friction_profile='BingX_retail_maker',
            freq_tier='<1/day',
            timeframe_tier='daily',
            capital_tier='500-1500',
            directionality='pure_carry',
        ),
        # Hybrid: cross-sectional + carry. Long the highest-funding coin, short lowest-funding.
        # Both legs contribute carry harvest, plus relative-funding-momentum drift.
        expected_avg_gross_pct_per_trade=0.06,
        expected_trades_per_day=0.15,
        capital_usd=500,
        user_infrastructure=['BingX'],
        bt_live_mappable=True,
        expected_regime_dependence=0.3,
        description='Long top-funding-rate coin (collect funding) + short bottom-funding coin (collect funding) hedged by spot. Two-sided carry.',
    ),

    StrategyProposal(
        name='C9_basis_widening_event',
        feature=StrategyFeature(
            mechanism='arbitrage',
            substrate='BTC_spot_perp_pair',
            execution='spot_perp_hedge',
            friction_profile='BingX_retail_maker',
            freq_tier='<1/day',
            timeframe_tier='1h',
            capital_tier='500-1500',
            directionality='market_neutral',
        ),
        # Rare event-driven: open carry only when basis widens > N std (e.g., volatility spikes,
        # halving, ETF news). High-quality entries, low frequency.
        expected_avg_gross_pct_per_trade=0.20,  # event-driven, high quality
        expected_trades_per_day=0.05,            # ~1.5/month
        capital_usd=500,
        user_infrastructure=['BingX'],
        bt_live_mappable=True,
        expected_regime_dependence=0.6,         # event-dependent
        description='Quiet most of the time; opens BTC spot-perp basis trade only when basis Z-score > 2.5 (rare event). Lower trade count, higher per-trade quality.',
    ),

    StrategyProposal(
        name='C10_xs_volatility_carry',
        feature=StrategyFeature(
            mechanism='cross_sectional',
            substrate='multi_coin_crypto_8',
            execution='maker_full_cycle',
            friction_profile='BingX_retail_maker',
            freq_tier='<1/day',
            timeframe_tier='daily',
            capital_tier='500-1500',
            directionality='market_neutral',
        ),
        # Short the highest-vol coin perp + long lowest-vol coin perp. Vol harvest like
        # equity low-vol anomaly. Independent of funding sign.
        expected_avg_gross_pct_per_trade=0.10,
        expected_trades_per_day=0.15,
        capital_usd=500,
        user_infrastructure=['BingX'],
        bt_live_mappable=True,
        expected_regime_dependence=0.4,
        description='Cross-sectional low-vol anomaly: short high-vol coin, long low-vol coin. Daily rebal. Tested in equities (Frazzini-Pedersen 2014) but not retail crypto.',
    ),
]


def main():
    print('=' * 100)
    print('PHASE 2 — Empty cell candidate evaluation')
    print('=' * 100)
    print(f'Candidates: {len(CANDIDATES)}')
    print()

    results = []
    for c in CANDIDATES:
        r = evaluate_proposal(c)
        results.append(r)

    # Sort by total reality score desc
    results.sort(key=lambda r: -r['reality']['total_score'])

    # Compact verdict report
    print(f'{"#":<3} {"name":<45} {"verdict":<8} {"score":<6} {"sim_to":<28} {"sim":<5}')
    print('-' * 100)
    for i, r in enumerate(results, 1):
        cc = r['cross_check']
        rl = r['reality']
        print(f'{i:<3} {r["proposal"]["name"]:<45} {rl["verdict"]:<8} {rl["total_score"]:.2f}/6 '
              f'{cc["most_similar_round"]:<28} {cc["max_similarity"]:.2f}')

    # Detailed for top 3
    print()
    print('=' * 100)
    print('TOP 3 detailed reasoning')
    print('=' * 100)
    for r in results[:3]:
        p = r['proposal']
        rl = r['reality']
        cc = r['cross_check']
        print()
        print(f'■ {p["name"]} — {rl["verdict"]} ({rl["total_score"]:.2f}/6, {rl["level_eligible"]})')
        print(f'  Description: {p.get("description","")[:120]}')
        print(f'  Most similar KB round: {cc["most_similar_round"]} ({cc["most_similar_verdict"]}, sim={cc["max_similarity"]:.3f})')
        print('  Dimensions:')
        for dim, dr in rl['dimensions'].items():
            print(f'    {dim:<26} {dr["score"]:.2f}  | {dr["detail"][:80]}')

    # Save full output
    out_path = Path(__file__).resolve().parent.parent.parent / 'results' / f'strategy_lab_phase2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'n_candidates': len(results),
        'rankings': [
            {
                'name': r['proposal']['name'],
                'verdict': r['reality']['verdict'],
                'total_score': r['reality']['total_score'],
                'level_eligible': r['reality']['level_eligible'],
                'description': r['proposal'].get('description', ''),
                'most_similar_kb': r['cross_check']['most_similar_round'],
                'similarity': r['cross_check']['max_similarity'],
                'dimensions': {dim: dr['score'] for dim, dr in r['reality']['dimensions'].items()},
            }
            for r in results
        ],
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print()
    print(f'Saved: {out_path}')

    # Save Plan template for top GO/CAUTION
    top = next((r for r in results if r['reality']['verdict'] in ('GO', 'CAUTION')), None)
    if top:
        plan_path = Path(__file__).resolve().parent.parent.parent / 'claudedocs' / f'pdca_plan_{top["proposal"]["name"]}.md'
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        with open(plan_path, 'w', encoding='utf-8') as f:
            f.write(top['plan_template_md'])
        print(f'Plan template for top candidate: {plan_path}')


if __name__ == '__main__':
    main()
