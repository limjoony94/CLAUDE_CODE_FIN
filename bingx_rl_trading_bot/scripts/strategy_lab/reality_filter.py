"""Reality Filter — 6-dimension feasibility score for new strategy proposals.

Each dimension scored 0~1. Aggregate verdict:
  - GO       (≥4.5/6): suitable for L1/L2 candidate development
  - CAUTION  (3.0~4.5): viable but needs careful validation
  - BLOCK    (<3.0): structural failure mode predicted

Dimensions:
  1. Friction arithmetic   — does avg_gross × freq beat friction × 2?
  2. Capital-scale binding — friction-as-fraction-of-position-size acceptable?
  3. Mechanism novelty     — distance from KB rounds (cosine similarity inverse)
  4. Substrate availability — user infrastructure matches?
  5. Path-dependent failure mode — regime/sample/cherry-pick risk
  6. 5-gate readiness      — can we audit BT-LIVE parity?
"""
from dataclasses import dataclass
from typing import Optional

from knowledge_base import FRICTION_PROFILES, empirical_friction_floor_pct
from cross_checker import StrategyFeature, cross_check


@dataclass
class StrategyProposal:
    """Full proposal for reality filter evaluation."""
    name: str                              # short identifier
    feature: StrategyFeature
    expected_avg_gross_pct_per_trade: float  # estimated, e.g., 0.10
    expected_trades_per_day: float           # estimated frequency
    capital_usd: float                       # capital scale
    user_infrastructure: list[str]           # ['BingX', 'Binance', 'Arbitrum_DeFi', 'TestNet']
    bt_live_mappable: Optional[bool] = None  # True if mechanism is simple enough for AST audit
    expected_regime_dependence: float = 0.5  # 0=invariant, 1=fully regime-dependent
    description: str = ''


# ============================================================================
# 6 Dimensions
# ============================================================================

def dim1_friction_arithmetic(p: StrategyProposal) -> dict:
    """avg_gross × freq vs friction × 2 (round-trip taker)."""
    fric = FRICTION_PROFILES.get(p.feature.friction_profile, {}).get('rt_pct', 0.10)
    rt_friction = fric  # already round-trip in our taxonomy

    daily_gross = p.expected_avg_gross_pct_per_trade * p.expected_trades_per_day
    daily_friction = rt_friction * p.expected_trades_per_day

    if daily_friction <= 0:
        ratio = float('inf')
    else:
        ratio = daily_gross / daily_friction

    if ratio >= 3.0:
        score = 1.0
    elif ratio >= 1.5:
        score = 0.7
    elif ratio >= 1.05:
        score = 0.4   # marginal — passes friction but tight
    elif ratio >= 0.5:
        score = 0.1   # close but below friction
    else:
        score = 0.0
    return {'score': score, 'detail': f'gross/friction ratio = {ratio:.2f} (gross {daily_gross:.4f}%/day, friction {daily_friction:.4f}%/day)'}


def dim2_capital_scale_binding(p: StrategyProposal) -> dict:
    """Friction-as-fraction-of-position-size penalty."""
    fric = FRICTION_PROFILES.get(p.feature.friction_profile, {}).get('rt_pct', 0.10)

    # On retail $500-1500, the absolute USD cost per trade is small but friction in % terms is high
    # (because position size is small relative to optimal scale).
    # KB evidence: $1500 capital → friction-as-fraction dominates regardless of mechanism.
    # We model: dependent on capital tier — higher capital relaxes binding.
    capital_score_map = {
        '<500':       0.2,
        '500-1500':   0.4,
        '1500-15k':   0.6,
        '15k-150k':   0.8,
        '>150k':      1.0,
    }
    score = capital_score_map.get(p.feature.capital_tier, 0.4)

    # For market-neutral / pure-carry, the binding is even tighter (smaller spreads)
    if p.feature.directionality in ('market_neutral', 'pure_carry'):
        score *= 0.85

    return {'score': score, 'detail': f'capital tier {p.feature.capital_tier} → score {score:.2f} ({"directional" if p.feature.directionality=="directional" else p.feature.directionality})'}


def dim3_mechanism_novelty(p: StrategyProposal) -> dict:
    """1 - max similarity to KB rounds (penalize variants of falsified)."""
    cmp = cross_check(p.feature, threshold=0.75)
    sim = cmp['max_similarity']
    novelty = 1.0 - sim
    # If most similar is FALSIFIED, even modest similarity hurts more
    if cmp['most_similar_verdict'] == 'FALSIFIED' and sim > 0.5:
        novelty *= 0.5
    return {
        'score': max(0.0, min(1.0, novelty)),
        'detail': f'most similar: {cmp["most_similar_round"]} (sim={sim:.3f}, {cmp["most_similar_verdict"]})',
    }


def dim4_substrate_availability(p: StrategyProposal) -> dict:
    """Does user infrastructure support this substrate?"""
    substrate_to_infra = {
        'BTC_perp_OHLCV_5m':       ['BingX', 'Binance'],
        'BTC_perp_OHLCV_15m':      ['BingX', 'Binance'],
        'BTC_perp_OHLCV_1h':       ['BingX', 'Binance'],
        'BTC_perp_OHLCV_daily':    ['BingX', 'Binance'],
        'BTC_perp_L2_orderbook':   ['BingX_websocket', 'Binance_L2'],
        'BTC_perp_trade_tape':     ['BingX_websocket'],
        'BTC_funding_only':        ['BingX'],
        'BTC_spot_perp_pair':      ['BingX'],
        'multi_coin_crypto_8':     ['BingX', 'Binance'],
        'multi_coin_crypto_30':    ['Binance'],
        'defi_L2_pools':           ['Arbitrum_DeFi', 'OP_DeFi', 'Base_DeFi'],
        'L2_microstructure_tick':  ['BingX_websocket', 'Binance_L2'],
    }
    required = substrate_to_infra.get(p.feature.substrate, [])
    if not required:
        return {'score': 0.5, 'detail': 'unknown substrate'}
    available = any(any(r in u for u in p.user_infrastructure) for r in required)
    score = 1.0 if available else 0.2
    # Cross-exchange execution requires multiple
    if p.feature.execution == 'cross_exchange':
        n_infra = len([u for u in p.user_infrastructure if u in ('BingX', 'Binance', 'OKX', 'Bybit')])
        if n_infra >= 2:
            score = 1.0
        else:
            score = 0.1
            return {'score': score, 'detail': 'cross-exchange requires ≥2 venues, user has ' + str(n_infra)}
    return {'score': score, 'detail': f'required {required}, user infra {p.user_infrastructure}'}


def dim5_path_dependent_failure(p: StrategyProposal) -> dict:
    """Predict regime dependence / sample variance / cherry-pick risk."""
    score = 1.0 - p.expected_regime_dependence
    # Penalty for very high freq (more bars, but more sensitive to regime shifts in BT)
    if p.feature.freq_tier == '>100/day':
        score *= 0.7
    if p.feature.freq_tier == '<1/day':
        # very low freq → small sample, high cherry-pick risk
        score *= 0.7
    # Bonus for market-neutral (less directional regime exposure)
    if p.feature.directionality == 'market_neutral':
        score = min(1.0, score * 1.1)
    if p.feature.directionality == 'pure_carry':
        score = min(1.0, score * 1.2)
    return {
        'score': max(0.0, min(1.0, score)),
        'detail': f'regime dependence {p.expected_regime_dependence:.2f}, freq {p.feature.freq_tier}, direction {p.feature.directionality}',
    }


def dim6_gate1_readiness(p: StrategyProposal) -> dict:
    """Can we audit BT-LIVE parity? Simpler mechanisms = easier audit."""
    # Mechanism complexity heuristic
    complexity_map = {
        'funding_carry':           0.9,  # simple: hold spot+perp, harvest funding
        'arbitrage':               0.9,
        'cross_sectional':         0.7,  # weekly rebal, moderate
        'defi_yield_rotation':     0.7,
        'session_time_filter':     0.8,
        'mean_reversion':          0.6,
        'pattern_based':           0.5,  # depends on pattern definition
        'trend_following':         0.4,  # multiple SL/TP/trail interactions (C1 lesson)
        'volatility_compression':  0.5,
        'grid_oscillation':        0.3,  # R26 lesson: re-arm cycle hard to model
        'microstructure':          0.4,  # L2 dynamics complex
    }
    score = complexity_map.get(p.feature.mechanism, 0.5)
    if p.bt_live_mappable is False:
        score *= 0.3
    elif p.bt_live_mappable is True:
        score = min(1.0, score * 1.2)
    return {
        'score': score,
        'detail': f'mechanism {p.feature.mechanism} complexity {complexity_map.get(p.feature.mechanism, 0.5):.1f}',
    }


# ============================================================================
# Aggregation
# ============================================================================

DIMENSIONS = [
    ('friction_arithmetic',     dim1_friction_arithmetic),
    ('capital_scale_binding',   dim2_capital_scale_binding),
    ('mechanism_novelty',       dim3_mechanism_novelty),
    ('substrate_availability',  dim4_substrate_availability),
    ('path_dependent_failure',  dim5_path_dependent_failure),
    ('gate1_readiness',         dim6_gate1_readiness),
]


def evaluate(p: StrategyProposal) -> dict:
    """Run 6-dim evaluation and aggregate."""
    results = {}
    total = 0.0
    for name, fn in DIMENSIONS:
        r = fn(p)
        results[name] = r
        total += r['score']

    if total >= 4.5:
        verdict = 'GO'
        level_eligible = 'L1_L2_potential'  # candidate for paper trade entry after Gate 1 audit
    elif total >= 3.0:
        verdict = 'CAUTION'
        level_eligible = 'L1_only_with_caveats'
    else:
        verdict = 'BLOCK'
        level_eligible = 'rejected'

    return {
        'name': p.name,
        'total_score': total,
        'max_score': 6.0,
        'verdict': verdict,
        'level_eligible': level_eligible,
        'dimensions': results,
    }


def report(eval_result: dict):
    print('=' * 80)
    print(f'Proposal: {eval_result["name"]}')
    print(f'Verdict:  {eval_result["verdict"]}  '
          f'({eval_result["total_score"]:.2f}/{eval_result["max_score"]})  '
          f'level: {eval_result["level_eligible"]}')
    print('=' * 80)
    for dim_name, r in eval_result['dimensions'].items():
        print(f'  {dim_name:<26} {r["score"]:.2f}  | {r["detail"]}')


if __name__ == '__main__':
    # Test: R26 grid variant (should BLOCK — high similarity to R26)
    p_r26_variant = StrategyProposal(
        name='R26_grid_variant_test',
        feature=StrategyFeature(
            mechanism='grid_oscillation',
            substrate='BTC_perp_OHLCV_1h',
            execution='maker_taker_mixed',
            friction_profile='BingX_retail_maker',
            freq_tier='1-10/day',
            timeframe_tier='1h',
            capital_tier='500-1500',
            directionality='directional',
        ),
        expected_avg_gross_pct_per_trade=0.10,
        expected_trades_per_day=3.0,
        capital_usd=500,
        user_infrastructure=['BingX'],
        bt_live_mappable=False,
        expected_regime_dependence=0.7,
    )
    report(evaluate(p_r26_variant))
    print()

    # Test: Funding-carry single-coin (should GO — similar to R5 deployable)
    p_carry = StrategyProposal(
        name='funding_carry_single_test',
        feature=StrategyFeature(
            mechanism='funding_carry',
            substrate='BTC_spot_perp_pair',
            execution='spot_perp_hedge',
            friction_profile='BingX_retail_maker',
            freq_tier='<1/day',
            timeframe_tier='1h',
            capital_tier='500-1500',
            directionality='pure_carry',
        ),
        expected_avg_gross_pct_per_trade=0.04,  # 0.04%/8h funding
        expected_trades_per_day=3.0,            # 3× 8h
        capital_usd=500,
        user_infrastructure=['BingX'],
        bt_live_mappable=True,
        expected_regime_dependence=0.3,
    )
    report(evaluate(p_carry))
