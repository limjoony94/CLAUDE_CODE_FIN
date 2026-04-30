"""Cross-checker — Cosine similarity vs 28 rounds prior.

Encode each round (and each new proposal) as a feature vector.
Compute cosine similarity. If similarity > 0.75 to any tested round → REJECT
(this proposal is a variant of a falsified mechanism).

Feature dimensions:
  - mechanism category (one-hot)
  - substrate category (one-hot)
  - execution category (one-hot)
  - friction profile (one-hot)
  - signal frequency tier (low/med/high)
  - timeframe tier (5m/15m/1h/daily)
  - capital scale tier ($500/$1500/$15000/$150k)
  - directional vs market-neutral (binary)
"""
import math
from dataclasses import dataclass
from typing import Optional

from knowledge_base import (
    KNOWLEDGE_BASE, MECHANISM_CATEGORIES, SUBSTRATE_CATEGORIES,
    EXECUTION_CATEGORIES, FRICTION_PROFILES,
)


# Categorical → vector
_FREQ_TIERS = ['<1/day', '1-10/day', '10-100/day', '>100/day']
_TIMEFRAME_TIERS = ['tick', '1m', '5m', '15m', '1h', '4h', 'daily', 'weekly', 'monthly']
_CAPITAL_TIERS = ['<500', '500-1500', '1500-15k', '15k-150k', '>150k']
_DIRECTIONALITY = ['directional', 'market_neutral', 'pure_carry']


@dataclass
class StrategyFeature:
    """Feature vector for cosine similarity comparison."""
    mechanism: str
    substrate: str
    execution: str
    friction_profile: str
    freq_tier: str = '1-10/day'
    timeframe_tier: str = '1h'
    capital_tier: str = '500-1500'
    directionality: str = 'directional'

    def to_vector(self) -> list[float]:
        v = []
        # mechanism one-hot (11 dims)
        v.extend([1.0 if m == self.mechanism else 0.0 for m in MECHANISM_CATEGORIES])
        # substrate one-hot (12 dims)
        v.extend([1.0 if s == self.substrate else 0.0 for s in SUBSTRATE_CATEGORIES])
        # execution one-hot (7 dims)
        v.extend([1.0 if e == self.execution else 0.0 for e in EXECUTION_CATEGORIES])
        # friction profile one-hot (4 dims)
        v.extend([1.0 if f == self.friction_profile else 0.0 for f in FRICTION_PROFILES.keys()])
        # freq tier one-hot (4 dims)
        v.extend([1.0 if t == self.freq_tier else 0.0 for t in _FREQ_TIERS])
        # timeframe tier one-hot (9 dims)
        v.extend([1.0 if t == self.timeframe_tier else 0.0 for t in _TIMEFRAME_TIERS])
        # capital tier one-hot (5 dims)
        v.extend([1.0 if c == self.capital_tier else 0.0 for c in _CAPITAL_TIERS])
        # directionality one-hot (3 dims)
        v.extend([1.0 if d == self.directionality else 0.0 for d in _DIRECTIONALITY])
        return v


def _round_to_feature(r) -> StrategyFeature:
    """Map a RoundEvidence to its StrategyFeature."""
    # Infer freq_tier from substrate
    freq_map = {
        'BTC_perp_OHLCV_5m':       '10-100/day',
        'BTC_perp_OHLCV_15m':      '1-10/day',
        'BTC_perp_OHLCV_1h':       '1-10/day',
        'BTC_perp_OHLCV_daily':    '<1/day',
        'BTC_perp_L2_orderbook':   '>100/day',
        'BTC_perp_trade_tape':     '>100/day',
        'BTC_funding_only':        '<1/day',
        'BTC_spot_perp_pair':      '<1/day',
        'multi_coin_crypto_8':     '1-10/day',
        'multi_coin_crypto_30':    '1-10/day',
        'defi_L2_pools':           '<1/day',
        'L2_microstructure_tick':  '>100/day',
    }
    tf_map = {
        'BTC_perp_OHLCV_5m':       '5m',
        'BTC_perp_OHLCV_15m':      '15m',
        'BTC_perp_OHLCV_1h':       '1h',
        'BTC_perp_OHLCV_daily':    'daily',
        'BTC_perp_L2_orderbook':   'tick',
        'BTC_perp_trade_tape':     'tick',
        'BTC_funding_only':        '1h',  # funding is 8h but treated as 1h tier
        'BTC_spot_perp_pair':      '1h',
        'multi_coin_crypto_8':     '1h',
        'multi_coin_crypto_30':    '1h',
        'defi_L2_pools':           'monthly',
        'L2_microstructure_tick':  'tick',
    }
    cap_map = {500: '500-1500', 1500: '500-1500'}
    dir_map = {
        'funding_carry':           'pure_carry',
        'arbitrage':               'market_neutral',
        'cross_sectional':         'market_neutral',
        'defi_yield_rotation':     'pure_carry',
    }
    return StrategyFeature(
        mechanism=r.mechanism,
        substrate=r.substrate,
        execution=r.execution,
        friction_profile=r.friction_profile,
        freq_tier=freq_map.get(r.substrate, '1-10/day'),
        timeframe_tier=tf_map.get(r.substrate, '1h'),
        capital_tier=cap_map.get(r.capital_assumption_usd, '500-1500'),
        directionality=dir_map.get(r.mechanism, 'directional'),
    )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def compare_to_kb(proposal: StrategyFeature) -> list[dict]:
    """Return list of (round_id, similarity, verdict) sorted by similarity desc."""
    pv = proposal.to_vector()
    out = []
    for r in KNOWLEDGE_BASE:
        rv = _round_to_feature(r).to_vector()
        sim = cosine_similarity(pv, rv)
        out.append({
            'round_id': r.round_id,
            'similarity': sim,
            'verdict': r.verdict,
            'falsification_reason': r.falsification_reason,
        })
    out.sort(key=lambda x: -x['similarity'])
    return out


def cross_check(proposal: StrategyFeature, threshold: float = 0.75) -> dict:
    """Return cross-check verdict: REJECT if max similarity > threshold to a FALSIFIED round."""
    cmp = compare_to_kb(proposal)
    top = cmp[0]
    # Reject if highest similarity is to a FALSIFIED or VACUOUS round above threshold
    rejected = False
    if top['similarity'] >= threshold and top['verdict'] in ('FALSIFIED', 'VACUOUS'):
        rejected = True
    return {
        'rejected': rejected,
        'max_similarity': top['similarity'],
        'most_similar_round': top['round_id'],
        'most_similar_verdict': top['verdict'],
        'reason': top.get('falsification_reason'),
        'top_5': cmp[:5],
    }


# ============================================================================
# Tests
# ============================================================================
if __name__ == '__main__':
    # Test 1: A proposal that's clearly R26 grid variant should be REJECTED
    proposal_grid_variant = StrategyFeature(
        mechanism='grid_oscillation',
        substrate='BTC_perp_OHLCV_1h',
        execution='maker_taker_mixed',
        friction_profile='BingX_retail_maker',
        freq_tier='1-10/day',
        timeframe_tier='1h',
        capital_tier='500-1500',
        directionality='directional',
    )
    print('=' * 80)
    print('Test 1: Grid variant (should match R26)')
    print('=' * 80)
    res = cross_check(proposal_grid_variant)
    print(f'Rejected: {res["rejected"]}')
    print(f'Max similarity: {res["max_similarity"]:.3f} to {res["most_similar_round"]} ({res["most_similar_verdict"]})')
    print()

    # Test 2: A novel proposal (funding × multi-coin × cross-exchange) should NOT be rejected
    proposal_novel = StrategyFeature(
        mechanism='arbitrage',
        substrate='BTC_spot_perp_pair',
        execution='cross_exchange',
        friction_profile='BingX_retail_taker',
        freq_tier='<1/day',
        timeframe_tier='1h',
        capital_tier='500-1500',
        directionality='market_neutral',
    )
    print('=' * 80)
    print('Test 2: Novel (cross-exchange arbitrage, untested)')
    print('=' * 80)
    res = cross_check(proposal_novel)
    print(f'Rejected: {res["rejected"]}')
    print(f'Max similarity: {res["max_similarity"]:.3f} to {res["most_similar_round"]} ({res["most_similar_verdict"]})')
    print('Top 5 nearest:')
    for x in res['top_5']:
        print(f'  {x["similarity"]:.3f}  {x["round_id"]:<40}  {x["verdict"]}')
