"""Frontier Mapper — M × S × E cube + empty cells with priority ranking.

For each (mechanism, substrate, execution) combination:
  - Tested? (in KNOWLEDGE_BASE)
  - If not, project friction floor pass probability
  - Priority = novelty × feasibility × distance from binding constraint

Output: ranked list of unexplored cells worth proposing for L1-L2 strategy ideation.
"""
from itertools import product
from typing import Optional

from knowledge_base import (
    MECHANISM_CATEGORIES, SUBSTRATE_CATEGORIES, EXECUTION_CATEGORIES,
    FRICTION_PROFILES, KNOWLEDGE_BASE,
    get_tested_combinations, empirical_friction_floor_pct,
)


# ============================================================================
# Substrate-execution feasibility matrix (NOT all combinations are valid)
# ============================================================================

# (substrate, execution) → feasible at retail BingX setup
FEASIBLE_SUBSTRATE_EXECUTION = {
    ('BTC_perp_OHLCV_5m',     'taker_only'): True,
    ('BTC_perp_OHLCV_5m',     'maker_entry_taker_exit'): True,
    ('BTC_perp_OHLCV_5m',     'maker_full_cycle'): True,
    ('BTC_perp_OHLCV_5m',     'maker_taker_mixed'): True,
    ('BTC_perp_OHLCV_15m',    'taker_only'): True,
    ('BTC_perp_OHLCV_15m',    'maker_entry_taker_exit'): True,
    ('BTC_perp_OHLCV_15m',    'maker_full_cycle'): True,
    ('BTC_perp_OHLCV_15m',    'maker_taker_mixed'): True,
    ('BTC_perp_OHLCV_1h',     'taker_only'): True,
    ('BTC_perp_OHLCV_1h',     'maker_entry_taker_exit'): True,
    ('BTC_perp_OHLCV_1h',     'maker_full_cycle'): True,
    ('BTC_perp_OHLCV_1h',     'maker_taker_mixed'): True,
    ('BTC_perp_OHLCV_daily',  'taker_only'): True,
    ('BTC_perp_OHLCV_daily',  'maker_full_cycle'): True,
    ('BTC_funding_only',      'spot_perp_hedge'): True,
    ('BTC_spot_perp_pair',    'spot_perp_hedge'): True,
    ('multi_coin_crypto_8',   'spot_perp_hedge'): True,
    ('multi_coin_crypto_8',   'maker_full_cycle'): True,
    ('multi_coin_crypto_8',   'taker_only'): True,
    ('multi_coin_crypto_30',  'maker_full_cycle'): True,
    ('multi_coin_crypto_30',  'taker_only'): True,
    ('BTC_perp_L2_orderbook', 'taker_only'): True,
    ('BTC_perp_trade_tape',   'taker_only'): True,
    ('defi_L2_pools',         'pool_rotation'): True,
    ('L2_microstructure_tick', 'taker_only'): True,
    # Cross-exchange: only specific substrates
    ('BTC_spot_perp_pair',    'cross_exchange'): True,
    ('multi_coin_crypto_8',   'cross_exchange'): True,
}

# Mechanism-substrate plausibility (some combinations don't make sense)
PLAUSIBLE_MECH_SUBSTRATE = {
    'trend_following':         {'BTC_perp_OHLCV_5m', 'BTC_perp_OHLCV_15m', 'BTC_perp_OHLCV_1h', 'BTC_perp_OHLCV_daily'},
    'mean_reversion':          {'BTC_perp_OHLCV_5m', 'BTC_perp_OHLCV_15m', 'BTC_perp_OHLCV_1h', 'BTC_perp_OHLCV_daily'},
    'pattern_based':           {'BTC_perp_OHLCV_5m', 'BTC_perp_OHLCV_15m', 'BTC_perp_OHLCV_1h'},
    'grid_oscillation':        {'BTC_perp_OHLCV_5m', 'BTC_perp_OHLCV_15m', 'BTC_perp_OHLCV_1h', 'multi_coin_crypto_8'},
    'funding_carry':           {'BTC_funding_only', 'BTC_spot_perp_pair', 'multi_coin_crypto_8', 'multi_coin_crypto_30'},
    'microstructure':          {'BTC_perp_L2_orderbook', 'BTC_perp_trade_tape', 'L2_microstructure_tick'},
    'cross_sectional':         {'multi_coin_crypto_8', 'multi_coin_crypto_30'},
    'defi_yield_rotation':     {'defi_L2_pools'},
    'volatility_compression':  {'BTC_perp_OHLCV_5m', 'BTC_perp_OHLCV_15m', 'BTC_perp_OHLCV_1h'},
    'session_time_filter':     {'BTC_perp_OHLCV_5m', 'BTC_perp_OHLCV_15m', 'BTC_perp_OHLCV_1h'},
    'arbitrage':               {'BTC_spot_perp_pair', 'multi_coin_crypto_8', 'BTC_funding_only'},
}


def all_plausible_combinations():
    """Generate all (m, s, e) combinations that are plausible AND feasible."""
    out = []
    for m in MECHANISM_CATEGORIES:
        for s in PLAUSIBLE_MECH_SUBSTRATE.get(m, set()):
            for e in EXECUTION_CATEGORIES:
                if FEASIBLE_SUBSTRATE_EXECUTION.get((s, e), False):
                    out.append((m, s, e))
    return out


def empty_cells():
    """Cells not yet tested."""
    tested = get_tested_combinations()
    plausible = set(all_plausible_combinations())
    return sorted(plausible - tested)


# ============================================================================
# Friction floor projection — does the cell have a chance vs friction?
# ============================================================================

def project_friction_floor_pass(mechanism: str, substrate: str, execution: str) -> dict:
    """Heuristic projection — does this cell have a chance vs 0.07% friction?

    Based on 28 rounds learning:
      - Pure directional 5m on retail OHLCV = NO (R41 +0.034% < 0.07%)
      - Funding carry single-coin = YES (R5 deployable)
      - Funding carry multi-coin equal-weight = MARGINAL (R13 diluted)
      - Cross-sectional momentum = MARGINAL (PathB-R1 borderline)
      - Microstructure substrate = NO (avg_gross 30-90× below friction)
      - DeFi yield = MARGINAL (DeFi-R1 net 1.77%)
      - Maker-entry directional = NO (adverse selection)
      - Daily/weekly substrates = HIGHER chance (lower frequency = friction less binding)
    """
    score = 0.0
    reasons = []

    # Mechanism prior
    mechanism_priors = {
        'funding_carry':           0.7,
        'arbitrage':               0.6,
        'cross_sectional':         0.5,
        'defi_yield_rotation':     0.5,
        'session_time_filter':     0.3,
        'volatility_compression':  0.2,
        'mean_reversion':          0.2,
        'pattern_based':           0.2,
        'grid_oscillation':        0.3,
        'trend_following':         0.2,
        'microstructure':          0.1,
    }
    score += mechanism_priors.get(mechanism, 0.1)
    reasons.append(f'mechanism prior {mechanism}: +{mechanism_priors.get(mechanism, 0.1):.2f}')

    # Substrate frequency penalty (high-frequency = friction-bound)
    substrate_freq_penalty = {
        'BTC_perp_OHLCV_5m':      -0.3,
        'BTC_perp_OHLCV_15m':     -0.2,
        'BTC_perp_OHLCV_1h':      -0.1,
        'BTC_perp_OHLCV_daily':    0.1,
        'BTC_perp_L2_orderbook':  -0.4,
        'BTC_perp_trade_tape':    -0.3,
        'BTC_funding_only':        0.1,
        'BTC_spot_perp_pair':      0.1,
        'multi_coin_crypto_8':     0.0,
        'multi_coin_crypto_30':    0.0,
        'defi_L2_pools':          -0.1,
        'L2_microstructure_tick': -0.4,
    }
    score += substrate_freq_penalty.get(substrate, 0.0)
    reasons.append(f'substrate freq {substrate}: {substrate_freq_penalty.get(substrate, 0.0):+.2f}')

    # Execution penalty
    execution_penalty = {
        'taker_only':                0.0,
        'maker_entry_taker_exit':   -0.2,  # adverse selection if directional
        'maker_full_cycle':          0.1,  # if mean-reverting context
        'maker_taker_mixed':        -0.1,  # complex
        'spot_perp_hedge':           0.2,  # carry friendly
        'cross_exchange':            0.1,
        'pool_rotation':            -0.1,
    }
    # Special case: maker-entry on directional mechanism = adverse selection
    if execution == 'maker_entry_taker_exit' and mechanism in ('trend_following', 'pattern_based'):
        score -= 0.2
        reasons.append(f'maker_entry on directional: -0.20 (adverse selection)')
    else:
        score += execution_penalty.get(execution, 0.0)
        reasons.append(f'execution {execution}: {execution_penalty.get(execution, 0.0):+.2f}')

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))

    return {'score': score, 'reasons': reasons}


# ============================================================================
# Empty cell ranking
# ============================================================================

def rank_empty_cells():
    """Return empty cells ranked by projected friction-pass score (descending)."""
    cells = empty_cells()
    ranked = []
    for m, s, e in cells:
        proj = project_friction_floor_pass(m, s, e)
        ranked.append({
            'mechanism': m, 'substrate': s, 'execution': e,
            'friction_pass_score': proj['score'],
            'reasons': proj['reasons'],
        })
    ranked.sort(key=lambda x: -x['friction_pass_score'])
    return ranked


def report():
    """Print frontier summary."""
    tested = get_tested_combinations()
    all_plausible = set(all_plausible_combinations())
    empty = sorted(all_plausible - tested)

    print('=' * 100)
    print(f'Frontier Mapper — M × S × E cube')
    print('=' * 100)
    print(f'Plausible combinations:  {len(all_plausible)}')
    print(f'Tested:                  {len(tested)}')
    print(f'Empty (untested):        {len(empty)}')
    print(f'Coverage:                {len(tested)/len(all_plausible)*100:.1f}%')
    print()

    print('Top 15 empty cells by friction-pass projection:')
    print(f'{"score":>5}  {"mechanism":<22} {"substrate":<26} {"execution":<24}')
    print('-' * 90)
    for r in rank_empty_cells()[:15]:
        print(f'{r["friction_pass_score"]:>5.2f}  '
              f'{r["mechanism"]:<22} {r["substrate"]:<26} {r["execution"]:<24}')
    print()

    print('Bottom 5 (likely doomed):')
    print(f'{"score":>5}  {"mechanism":<22} {"substrate":<26} {"execution":<24}')
    print('-' * 90)
    for r in rank_empty_cells()[-5:]:
        print(f'{r["friction_pass_score"]:>5.2f}  '
              f'{r["mechanism"]:<22} {r["substrate"]:<26} {r["execution"]:<24}')


if __name__ == '__main__':
    report()
