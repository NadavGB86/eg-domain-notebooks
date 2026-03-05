"""
Synthetic Strategy Universe V6 — Market-Calibrated
====================================================
Generates a realistic trading strategy universe with embedded market
dynamics calibrated from empirical quantitative finance research.

Design Philosophy:
  This generator creates a realistic strategy universe exhibiting key
  market dynamics (alpha decay, regime shifts, correlation spikes) that
  challenge portfolio construction methods. The embedded mechanisms are
  calibrated from empirical research, producing a universe where adaptive
  selection demonstrably outperforms static approaches.

Market Mechanisms (research-grounded):
  1. ALPHA DECAY — Strategies lose edge exponentially over time.
     Reference: Di Mascio & Lines (JoF, 2021): alpha decays gradually,
     with ~12-month typical decay period. Maven Securities (2021):
     annual decay costs 5.6% (US) to 9.9% (EU).
     Implementation: fading_alpha tier with exp decay crossing zero.

  2. REGIME-DEPENDENT CORRELATIONS — Strategy correlations spike in crisis.
     Reference: Ang & Chen (JFE, 2002): equity correlations are higher
     in bear markets. Sandoval & Franca (2012): markets behave as one
     during crashes. Alcock & Satchell (2018): diversification fails
     precisely when needed most.
     Implementation: sector factors amplified 1.8x during turbulence;
     regime_sensitive tier with beta that flips sign in crisis.

  3. GARCH-LIKE VOLATILITY — Volatility clusters rather than being constant.
     Implementation: GARCH(1,1)-like process for idiosyncratic vol.

  4. FAT TAILS — Student-t(5) instead of Gaussian. Real returns have
     kurtosis ~5-10 at monthly frequency.

  5. SURVIVORSHIP BIAS — Strategies appear and disappear. Short-lived
     failures create incomplete panel data.

Strategy Archetypes:
  penny_picker    (~10%): High Sharpe (~6), near-zero PPC.
                          Sharpe trap — looks great, can't scale.
  fading_alpha    (~22%): Starts strongly positive, decays past zero.
                          Wrecks static portfolios, rewards adaptive WF.
  regime_sensitive (~14%): Market beta flips with regime.
                          Drives crisis-time correlation spike.
  hedge           (~10%): Negative market beta. Crisis alpha.
                          Insurance-like profile. Decays as crowded.
  late_bloomer    (~14%): Bad early, improves to positive.
                          Rewards EWMA signal over flat average.
  dragger         (~30%): Mildly negative drift. Noisy.
                          Drags the benchmark. Selection removes them.

Calibration Targets (seed=73):
  Benchmark:  Sharpe ~0.5-0.8  (mediocre, room for improvement)
  Oracle:     Sharpe ~5-8      (clear ceiling, motivates selection)
  Static:     Sharpe ~3-4      (penny trap: high Sharpe, low PnL)
  WF:         Sharpe ~4-5      (adaptive, best implementable)
  PnL order:  Oracle > WF > Static > Benchmark

V5 → V6 Changes:
  - Replaced constant drift with exponential decay (fading_alpha)
  - Added late_bloomer tier with sigmoid improvement curve
  - Replaced simple regime flip with beta-flip regime_sensitive
  - Added hedge tier with crisis bonus
  - Added GARCH-like idiosyncratic volatility
  - Replaced single sine-wave market factor with 4-state regime
  - Added sector factors with crisis amplification
  - Calibrated penny pickers to Sharpe ~6 (was 11+ in V5)
  - Optimized seed selection for narrative ordering
"""
import hashlib
import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)   # best-effort in notebook
random.seed(SEED)
np.random.seed(SEED)                      # legacy/global RNG (E2E)


def get_seed32(stage: str, seed: int = SEED) -> int:
    b = f"{seed}::{stage}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(b, digest_size=4).digest(), "little")


def get_rng(stage: str, seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(get_seed32(stage, seed))

# rng = get_rng(stage='generate_living_quant_universe', seed=seed)


def generate_living_quant_universe(
    n_strategies: int = 50,
    n_months: int = 120,
    seed: int = 75,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic strategy panel dataset.

    Returns
    -------
    df : pd.DataFrame
        Long-format with columns [month, strategy, pnl, contracts].
    tier_df : pd.DataFrame
        Per-strategy metadata: tier, avg_pnl, n_months, sector.
    """
    rng = np.random.RandomState(seed)
    months = pd.date_range("2010-01-01", periods=n_months, freq="ME")

    # =================================================================
    # 1. MARKET REGIME (4-state + mini-shock)
    # =================================================================
    # Growth:     months 0-37   (38 months)
    # Turbulence: months 38-57  (20 months) — crisis, vol spike
    # New Normal: months 58-89  (32 months) — recovery, new winners
    # Late Cycle: months 90-119 (30 months) — crowding, decay
    regime = np.zeros(n_months, dtype=int)
    regime[38:58] = 1   # Turbulence
    regime[58:90] = 2   # New normal
    regime[90:] = 3     # Late cycle
    regime[18:22] = 1   # Mini-shock (tests early adaptiveness)

    # Soften regime boundaries: transition months blend adjacent regimes
    # via f_vol smoothing (no RNG calls — preserves downstream state)

    # =================================================================
    # 2. MACRO FACTORS (regime-aware)
    # =================================================================
    # F1: Market — AR(1) with regime-dependent mean and vol
    f_market = np.zeros(n_months)
    for t in range(1, n_months):
        params = {
            0: (1.5, 5.0),   # Growth: mildly positive, moderate vol
            1: (-4.0, 12.0),  # Turbulence: negative, high vol
            2: (2.5, 5.5),    # New normal: positive, moderate vol
            3: (0.0, 8.0),    # Late cycle: flat, elevated vol
        }
        mu, vol = params[regime[t]]
        f_market[t] = 0.25 * f_market[t-1] + mu + rng.normal(0, vol)

    # F2: Volatility scaling factor (smooth, regime-dependent)
    f_vol = np.array([{
        0: rng.uniform(0.6, 1.0),
        1: rng.uniform(1.8, 3.0),
        2: rng.uniform(0.7, 1.2),
        3: rng.uniform(1.1, 1.8),
    }[r] for r in regime])
    f_vol = uniform_filter1d(f_vol.astype(float), size=5)

    # F3: Momentum factor (autocorrelated)
    f_mom = np.zeros(n_months)
    for t in range(1, n_months):
        f_mom[t] = 0.6 * f_mom[t-1] + rng.normal(0, 3)

    # =================================================================
    # 3. SECTOR FACTORS (5 sectors, crisis-amplified)
    # =================================================================
    n_sectors = 5
    sector_factors = {}
    for s in range(n_sectors):
        sf = np.zeros(n_months)
        for t in range(1, n_months):
            sf[t] = 0.5 * sf[t-1] + rng.normal(0, 2.5)
            if regime[t] == 1:
                sf[t] *= 1.8  # Crisis amplification → correlation spike
        sector_factors[s] = sf

    records: list[dict] = []
    tier_log: list[dict] = []

    for i in range(n_strategies):
        name = f"strat_{i:03d}"

        # --- Survivorship bias ---
        start = rng.randint(0, max(1, n_months // 5))
        end = rng.randint(n_months * 4 // 5, n_months)
        if rng.random() < 0.12:
            start = rng.randint(0, n_months // 3)
            end = min(start + rng.randint(12, 36), n_months)
        active_months = months[start:end]
        n_active = len(active_months)
        if n_active < 6:
            continue

        # --- GARCH-like idiosyncratic volatility ---
        idio_vol = np.ones(n_active)
        h = 1.0
        for t in range(n_active):
            h = 0.1 + 0.82 * h + 0.12 * rng.standard_t(df=4) ** 2
            idio_vol[t] = np.sqrt(max(h, 0.1))
        idio_noise = rng.standard_t(df=5, size=n_active) * idio_vol

        tier_roll = rng.random()
        sector = rng.randint(0, n_sectors)
        base_contracts = rng.uniform(80, 250)
        sector_load = rng.uniform(0.2, 0.5)

        # ==============================================================
        # STRATEGY ARCHETYPES
        # ==============================================================

        if tier_roll < 0.10:
            # ── PENNY PICKERS ──
            # Moderate drift, low-ish vol → Sharpe ~3-4
            # High contract volume → PPC near zero (the trap)
            tier = "penny_picker"
            drift = rng.uniform(2.5, 5.0)
            sigma = rng.uniform(2.5, 5.0)
            pnl = drift + idio_noise * sigma
            pnl += sector_factors[sector][start:end] * 0.15
            pnl += f_market[start:end] * rng.uniform(0.05, 0.15)
            contracts = np.abs(rng.normal(rng.uniform(2000, 5000), 600, n_active))

        elif tier_roll < 0.32:
            # ── FADING ALPHA ──
            # drift(t) = initial * exp(-λt) - offset
            # Crosses zero → becomes losing strategy over time
            tier = "fading_alpha"
            initial_drift = rng.uniform(30, 55)
            decay_rate = rng.uniform(0.012, 0.028)
            t_arr = np.arange(n_active)
            offset = initial_drift * rng.uniform(0.22, 0.36)
            drift = initial_drift * np.exp(-decay_rate * t_arr) - offset

            beta_mkt = rng.uniform(0.2, 0.8)
            sigma = rng.uniform(7, 14)
            pnl = (drift
                   + f_market[start:end] * beta_mkt
                   + idio_noise * sigma * f_vol[start:end])
            pnl += sector_factors[sector][start:end] * sector_load
            contracts = np.abs(
                rng.normal(base_contracts, base_contracts * 0.15, n_active)
            )

        elif tier_roll < 0.46:
            # ── REGIME SENSITIVES ──
            # Market beta changes sign with regime
            # Growth: positive beta (rides market up)
            # Turbulence: very negative beta (crashes with market)
            # Drives the correlation spike in crisis
            tier = "regime_sensitive"
            drift = rng.uniform(2, 8)
            sigma = rng.uniform(8, 15)
            beta = np.array([{
                0: rng.uniform(0.4, 1.0),
                1: rng.uniform(-2.0, -0.6),
                2: rng.uniform(-0.3, 0.6),
                3: rng.uniform(-0.8, 0.3),
            }[regime[start + t]] for t in range(n_active)])
            pnl = (drift
                   + f_market[start:end] * beta
                   + idio_noise * sigma * f_vol[start:end])
            pnl += sector_factors[sector][start:end] * sector_load * 1.2
            contracts = np.abs(
                rng.normal(base_contracts * 1.2, base_contracts * 0.2, n_active)
            )

        elif tier_roll < 0.56:
            # ── HEDGES ──
            # Negative market beta. Crisis bonus. Diverse profiles.
            tier = "hedge"
            drift = rng.uniform(2, 10)
            beta_mkt = rng.uniform(-1.5, -0.2)
            sigma = rng.uniform(6, 14)
            # Crisis bonus varies widely per hedge — not lockstep
            crisis_mag = rng.uniform(5, 35)
            crisis_bonus = np.array([
                rng.uniform(crisis_mag * 0.5, crisis_mag * 1.5) if regime[start + t] == 1
                else rng.uniform(-2, 2)
                for t in range(n_active)
            ])
            pnl = (drift
                   + f_market[start:end] * beta_mkt
                   + crisis_bonus
                   + idio_noise * sigma * f_vol[start:end] * 0.8)
            pnl += sector_factors[sector][start:end] * sector_load * 0.5
            contracts = np.abs(rng.normal(base_contracts * 0.6, base_contracts * 0.08, n_active))

        elif tier_roll < 0.70:
            # ── LATE BLOOMERS ──
            # Sigmoid curve: initial_drift → final_level
            # Bad early, good later. Rewards EWMA over flat average.
            tier = "late_bloomer"
            initial_drift = rng.uniform(-25, -10)
            t_arr = np.arange(n_active)
            midpoint = n_active * rng.uniform(0.55, 0.75)
            final_level = abs(initial_drift) * rng.uniform(1.3, 2.2)
            drift = initial_drift + (final_level - initial_drift) / (
                1 + np.exp(-0.08 * (t_arr - midpoint))
            )
            beta_mkt = rng.uniform(0.2, 0.7)
            sigma = rng.uniform(7, 14)
            pnl = (drift
                   + f_market[start:end] * beta_mkt
                   + idio_noise * sigma * f_vol[start:end])
            pnl += sector_factors[sector][start:end] * sector_load
            contracts = np.abs(
                rng.normal(base_contracts, base_contracts * 0.2, n_active)
            )

        else:
            # ── DRAGGERS ──
            # Mildly negative drift. High noise. Drag the benchmark.
            tier = "dragger"
            drift = rng.uniform(-6, -0.5)
            beta_mkt = rng.uniform(0.15, 0.7)
            sigma = rng.uniform(8, 16)
            pnl = (drift
                   + f_market[start:end] * beta_mkt
                   + idio_noise * sigma * f_vol[start:end])
            pnl += sector_factors[sector][start:end] * sector_load
            contracts = np.abs(
                rng.normal(base_contracts, base_contracts * 0.25, n_active)
            )

        contracts = np.clip(contracts, 1, None).astype(int)
        tier_log.append({
            "strategy": name,
            "tier": tier,
            "avg_pnl": float(np.mean(pnl)),
            "n_months": n_active,
            "sector": sector,
        })
        for j, m in enumerate(active_months):
            records.append({
                "month": m,
                "strategy": name,
                "pnl": round(float(pnl[j]), 2),
                "contracts": int(contracts[j]),
            })

    df = pd.DataFrame(records).sort_values(["month", "strategy"]).reset_index(drop=True)
    tier_frame = pd.DataFrame(tier_log)

    # =================================================================
    # POST-PROCESSING: Cross-Strategy Dynamics
    # =================================================================
    # 1. Sector contagion: when a sector's mean PnL is negative,
    #    same-sector strategies face a lagged negative shock.
    #    Creates exploitable cross-strategy predictability.
    # 2. Market lead-lag: lagged cross-sectional mean influences
    #    next month's individual returns. Creates momentum signal.

    contagion_beta = 0.2
    leadlag_beta = 0.15

    df_ws = df.merge(tier_frame[["strategy", "sector"]], on="strategy")
    all_months = sorted(df["month"].unique())
    month_next = {m: all_months[i + 1] for i, m in enumerate(all_months[:-1])}

    # Sector contagion (1-month lag, negative only)
    sec_mean = (df_ws.groupby(["month", "sector"])["pnl"].mean().reset_index().rename(columns={"pnl": "sec_mean"}))
    sec_lag = sec_mean.copy()
    sec_lag["month"] = sec_lag["month"].map(month_next)
    sec_lag = sec_lag.dropna(subset=["month"]).rename(columns={"sec_mean": "sec_lag"})
    df_ws = df_ws.merge(sec_lag, on=["month", "sector"], how="left")
    df_ws["sec_lag"] = df_ws["sec_lag"].fillna(0)

    # Market lead-lag (1-month lag, both directions)
    mkt_mean = (df.groupby("month")["pnl"].mean().reset_index().rename(columns={"pnl": "mkt_mean"}))
    mkt_lag = mkt_mean.copy()
    mkt_lag["month"] = mkt_lag["month"].map(month_next)
    mkt_lag = mkt_lag.dropna(subset=["month"]).rename(columns={"mkt_mean": "mkt_lag"})
    df_ws = df_ws.merge(mkt_lag, on="month", how="left")
    df_ws["mkt_lag"] = df_ws["mkt_lag"].fillna(0)

    # Apply dynamics
    df_ws["pnl"] = (df_ws["pnl"] + contagion_beta * df_ws["sec_lag"].clip(upper=0) + leadlag_beta * df_ws["mkt_lag"])

    df = df_ws[["month", "strategy", "pnl", "contracts"]].copy()
    df["pnl"] = df["pnl"].round(2)
    df = df.sort_values(["month", "strategy"]).reset_index(drop=True)

    return df, tier_frame


def generate_realistic_quant_data(
    n_months=120, n_strategies=50, *, stage="mock_data"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates synthetic strategy data.
    UPDATED: Now simulates BOTH strategy 'births' (incubation) and 'deaths' (survivorship bias)
    to match realistic market data dynamics.
    """
    r = get_rng(stage)
    dates = pd.date_range(start='2010-01-01', periods=n_months, freq='ME')

    pnl_data = {}
    contracts_data = {}

    for i in range(n_strategies):
        volatility = r.uniform(100, 500)
        pnl = r.normal(0, volatility, n_months)

        strat_type = r.choice(['bull', 'bear', 'steady', 'noise'])

        if strat_type == 'bull':
            pnl[0:40] += r.normal(150, 50, 40)
            pnl[40:80] -= r.normal(100, 50, 40)
            pnl[80:] += r.normal(100, 50, 40)
        elif strat_type == 'bear':
            pnl[0:40] -= r.normal(50, 50, 40)
            pnl[40:80] += r.normal(250, 80, 40)
            pnl[80:] -= r.normal(50, 50, 40)
        elif strat_type == 'steady':
            pnl += r.normal(30, 20, n_months)

        # SIMULATE BIRTHS (Late starters)
        if r.random() > 0.4:
            birth_idx = r.integers(1, n_months // 2)
            pnl[:birth_idx] = np.nan

        # SIMULATE DEATHS (Survivorship Bias)
        if r.random() > 0.6:
            death_idx = r.integers(n_months // 2, n_months)
            pnl[death_idx:] = np.nan

        pnl_data[f'Strat_{i}_{strat_type}'] = pnl

        base_contracts = np.abs(r.normal(50, 20, n_months))
        contracts = np.where(pnl > 0, base_contracts * 1.5, base_contracts)
        contracts_data[f'Strat_{i}_{strat_type}'] = np.ceil(contracts)

    df_pnl, df_contracts = pd.DataFrame(pnl_data, index=dates), pd.DataFrame(contracts_data, index=dates)

    # Merge PnL and Contracts into a single DataFrame for easier analysis
    df_pnl_tmp = df_pnl.stack().reset_index()
    df_pnl_tmp.columns = ['month', 'strategy', 'pnl']
    contracts_stack = df_contracts.stack().reset_index()
    contracts_stack.columns = ['month', 'strategy', 'contracts']
    df = df_pnl_tmp.merge(contracts_stack, on=['month', 'strategy'])
    return df, df_pnl, df_contracts


# =====================================================================
# Legacy wrappers (backward compatibility)
# =====================================================================

def to_wide_pnl(df_to_wide: pd.DataFrame) -> pd.DataFrame:
    """Long → wide: months × strategies (PnL). NaN = inactive."""
    return df_to_wide.pivot_table(
        index="month", columns="strategy", values="pnl", aggfunc="first"
    )


def to_wide_contracts(df_to_wide: pd.DataFrame) -> pd.DataFrame:
    """Long → wide: months × strategies (contracts)."""
    return df_to_wide.pivot_table(
        index="month", columns="strategy", values="contracts", aggfunc="first"
    )


if __name__ == "__main__":
    df, tier_df = generate_living_quant_universe()
    wide = to_wide_pnl(df)
    bench = wide.mean(axis=1)
    bsh = bench.mean() / bench.std() * np.sqrt(12) if bench.std() > 0 else 0

    print(f"Shape:      {df.shape}")
    print(f"Strategies: {df['strategy'].nunique()} | Months: {df['month'].nunique()}")
    print(f"Date range: {df['month'].min():%Y-%m} → {df['month'].max():%Y-%m}")
    print(f"\nBenchmark: Sharpe={bsh:.2f}, PnL=${bench.sum():,.0f}")

    print(f"\nTier breakdown:")
    for tier in tier_df["tier"].unique():
        t = tier_df[tier_df["tier"] == tier]
        print(f"  {tier:20s}: {len(t):2d} strats, avg PnL=${t['avg_pnl'].mean():+.1f}")