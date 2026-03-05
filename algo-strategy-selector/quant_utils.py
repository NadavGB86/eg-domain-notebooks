"""
Quant Metrics & Portfolio Construction Utilities
======================================================
Reusable metrics, portfolio construction, and walk-forward backtesting tools
for strategy selection under SUM (free capital) aggregation.
"""
from __future__ import annotations
import inspect
import numpy as np
import pandas as pd
from typing import Callable, Optional, Any
import matplotlib.pyplot as plt
from pandas import Timedelta
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# ============================================================
# CONSTANTS
# ============================================================
MOCK = True
SEED = 42
ANNUALIZATION_FACTOR = np.sqrt(12)   # Monthly → Annual Sharpe
LOOKBACK_DEFAULT = 12                # Walk-forward lookback window (months)
TOP_K_DEFAULT = None                 # All positive-scoring strategies (SUM: capital is free)
ALPHA_DEFAULT = 1.0                  # Sharpe-PPC blending (calibrated in Part 4)
MIN_HISTORY = 6                      # Minimum months before a strategy i

np.random.seed(SEED)


# ============================================================
# METRICS
# ============================================================

def compute_sharpe(returns: pd.Series, annualize: bool = True) -> float:
    """Sharpe ratio (Rf = 0). Annualized from monthly by default."""
    clean = returns.dropna()
    if len(clean) < 2:
        return np.nan
    mu, sigma = clean.mean(), clean.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return np.nan
    sharpe = mu / sigma
    return sharpe * ANNUALIZATION_FACTOR if annualize else sharpe


def compute_ppc(pnl: pd.Series, contracts: pd.Series) -> float:
    """Profit Per Contract = total_pnl / total_contracts."""
    total_c = contracts.sum()
    return pnl.sum() / total_c if total_c > 0 else np.nan


def compute_max_drawdown(cumulative_pnl: pd.Series) -> float:
    """Max peak-to-trough decline in cumulative PnL (negative number)."""
    running_max = cumulative_pnl.cummax()
    return (cumulative_pnl - running_max).min()


def combined_score(sharpe: float, ppc: float, alpha: float = ALPHA_DEFAULT) -> float:
    """Combined metric: Sharpe × log(1 + α·PPC).

    Multiplicative gate: strategy must have BOTH positive Sharpe AND positive
    PPC to score well. Log dampens extreme PPC (diminishing returns to
    efficiency). Alpha controls PPC sensitivity.

    Returns NaN for negative Sharpe (filter out losers) or NaN inputs.
    """
    if np.isnan(sharpe) or np.isnan(ppc):
        return np.nan
    if sharpe <= 0:
        return np.nan
    return sharpe * np.log1p(max(alpha * ppc, 0))


def summary_metrics(
    pnl: pd.Series,
    contracts: Optional[pd.Series] = None,
    label: str = "",
) -> dict:
    """Full performance summary for a PnL series."""
    cum = pnl.cumsum()
    result = {
        "label": label,
        "total_pnl": pnl.sum(),
        "sharpe": compute_sharpe(pnl),
        "max_drawdown": compute_max_drawdown(cum),
        "n_months": len(pnl),
        "pct_profitable": (pnl > 0).mean() * 100,
        "avg_monthly_pnl": pnl.mean(),
        "monthly_vol": pnl.std(),
    }
    if contracts is not None:
        result["ppc"] = compute_ppc(pnl, contracts)
        result["total_contracts"] = contracts.sum()
    return result


# ============================================================
# DATA MANIPULATION
# ============================================================

def to_wide_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """Long → wide pivot: months × strategies (PnL). NaN = inactive."""
    return df.pivot_table(index="month", columns="strategy", values="pnl", aggfunc="first")


def to_wide_contracts(df: pd.DataFrame) -> pd.DataFrame:
    """Long → wide pivot: months × strategies (contracts)."""
    return df.pivot_table(index="month", columns="strategy", values="contracts", aggfunc="first")


def per_strategy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-strategy Sharpe, PPC, combined score, lifetime, profitability."""
    rows = []
    for name, g in df.groupby("strategy"):
        s = compute_sharpe(g["pnl"])
        p = compute_ppc(g["pnl"], g["contracts"])
        rows.append({
            "strategy": name,
            "sharpe": s,
            "ppc": p,
            "combined": combined_score(s, p),
            "total_pnl": g["pnl"].sum(),
            "total_contracts": g["contracts"].sum(),
            "n_months": len(g),
            "avg_pnl": g["pnl"].mean(),
            "vol": g["pnl"].std(),
            "pct_profitable": (g["pnl"] > 0).mean() * 100,
            "first_month": g["month"].min(),
            "last_month": g["month"].max(),
        })
    return pd.DataFrame(rows).set_index("strategy").sort_values("sharpe", ascending=False)


def validate_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Validate panel data: required columns, no duplicates, sorted."""
    required = {"month", "strategy", "pnl", "contracts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])
    n_before = len(df)
    df = df.drop_duplicates(subset=["month", "strategy"], keep="first")
    if (dropped := n_before - len(df)) > 0:
        print(f"WARNING: Dropped {dropped} duplicate (month, strategy) rows")
    return df.sort_values(["month", "strategy"]).reset_index(drop=True)


# ============================================================
# BENCHMARK & ORACLE
# ============================================================

def compute_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Benchmark: all strategies, every month.
    Uses SUM — capital is free (Rf=0), each strategy runs independently.
    """
    return df.groupby("month").agg(
        pnl=("pnl", "sum"),
        contracts=("contracts", "sum"),
        n_strategies=("strategy", "count"),
    ).reset_index()


def compute_oracle(df_orc: pd.DataFrame) -> pd.DataFrame:
    """Oracle: perfect foresight — each month, activate all profitable strategies.

    Under SUM aggregation (capital is free), every positive strategy adds value.
    This is the theoretical ceiling: clip(lower=0).sum().
    """
    pos = df_orc[df_orc["pnl"] > 0]
    oracle_ = pos.groupby("month").agg(
        pnl=("pnl", "sum"),
        contracts=("contracts", "sum"),
        n_strategies=("strategy", "count"),
    ).reset_index()
    # Add months with zero positives
    all_months = pd.DataFrame({"month": sorted(df_orc["month"].unique())})
    oracle_ = all_months.merge(oracle_, on="month", how="left").fillna(0)
    oracle_[["contracts", "n_strategies"]] = (
        oracle_[["contracts", "n_strategies"]].astype(int)
    )
    return oracle_


# ============================================================
# PORTFOLIO CONSTRUCTION
# ============================================================

def portfolio_pnl_equal_weight(wide_pnl: pd.DataFrame, selected: list[str]) -> pd.Series:
    """Equal-weight portfolio PnL for selected strategies."""
    return wide_pnl[selected].mean(axis=1)


def greedy_forward_select(
    wide_pnl: pd.DataFrame,
    metric_fn: Callable[..., float] = compute_sharpe,
    wide_contracts: pd.DataFrame | None = None,
    max_strategies: int = 50,
    min_strategies: int = 3,
    verbose: bool = False,
) -> list[str]:
    """Greedy forward selection maximizing a portfolio-level metric.

    At each step, adds the strategy that most improves the portfolio metric.
    Stops when no addition improves the metric (after min_strategies reached).
    Portfolio PnL = SUM of selected strategies' PnL (capital is free).

    metric_fn signature:
        - f(pnl: pd.Series) -> float                        (e.g. compute_sharpe)
        - f(pnl: pd.Series, contracts: pd.Series) -> float  (e.g. combined)

    If metric_fn accepts 2 positional args, wide_contracts must be provided.
    """
    sig = inspect.signature(metric_fn)
    n_required = len([
        p for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
    ])
    needs_contracts = n_required >= 2

    if needs_contracts and wide_contracts is None:
        raise ValueError(
            "metric_fn requires contracts but wide_contracts was not provided"
        )

    candidates = list(wide_pnl.columns)
    selected: list[str] = []
    current_score = -np.inf

    for step in range(min(max_strategies, len(candidates))):
        best_candidate, best_score = None, -np.inf

        for c in candidates:
            trial = selected + [c]
            port_pnl = wide_pnl[trial].sum(axis=1).dropna()

            if needs_contracts:
                port_contracts = wide_contracts[trial].sum(axis=1).dropna()
                score = metric_fn(port_pnl, port_contracts)
            else:
                score = metric_fn(port_pnl)

            if not np.isnan(score) and score > best_score:
                best_score = score
                best_candidate = c

        if best_candidate is None:
            break

        # After minimum reached, only add if it strictly improves
        if step >= min_strategies and best_score <= current_score:
            break

        selected.append(best_candidate)
        candidates.remove(best_candidate)
        current_score = best_score

        if verbose:
            print(f"  Step {step+1}: +{best_candidate} → score={current_score:.4f}")

    return selected


def filter_select(
    df: pd.DataFrame,
    metric_fn: Callable[[float, float, float], float] = combined_score,
    alpha: float = ALPHA_DEFAULT,
    threshold: float = 0.0,
    ppc_floor: float = 0.0,
) -> list[str]:
    """Filter-based selection: include all strategies passing a quality bar.

    Under SUM (capital is free), every strategy with positive score adds value.
    Each inclusion decision is independent — no portfolio-level optimization.

    The PPC floor provides α-dependent breadth control: higher α → higher
    minimum PPC → fewer strategies pass → more concentrated but efficient.

    Args:
        df: Long-format panel with month, strategy, pnl, contracts.
        metric_fn: Per-strategy scoring function(sharpe, ppc, alpha) -> float.
        alpha: PPC sensitivity parameter for combined_score.
        threshold: Minimum combined score for inclusion (default 0).
        ppc_floor: Minimum PPC for inclusion (default 0). Use α × median_ppc × k
                   for data-adaptive breadth control.

    Returns:
        List of strategy names passing both thresholds, sorted by score descending.
    """
    scores = {}
    for name, g in df.groupby("strategy"):
        s = compute_sharpe(g["pnl"])
        p = compute_ppc(g["pnl"], g["contracts"])
        if np.isnan(s) or np.isnan(p):
            continue
        if p < ppc_floor:
            continue
        sc = metric_fn(s, p, alpha)
        if not np.isnan(sc) and sc > threshold:
            scores[name] = sc
    return sorted(scores, key=scores.get, reverse=True)


# ============================================================
# WALK-FORWARD SELECTOR
# ============================================================

class WalkForwardSelector:
    """Monthly strategy selection using only historical data.

    Temporal integrity: at decision time T, ALL information used comes
    from times strictly < T. This is enforced in compute_signals().

    Signal types:
        - "flat" (default): Equal-weight rolling window signals.
        - "ewma": Exponentially weighted signals. Recent data weighted more.
          Reacts faster to regime changes. Always uses combined score.

    Selection modes:
        - "rank" (default): Score strategies individually, pick top-K.
          Simple, robust on short windows. No correlation awareness.
        - "greedy": Greedy forward selection on lookback window each month.
          Portfolio-level metric, correlation-aware. Best with EWMA signals.
        - "cluster": Cluster-aware round-robin. Ensures diversification
          across correlation clusters before concentrating in any one.

    Enhancement toggles (composable, off by default):
        - turnover_penalty: Penalize newcomers to reduce churn.
        - regime_adaptive: Adjust K based on rolling market volatility.

    Allocation modes:
        - "equal": 1/N weighting. Zero estimation error.
        - "inv_vol" (default): Inverse-volatility. Risk parity principle.
        - "signal_prop": Proportional to signal strength.
        - "markowitz": Mean-variance optimized. Fragile OOS with limited data.
    """

    SCORE_FNS = {"sharpe", "ppc", "momentum", "consistency", "combined"}
    ALLOCATION_MODES = {"equal", "inv_vol", "signal_prop", "markowitz"}
    SIGNAL_TYPES = {"flat", "ewma"}
    SELECTION_MODES = {"rank", "greedy", "cluster"}

    def __init__(
        self,
        lookback_months: int = LOOKBACK_DEFAULT,
        top_k: int | None = TOP_K_DEFAULT,
        score_fn: str = "combined",
        allocation: str = "inv_vol",
        alpha: float = ALPHA_DEFAULT,
        min_history: int = MIN_HISTORY,
        selection_mode: str = "rank",
        # --- Signal type ---
        signal_type: str = "flat",
        ewma_halflife: int = 6,
        # --- Enhancement toggles ---
        turnover_penalty: float = 0.0,
        # --- Signal quality enhancements (Part 7) ---
        sector_map: dict[str, int] | None = None,
        peer_penalty: float = 0.0,
        cs_momentum_boost: float = 0.0,
        adaptive_halflife: bool = False,
        cluster_aware: bool = False,
        n_clusters: int = 3,
        regime_adaptive: bool = False,
        vol_window: int = 12,
        k_calm: int | None = None,
        k_volatile: int | None = None,
    ):
        self.lookback_months = lookback_months
        self.top_k = top_k
        self.score_fn = score_fn
        self.allocation = allocation
        self.alpha = alpha
        self.min_history = min_history
        self.signal_type = signal_type
        self.ewma_halflife = ewma_halflife
        self.turnover_penalty = turnover_penalty
        self.sector_map = sector_map or {}
        self.peer_penalty = peer_penalty
        self.cs_momentum_boost = cs_momentum_boost
        self.adaptive_halflife = adaptive_halflife
        self.regime_adaptive = regime_adaptive
        self.vol_window = vol_window
        self.k_calm = k_calm if k_calm is not None else (top_k + 2 if top_k is not None else None)
        self.k_volatile = k_volatile if k_volatile is not None else (max(top_k - 2, 2) if top_k is not None else None)
        self.n_clusters = n_clusters

        # Cluster-aware overrides selection_mode
        if cluster_aware:
            selection_mode = "cluster"
        self.selection_mode = selection_mode

        if score_fn not in self.SCORE_FNS:
            raise ValueError(f"Unknown score_fn '{score_fn}'. Choose from {self.SCORE_FNS}")
        if allocation not in self.ALLOCATION_MODES:
            raise ValueError(f"Unknown allocation '{allocation}'. Choose from {self.ALLOCATION_MODES}")
        if selection_mode not in self.SELECTION_MODES:
            raise ValueError(f"Unknown selection_mode '{selection_mode}'. Choose from {self.SELECTION_MODES}")
        if signal_type not in self.SIGNAL_TYPES:
            raise ValueError(f"Unknown signal_type '{signal_type}'. Choose from {self.SIGNAL_TYPES}")

        # Mutable state — reset on each backtest()
        self._prev_selected: set[str] = set()
        self.monthly_selections: dict[pd.Timestamp, list[str]] = {}
        self.monthly_weights: dict[pd.Timestamp, dict[str, float]] = {}
        self.results_df: Optional[pd.DataFrame] = None

    def compute_signals(self, df: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        """Score each strategy using data STRICTLY BEFORE as_of.

        This is the temporal firewall - no data from time >= as_of is touched.
        Dispatches to flat or EWMA signal computation based on signal_type.
        """
        if self.signal_type == "ewma":
            signals = self._compute_signals_ewma(df, as_of)
        else:
            signals = self._compute_signals_flat(df, as_of)

        # Only strategies active in the last observed month are eligible.
        # EWMA preserves reputation across gaps, but capital cannot be
        # allocated to a strategy that has stopped trading.
        history = df[df["month"] < as_of]
        if not history.empty:
            last_month = history["month"].max()
            alive = set(history[history["month"] == last_month]["strategy"])
            signals = signals[signals.index.isin(alive)]

        return signals

    def _compute_signals_flat(self, df: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        """Flat rolling window signals — equal weight on all observations."""
        history = df[df["month"] < as_of]
        if self.lookback_months:
            cutoff = as_of - pd.DateOffset(months=self.lookback_months)
            history = history[history["month"] >= cutoff]

        signals = {}
        for strat, g in history.groupby("strategy"):
            if len(g) < self.min_history:
                continue
            if self.score_fn == "sharpe":
                signals[strat] = compute_sharpe(g["pnl"])
            elif self.score_fn == "ppc":
                signals[strat] = compute_ppc(g["pnl"], g["contracts"])
            elif self.score_fn == "momentum":
                signals[strat] = g["pnl"].sum()
            elif self.score_fn == "consistency":
                signals[strat] = (g["pnl"] > 0).mean()
            elif self.score_fn == "combined":
                s = compute_sharpe(g["pnl"])
                p = compute_ppc(g["pnl"], g["contracts"])
                signals[strat] = combined_score(s, p, self.alpha)

        return pd.Series(signals).dropna().sort_values(ascending=False)

    def _get_effective_halflife(self, df: pd.DataFrame, as_of: pd.Timestamp) -> int:
        """Adaptive EWMA halflife based on trailing market volatility.

        High vol → shorter halflife (react faster to regime changes).
        Low vol → longer halflife (more stable estimates).
        """
        history = df[df["month"] < as_of]
        market = history.groupby("month")["pnl"].mean().sort_index()
        if len(market) < self.vol_window:
            return self.ewma_halflife
        vol = market.rolling(self.vol_window).std()
        recent = vol.iloc[-1]
        median = vol.dropna().median()
        if np.isnan(recent) or np.isnan(median) or median == 0:
            return self.ewma_halflife
        if recent > median * 1.3:
            return max(self.ewma_halflife // 2, 2)
        if recent < median * 0.7:
            return min(self.ewma_halflife * 3 // 2, 18)
        return self.ewma_halflife

    def _compute_signals_ewma(self, df: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        """EWMA signals — exponential decay, recent data weighted more.

        Always uses combined score (Sharpe × log(1 + α·PPC)).
        The EWMA Sharpe estimate uses ewm().mean() and ewm().std() at the
        last available observation, giving a recency-weighted risk-adjusted score.
        """
        history = df[df["month"] < as_of]
        if self.lookback_months:
            cutoff = as_of - pd.DateOffset(months=self.lookback_months)
            history = history[history["month"] >= cutoff]

        signals = {}
        for strat, g in history.groupby("strategy"):
            if len(g) < self.min_history:
                continue
            g_sorted = g.sort_values("month")
            pnl = g_sorted["pnl"]
            contracts = g_sorted["contracts"]
            hl = self._get_effective_halflife(df, as_of) if self.adaptive_halflife else self.ewma_halflife
            ewm = pnl.ewm(halflife=hl)
            mu, sigma = ewm.mean().iloc[-1], ewm.std().iloc[-1]
            if sigma <= 0 or np.isnan(sigma):
                continue
            s = (mu / sigma) * ANNUALIZATION_FACTOR

            if self.score_fn == "sharpe":
                signals[strat] = s
            elif self.score_fn == "ppc":
                signals[strat] = compute_ppc(pnl, contracts)
            elif self.score_fn == "momentum":
                # EWMA-weighted cumulative: use ewm mean × N as proxy
                signals[strat] = mu * len(pnl)
            elif self.score_fn == "consistency":
                signals[strat] = (pnl > 0).mean()
            elif self.score_fn == "combined":
                p = compute_ppc(pnl, contracts)
                signals[strat] = combined_score(s, p, self.alpha)

        return pd.Series(signals).dropna().sort_values(ascending=False)

    def _select_rank(self, signals: pd.Series) -> list[str]:
        """Top-K strategies by individual signal score. Only keeps positive.
        top_k=None → all positive-scoring (variable K under SUM).
        """
        positive = signals[signals > 0]
        if len(positive) == 0:
            return []
        if self.top_k is None:
            return list(positive.index)
        return list(positive.head(self.top_k).index)

    def _select_greedy(self, df: pd.DataFrame, as_of: pd.Timestamp,
                       signals: pd.Series | None = None,
                       k: int | None = None) -> list[str]:
        """Greedy forward selection on lookback window — portfolio-level metric.

        When signals are provided (e.g. from EWMA):
          - Only strategies with positive signal are candidates
          - Highest-signal strategy seeds the portfolio
          - Greedy evaluates diversification on historical data
        This is the EWMA+greedy interaction: EWMA provides a better-ordered
        menu, greedy builds a diversified portfolio from it.
        """
        raw_k = k if k is not None else self.top_k
        history = df[df["month"] < as_of]
        if self.lookback_months:
            cutoff = as_of - pd.DateOffset(months=self.lookback_months)
            history = history[history["month"] >= cutoff]

        w_pnl = history.pivot_table(index="month", columns="strategy",
                                    values="pnl", aggfunc="first")
        w_contracts = history.pivot_table(index="month", columns="strategy",
                                          values="contracts", aggfunc="first")

        obs_counts = w_pnl.notna().sum()
        eligible = obs_counts[obs_counts >= self.min_history].index.tolist()

        # Pre-filter using signals if available
        if signals is not None:
            positive_signals = signals[signals > 0]
            eligible = [s for s in positive_signals.index if s in eligible]

        effective_k = raw_k if raw_k is not None else len(eligible)
        if len(eligible) < 2:
            return eligible[:effective_k]

        w_pnl = w_pnl[eligible]
        w_contracts = w_contracts[eligible]

        if self.score_fn in ("sharpe", "momentum", "consistency"):
            selected = greedy_forward_select(
                w_pnl, metric_fn=compute_sharpe,
                max_strategies=effective_k, min_strategies=2, verbose=False,
            )
        else:
            def _metric(pnl: pd.Series, contracts: pd.Series) -> float:
                s = compute_sharpe(pnl)
                p = pnl.sum() / contracts.sum() if contracts.sum() > 0 else 0.0
                return combined_score(s, p, self.alpha)

            selected = greedy_forward_select(
                w_pnl, metric_fn=_metric, wide_contracts=w_contracts,
                max_strategies=effective_k, min_strategies=2, verbose=False,
            )
        return selected

    def _modify_signals(self, signals: pd.Series) -> pd.Series:
        """Apply turnover penalty to newcomers (if enabled).

        Strategies not in previous selection get their score reduced by
        turnover_penalty. Encourages persistence, reduces churn.
        """
        if self.turnover_penalty <= 0 or not self._prev_selected:
            return signals
        adj = signals.copy()
        for s in adj.index:
            if s not in self._prev_selected:
                adj[s] -= self.turnover_penalty
        return adj.sort_values(ascending=False)

    def _enhance_signals(self, signals: pd.Series, df: pd.DataFrame, as_of: pd.Timestamp, ) -> pd.Series:
        """Apply signal-quality enhancements using cross-strategy information.

        Unlike _modify_signals (which penalizes newcomers), these enhancements
        use cross-sectional patterns to improve per-strategy score accuracy.
        Each enhancement can push marginal scores below zero → exclusion.

        Enhancements (composable, additive):
          - peer_penalty: If same-sector peers have negative recent returns,
            penalize the strategy. Exploits sector contagion (lag-1 cross-corr).
          - cs_momentum_boost: If market trend aligns with strategy beta,
            boost the score. Exploits cross-sectional momentum.
          - adaptive_halflife: Already applied in compute_signals (modifies
            EWMA halflife before scoring). Not applied here.
        """
        if self.peer_penalty <= 0 and self.cs_momentum_boost <= 0:
            return signals

        history = df[df["month"] < as_of]
        if self.lookback_months:
            cutoff = as_of - pd.DateOffset(months=self.lookback_months)
            history = history[history["month"] >= cutoff]

        w_pnl = history.pivot_table(index="month", columns="strategy", values="pnl", aggfunc="first")
        adj = signals.copy()

        # --- Sector peer penalty ---
        if self.peer_penalty > 0 and self.sector_map:
            for strat in adj.index:
                sector = self.sector_map.get(strat)
                if sector is None:
                    continue
                peers = [s for s in adj.index if self.sector_map.get(s) == sector and s != strat and s in w_pnl.columns]
                if not peers or len(w_pnl) < 3:
                    continue
                peer_recent = w_pnl[peers].iloc[-3:].mean().mean()
                if peer_recent < 0:
                    penalty = self.peer_penalty * abs(peer_recent) / 15
                    adj[strat] -= penalty

        # --- Cross-sectional momentum boost ---
        if self.cs_momentum_boost > 0 and len(w_pnl) >= 6:
            market = w_pnl.mean(axis=1)
            mkt_trend = market.iloc[-3:].mean()
            for strat in adj.index:
                series = w_pnl.get(strat)
                if series is None:
                    continue
                series = series.dropna()
                common = series.index.intersection(market.index)
                if len(common) < 6:
                    continue
                cov_mat = np.cov(series.reindex(common), market.reindex(common))
                var_m = market.reindex(common).var()
                beta = cov_mat[0, 1] / var_m if var_m > 0 else 0
                alignment = beta * mkt_trend
                adj[strat] += self.cs_momentum_boost * alignment / 30

        return adj.sort_values(ascending=False)

    def _determine_k(self, df: pd.DataFrame, as_of: pd.Timestamp) -> int | None:
        """Return top_k or regime-adaptive K based on rolling volatility.

        In volatile regimes: fewer strategies (concentrate on best).
        In calm regimes: more strategies (broader diversification).
        Uses only data strictly before as_of (temporal integrity).
        """
        if not self.regime_adaptive:
            return self.top_k  # None = all positive-scoring
        if self.top_k is None:
            return None        # all-positive mode, regime doesn't restrict

        history = df[df["month"] < as_of]
        monthly_agg = history.groupby("month")["pnl"].mean().sort_index()
        if len(monthly_agg) < self.vol_window:
            return self.top_k

        rolling_vol = monthly_agg.rolling(self.vol_window).std()
        recent_vol = rolling_vol.iloc[-1]
        if np.isnan(recent_vol):
            return self.top_k

        hist_median = rolling_vol.dropna().median()
        return self.k_volatile if recent_vol > hist_median else self.k_calm

    def _select_cluster(self, signals: pd.Series, df: pd.DataFrame,
                        as_of: pd.Timestamp, k: int | None) -> list[str]:
        """Cluster-aware round-robin selection.

        1. Cluster strategies by correlation on lookback window.
        2. Round-robin: take best from each cluster until K filled.
        Ensures diversification across correlation groups.
        k=None → all positive-scoring.
        """
        positive = signals[signals > 0]
        effective_k = k if k is not None else len(positive)
        if len(positive) < 3:
            return list(positive.head(effective_k).index)

        # Build correlation matrix from lookback window
        history = df[df["month"] < as_of]
        if self.lookback_months:
            cutoff = as_of - pd.DateOffset(months=self.lookback_months)
            history = history[history["month"] >= cutoff]

        w_pnl = history.pivot_table(
            index="month", columns="strategy", values="pnl", aggfunc="first"
        )
        common = sorted(set(positive.index) & set(w_pnl.columns))
        if len(common) < 3:
            return list(positive.head(k).index)

        corr = w_pnl[common].corr()
        dist = np.clip((1 - corr.abs()).to_numpy(copy=True, dtype=float), 0, 2)
        np.fill_diagonal(dist, 0)
        dist = np.nan_to_num(dist, nan=1.0)
        condensed = squareform(dist, checks=False)
        Z = linkage(condensed, method="ward")
        n_clust = min(self.n_clusters, len(common))
        labels = fcluster(Z, t=n_clust, criterion="maxclust")
        strat_clusters = dict(zip(common, labels))

        # Build per-cluster queues ordered by signal
        cluster_queues: dict[int, list[str]] = {}
        for s in positive.index:
            if s in strat_clusters:
                c = strat_clusters[s]
                cluster_queues.setdefault(c, []).append(s)

        # Round-robin across clusters
        selected = []
        active_clusters = sorted(cluster_queues.keys())
        while len(selected) < effective_k and active_clusters:
            exhausted = []
            for c in active_clusters:
                if cluster_queues[c]:
                    selected.append(cluster_queues[c].pop(0))
                    if len(selected) >= effective_k:
                        break
                else:
                    exhausted.append(c)
            for c in exhausted:
                active_clusters.remove(c)

        return selected

    def select(self, signals: pd.Series, df: pd.DataFrame = None,
               as_of: pd.Timestamp = None, k: int | None = None) -> list[str]:
        """Select strategies based on selection_mode.

        k overrides self.top_k if provided (used by regime-adaptive).
        """
        effective_k = k if k is not None else self.top_k

        if self.selection_mode == "cluster" and df is not None and as_of is not None:
            return self._select_cluster(signals, df, as_of, effective_k)
        if self.selection_mode == "greedy" and df is not None and as_of is not None:
            return self._select_greedy(df, as_of, signals=signals, k=effective_k)
        # rank mode
        positive = signals[signals > 0]
        if len(positive) == 0:
            return []
        if effective_k is None:
            return list(positive.index)
        return list(positive.head(effective_k).index)

    def allocate(self, selected: list[str], signals: pd.Series,
                 history: pd.DataFrame) -> dict[str, float] | dict[Any, Any] | dict[Any, Timedelta]:
        """Assign weights to selected strategies."""
        n = len(selected)
        if n == 0:
            return {}

        if self.allocation == "equal":
            return {s: 1.0 for s in selected}

        elif self.allocation == "signal_prop":
            vals = signals.reindex(selected).fillna(0)
            total = vals.sum()
            if total <= 0:
                return {s: 1.0 / n for s in selected}
            return (vals / total).to_dict()

        elif self.allocation == "inv_vol":
            vols = {}
            for s in selected:
                g = history[history["strategy"] == s]
                v = g["pnl"].std()
                if v > 0:
                    vols[s] = v
            if not vols:
                return {s: 1.0 / n for s in selected}
            inv = pd.Series({s: 1.0 / v for s, v in vols.items()})
            covered = set(inv.index)
            uncovered = [s for s in selected if s not in covered]
            if uncovered:
                inv_sum = inv.sum()
                covered_weight = inv_sum / (inv_sum + len(uncovered))
                weights = (inv / inv.sum() * covered_weight).to_dict()
                unc_w = (1 - covered_weight) / len(uncovered)
                for s in uncovered:
                    weights[s] = unc_w
                return weights
            return (inv / inv.sum()).to_dict()

        elif self.allocation == "markowitz":
            return self._allocate_markowitz(selected, history)

        raise ValueError(f"Unknown allocation: {self.allocation}")

    def _allocate_markowitz(self, selected: list[str],
                            history: pd.DataFrame) -> dict[str, float]:
        """Mean-variance optimized weights via SLSQP.

        Maximizes Sharpe on the lookback window. Falls back to equal weight
        if optimization fails or insufficient data. Expected to underperform
        simpler methods OOS due to estimation error (DeMiguel et al. 2009).
        """
        n = len(selected)
        w_pnl = history.pivot_table(index="month", columns="strategy",
                                    values="pnl", aggfunc="first")
        avail = [s for s in selected if s in w_pnl.columns]
        if len(avail) < 2:
            return {s: 1.0 / n for s in selected}
        sub = w_pnl[avail].dropna()
        if len(sub) < len(avail) + 2:
            return {s: 1.0 / n for s in selected}

        def neg_sharpe(w: np.ndarray) -> float:
            port = sub.values @ w
            mu, sigma = port.mean(), port.std()
            if sigma == 0:
                return 0.0
            return -(mu / sigma) * ANNUALIZATION_FACTOR

        n_a = len(avail)
        constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
        bounds = [(0.0, 1.0)] * n_a
        x0 = np.ones(n_a) / n_a

        result = minimize(neg_sharpe, x0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 200})

        if result.success:
            weights = dict(zip(avail, result.x))
        else:
            weights = {s: 1.0 / n_a for s in avail}

        for s in selected:
            if s not in weights:
                weights[s] = 1.0 / n
        total = sum(weights.values())
        if total > 0:
            weights = {s: w / total for s, w in weights.items()}
        return weights

    def backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full walk-forward backtest with 5-stage pipeline.

        Pipeline per decision month T:
          1. SIGNAL:  compute_signals(df, T)  — flat or EWMA
          2. MODIFY:  _modify_signals(signals) — turnover penalty
          3. K:       _determine_k(df, T)     — fixed or regime-adaptive
          4. SELECT:  select(signals, df, T, k) — rank / greedy / cluster
          5. ALLOCATE + RECORD

        NaN handling: inactive strategies contribute 0 PnL (cash drag).
        """
        all_months = sorted(df["month"].unique())
        warmup = self.lookback_months or self.min_history

        # Reset mutable state
        self._prev_selected = set()
        self.monthly_selections = {}
        self.monthly_weights = {}

        results = []
        for i, T in enumerate(all_months):
            if i < warmup:
                month_data = df[df["month"] == T]
                results.append({
                    "month": T, "pnl": month_data["pnl"].sum(),
                    "contracts": month_data["contracts"].sum(),
                    "n_selected": len(month_data), "is_warmup": True,
                })
                continue

            # 1. SIGNAL
            signals = self.compute_signals(df, as_of=T)

            # 2a. MODIFY (turnover penalty)
            signals = self._modify_signals(signals)

            # 2b. ENHANCE (cross-strategy signal quality)
            signals = self._enhance_signals(signals, df, as_of=T)

            # 3. K (regime-adaptive)
            k = self._determine_k(df, as_of=T)

            # 4. SELECT
            selected = self.select(signals, df=df, as_of=T, k=k)
            self.monthly_selections[T] = selected
            self._prev_selected = set(selected)

            month_data = df[df["month"] == T]
            if not selected:
                results.append({"month": T, "pnl": 0.0, "contracts": 0,
                                "n_selected": 0, "is_warmup": False})
                continue

            # 5. ALLOCATE + RECORD
            history = df[df["month"] < T]
            weights = self.allocate(selected, signals, history)
            self.monthly_weights[T] = weights

            active = month_data.set_index("strategy")
            port_pnl, port_contracts = 0.0, 0.0
            for s in selected:
                if s in active.index:
                    w = weights.get(s, 1.0)
                    port_pnl += active.loc[s, "pnl"] * w
                    port_contracts += active.loc[s, "contracts"] * w

            results.append({
                "month": T, "pnl": port_pnl, "contracts": port_contracts,
                "n_selected": len(selected), "is_warmup": False,
            })

        self.results_df = pd.DataFrame(results)
        return self.results_df

    def get_summary(self) -> dict:
        """Performance summary of the backtest."""
        if self.results_df is None:
            raise RuntimeError("Call backtest() first")
        r = self.results_df[~self.results_df["is_warmup"]]
        tags = []
        if self.signal_type != "flat":
            tags.append(self.signal_type)
        if self.selection_mode != "rank":
            tags.append(self.selection_mode)
        if self.allocation != "equal":
            tags.append(self.allocation)
        if self.turnover_penalty > 0:
            tags.append(f"tp={self.turnover_penalty}")
        if self.peer_penalty > 0:
            tags.append(f"peer={self.peer_penalty}")
        if self.cs_momentum_boost > 0:
            tags.append(f"cs={self.cs_momentum_boost}")
        if self.adaptive_halflife:
            tags.append("adapt_hl")
        if self.regime_adaptive:
            tags.append("regime")
        extra = ",".join(tags)
        k_str = "all" if self.top_k is None else str(self.top_k)
        label = f"WF({self.score_fn}, K={k_str}, L={self.lookback_months}"
        if extra:
            label += f", {extra}"
        label += ")"
        return summary_metrics(r["pnl"], r["contracts"], label=label)


# ============================================================
# VISUALIZATION
# ============================================================

def show_or_fig(notebook, obj) -> Optional[plt.Figure]:
    """helper viz return statement"""
    if notebook:
        obj.show()
        return None
    else:
        return obj


def plot_cumulative_pnl(
    series_dict: dict[str, pd.Series],
    title: str = "Cumulative PnL Comparison",
    notebook: bool = True,
) -> Optional[plt.Figure]:
    """Overlay cumulative PnL curves for multiple approaches."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, pnl in series_dict.items():
        cum = pnl.cumsum()
        ax.plot(cum.index, cum.values, label=label, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.legend(loc="upper left")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    fig.tight_layout()
    show_or_fig(notebook, plt if notebook else fig)


def build_comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """Formatted comparison table from {approach: summary_metrics_dict}."""
    rows = []
    for name, m in results.items():
        rows.append({
            "Approach": name,
            "Total PnL": f"${m.get('total_pnl', 0):,.0f}",
            "Sharpe": f"{m.get('sharpe', 0):.2f}",
            "PPC": f"{m.get('ppc', 0):.4f}" if m.get("ppc") is not None else "N/A",
            "Max DD": f"${m.get('max_drawdown', 0):,.0f}",
            "% Profitable": f"{m.get('pct_profitable', 0):.1f}%",
            "Avg Monthly": f"${m.get('avg_monthly_pnl', 0):,.1f}",
        })
    return pd.DataFrame(rows).set_index("Approach")


def plot_selection_heatmap(
    selections: dict[pd.Timestamp, list[str]],
    all_strategies: list[str],
    title: str = "Strategy Selection Over Time",
    notebook: bool = True,
) -> Optional[plt.Figure]:
    """Binary heatmap: which strategies were selected each month."""
    months = sorted(selections.keys())
    matrix = pd.DataFrame(0, index=months, columns=all_strategies)
    for m, strats in selections.items():
        for s in strats:
            if s in matrix.columns:
                matrix.loc[m, s] = 1
    ever = matrix.columns[matrix.sum() > 0]
    matrix = matrix[ever]

    fig, ax = plt.subplots(figsize=(14, max(4, len(ever) * 0.3)))
    ax.imshow(matrix.T.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(len(ever)))
    ax.set_yticklabels(ever, fontsize=8)
    step = max(1, len(months) // 12)
    ax.set_xticks(range(0, len(months), step))
    ax.set_xticklabels([m.strftime("%Y-%m") for m in months[::step]], rotation=45, fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    show_or_fig(notebook, plt if notebook else fig)


def plot_parameter_sensitivity(
    param_name: str,
    param_values: list,
    metrics: list[dict],
    metric_keys: list[str] = ("sharpe", "total_pnl"),
    notebook: bool = True,
) -> Optional[plt.Figure]:
    """Plot how performance varies with a parameter."""
    fig, axes = plt.subplots(1, len(metric_keys), figsize=(6 * len(metric_keys), 5))
    if len(metric_keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, metric_keys):
        values = [m.get(key, np.nan) for m in metrics]
        ax.plot(param_values, values, "o-", linewidth=2)
        ax.set_xlabel(param_name)
        ax.set_ylabel(key)
        ax.set_title(f"{key} vs {param_name}")
    fig.tight_layout()
    show_or_fig(notebook, plt if notebook else fig)


# ============================================================
# DIAGNOSTICS
# ============================================================

def compute_turnover(selections: dict[pd.Timestamp, list[str]]) -> pd.Series:
    """Monthly turnover: fraction of strategies that changed vs prior month."""
    months = sorted(selections.keys())
    if len(months) < 2:
        return pd.Series(dtype=float)
    turnover = {}
    for i in range(1, len(months)):
        prev, curr = set(selections[months[i-1]]), set(selections[months[i]])
        turnover[months[i]] = len(curr - prev) / len(curr) if curr else np.nan
    return pd.Series(turnover).sort_index()


def compute_hhi(weights: dict[str, float]) -> float:
    """Herfindahl-Hirschman Index: sum of squared weights.
    1/N = perfectly diversified. 1.0 = fully concentrated."""
    if not weights:
        return np.nan
    return sum(w**2 for w in weights.values())
