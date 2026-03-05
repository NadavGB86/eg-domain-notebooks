# Strategy Selection & Portfolio Construction

A proof-of-concept exploring progressive strategy selection methods for
a multi-strategy trading universe — from theoretical bounds (oracle) to
implementable walk-forward systems with cross-strategy signal enrichment.

## Structure

- `strategy_selection.ipynb` — Main analysis notebook (8 parts)
- `quant_utils.py` — Metrics, portfolio construction, walk-forward backtester
- `synthetic_data.py` — Calibrated synthetic strategy universe generator

## Quick Start
```bash
pip install numpy pandas matplotlib scipy seaborn plotly
jupyter notebook strategy_selection.ipynb
```

## Key Findings

- **The Sharpe trap**: Greedy Sharpe optimization selects statistically
  Beautiful but commercially useless strategies
- **SUM aggregation**: When capital is free, selection is binary (on/off) —
  include everything above the quality bar
- **Adaptability beats hindsight**: Walk-forward selection outperforms
  static full-history optimization by adapting to regime changes
- **Cross-strategy signals add value**: Sector peer contagion and
  market momentum improves per-strategy scoring
