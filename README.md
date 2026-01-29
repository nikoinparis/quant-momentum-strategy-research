# Quant Research Project 1  
## Momentum Strategy with Volatility Regime Filtering (SPY)

---

## Overview
This project implements and evaluates a **time-series momentum trading strategy** on SPY, and demonstrates how **volatility regime filtering** can materially improve risk-adjusted performance.

The goal of the project is not to optimize a single backtest, but to:
- build a clean research pipeline,
- validate performance across time using rolling windows,
- test robustness to parameter changes,
- and present results using standard financial metrics.

---

## Strategy Description

### Baseline: Momentum (60-day)
- Signal:
  
  \[
  s_t = \frac{P_t}{P_{t-60}} - 1
  \]

- Position rule:
  - Long (+1) if signal > 0  
  - Short (−1) if signal < 0  
  - Flat (0) otherwise  

---

### Volatility Regime Filter
- Rolling 20-day standard deviation of daily log returns
- Trading is allowed only when:

  \[
  \sigma_{20} \leq \text{vol\_threshold}
  \]

- When volatility exceeds the threshold, positions are gated to 0 (flat)

This filter is designed to reduce exposure during high-volatility regimes where momentum signals tend to be noisy.

---

## Data
- Instrument: **SPY (S&P 500 ETF)**
- Frequency: Daily
- Price field: Adjusted Close
- Source: `yfinance`
- Date range: 2015–2026

Downloaded price data is cached locally and **not committed** to the repository.

---

## Evaluation Metrics
Metrics are computed from daily log returns:

- Final Equity
- Annualized Return
- Annualized Volatility
- Sharpe Ratio (risk-free rate = 0)
- Maximum Drawdown
- Rolling 3-year Sharpe (monthly step)

---

## Key Results

### Aggregate Performance

| Strategy | Final Equity | Annual Return | Annual Vol | Sharpe | Max Drawdown |
|--------|--------------|---------------|------------|--------|--------------|
| Momentum (60d) | 1.916 | 6.12% | 17.75% | 0.33 | −25.55% |
| Vol-Filtered Momentum | 2.212 | 7.51% | 14.99% | 0.48 | −24.95% |

Volatility filtering improves both **returns** and **risk-adjusted performance**.

---

## Rolling Window Validation

Rolling 3-year Sharpe ratios (monthly step) show that the volatility-filtered strategy:
- has higher mean and median Sharpe,
- performs better across most market regimes,
- recovers faster after volatility shocks.

| Metric | Momentum | Vol-Filtered |
|------|----------|--------------|
| Mean Sharpe | 0.325 | **0.417** |
| Median Sharpe | 0.315 | **0.405** |
| Max Sharpe | 1.01 | **1.45** |

---

## Volatility Threshold Sensitivity

| Vol Threshold | Final Equity | Sharpe | Max Drawdown | Avg Trading % |
|--------------|--------------|--------|--------------|---------------|
| 0.015 | 2.64 | 0.67 | −18.7% | 88.3% |
| 0.020 | 2.31 | 0.51 | −24.6% | 96.1% |
| 0.025 | 2.19 | 0.46 | −24.3% | 97.2% |

Performance degrades smoothly as the filter loosens, indicating the improvement is **structural rather than overfit**.

---

## Figures

Key plots are saved under:

results/figures/

Including:
- Strategy equity curves
- Drawdown comparison
- Rolling 3-year Sharpe comparison

---

## Project Structure
.
├── src/
│   ├── strategies.py        # signals, position rules, regime filters
│   ├── backtester.py        # vectorized backtest engine
│   ├── metrics.py           # performance metrics
│   ├── data_loader.py       # price download + caching
│   └── config.py
├── scripts/
│   ├── make_dataset.py
│   ├── report_metrics.py
│   ├── grid_search_momentum.py
│   ├── rolling_window_vol_compare.py
│   └── vol_threshold_sensitivity.py
├── results/
│   └── figures/
├── README.md
├── pyproject.toml
└── .gitignore

---

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

python -m scripts.make_dataset
python -m scripts.report_metrics
python -m scripts.grid_search_momentum
python -m scripts.rolling_window_vol_compare
python -m scripts.vol_threshold_sensitivity
```


⸻

Notes
	•	This project is for research and educational purposes only
	•	No live trading or execution logic is included
	•	All analysis uses daily data and simple position sizing

⸻

Next Steps

This project serves as a foundation for:
	•	ML-based return forecasting,
	•	multi-asset portfolio construction,
	•	dynamic risk allocation and regime detection.
