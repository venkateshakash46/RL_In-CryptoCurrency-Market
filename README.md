# Deep RL for Risk-Aware Cryptocurrency Portfolio Optimization

A reproducible, research-oriented upgrade of the original `PortfolioAllocation.ipynb`.
It uses a DQN with a covariance-matrix state, per-asset HOLD/BUY/SELL action values,
a forward-period risk-adjusted reward, a chronological train/test split, transaction
costs, passive benchmarks, risk metrics, charts, and an optional Streamlit dashboard.

> **Research/education only.** This is not investment advice. Historical backtests do not
> guarantee future performance. The original notebook's CSV (`data/crypto_portfolio.csv`)
> was not included with the uploaded notebook, so you must supply it for real-data results.
> A synthetic-data mode is included only to smoke-test the code; never present synthetic
> results as historical cryptocurrency performance.

## Project layout

```text
crypto_dqn_portfolio_project/
├── app.py                  # Optional interactive dashboard
├── train.py                # Train, evaluate, save results and model
├── make_demo_data.py       # Creates synthetic data for a software smoke test
├── requirements.txt
├── README.md
├── data/README.md
├── src/
│   ├── __init__.py
│   └── portfolio_rl.py     # Data, DQN, weights, backtest, metrics, plots
└── tests/
    └── test_core.py
```

## 1. Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Add your historical data

Create `data/crypto_portfolio.csv` with a date column and one column per asset:

```csv
Date,BTC,ETH,SOL,...
2021-01-01,29374,730, ...
```

- One row per date, oldest to newest.
- Values should be positive adjusted/close prices, not returns.
- The uploaded notebook expects 15 assets; this implementation supports any number of assets.
- Make sure there are no duplicate dates and no missing/zero prices. Do not mix future dates into training.
- Data must cover at least `lookback + 2` rows; a longer history is strongly recommended.

The source notebook references `data/crypto_portfolio.csv`, but that CSV was not present among the uploaded files. Put your real file at that path, or use `--data path/to/file.csv`.

## 3. Run the pipeline

With real data:

```bash
python train.py --data data/crypto_portfolio.csv --episodes 20 --lookback 180 --rebalance 30
```

Software smoke test only (synthetic data):

```bash
python make_demo_data.py --output data/demo_prices.csv
python train.py --data data/demo_prices.csv --episodes 3 --lookback 60 --rebalance 15 --demo
```

Outputs are saved in `outputs/`:
- `metrics.csv`: out-of-sample metrics for DQN, equal-weight and buy-and-hold.
- `equity_curves.csv`: out-of-sample equity curves.
- `allocations.csv`: portfolio weights at each rebalance.
- `performance.png`, `drawdown.png`, `allocations.png`: figures.
- `dqn_model.pt`: trained model weights and configuration.

`--demo` is a reminder flag recorded in the run metadata; it does not change the algorithm.

## 4. Optional dashboard

After running training:

```bash
streamlit run app.py
```

The dashboard reads files from `outputs/`. Run training first.

## Methodology

### State
At decision date `t`, the state is the covariance matrix of daily percentage returns over
the preceding `lookback` **price observations**. With 180 price rows, `pct_change()` gives
179 return rows, and the covariance matrix is `N_assets × N_assets`.

### Actions and weights
The network predicts three Q-values for each asset: HOLD, BUY, SELL. The agent selects
one action per asset. In the default **long-only** configuration, BUY-selected assets
receive normalized positive weights and HOLD/SELL assets receive zero; if no asset is
selected to BUY, the portfolio falls back to equal weights. The weight builder uses the
agent's action and Q-values; the DQN does not directly output weights.

### Reward and transition
The selected allocation is held over the following rebalance interval. Daily portfolio
returns are calculated using the weights known at that decision date, with transaction
costs deducted on rebalance. The training reward is a clipped, annualized Sharpe-like
score computed from that forward interval. The same scalar portfolio reward is assigned
to each asset's selected action because portfolio performance is global.

### Evaluation
Data is split chronologically into training and test periods. The test set is not used to
fit the DQN. The first test decision uses only historical prices available at that date.
Benchmarks are evaluated on the same test dates and with the same cost assumptions where
applicable.

Metrics:
- Total return
- Annualized volatility
- Annualized Sharpe ratio (risk-free rate assumed 0)
- Maximum drawdown
- Calmar ratio
- Turnover

## Important limitations

- This is a compact, educational multi-head DQN, not a claim of state-of-the-art RL.
- A single portfolio-level reward is shared across asset-specific action heads; credit
  assignment is approximate.
- Covariance estimates are noisy and sensitive to the lookback length.
- Results depend on the assets, dates, data quality, hyperparameters, random seed, and fees.
- Try several seeds and walk-forward validation before making any performance claim.
- The original notebook's action/weight and evaluation code had implementation quirks; this
  project uses a clearer, testable version rather than silently treating its output as valid.

## Suggested project-report sections

1. Problem statement and motivation
2. Data and preprocessing
3. MDP formulation: state, action, reward, transition
4. DQN architecture and training
5. Baselines and transaction-cost assumptions
6. Out-of-sample results and risk analysis
7. Limitations and future work

Do not claim the DQN is "optimal" unless you have evidence across held-out periods and
robustness tests.
