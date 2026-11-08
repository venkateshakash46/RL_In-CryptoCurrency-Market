import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/demo_prices.csv")
    p.add_argument("--rows", type=int, default=900)
    p.add_argument("--assets", type=int, default=15)
    p.add_argument("--seed", type=int, default=13)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)

    common = rng.normal(0.0003, 0.025, size=a.rows)
    returns = np.column_stack([
        0.0002 + (0.35 + 0.03 * (i % 5)) * common + rng.normal(0, 0.018 + 0.002*(i % 4), a.rows)
        for i in range(a.assets)
    ])
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    dates = pd.date_range("2020-01-01", periods=a.rows, freq="D")
    df = pd.DataFrame(prices, columns=[f"ASSET_{i+1:02d}" for i in range(a.assets)])
    df.insert(0, "Date", dates.strftime("%Y-%m-%d"))
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.output, index=False)
    print(f"Wrote {a.rows} synthetic rows to {a.output}. NOT real market data.")
if __name__ == "__main__":
    main()
