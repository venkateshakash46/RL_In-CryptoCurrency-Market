import argparse, json, os, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.portfolio_rl import load_prices, DQNAgent, backtest, annualized_metrics, save_plots

def main():
    p = argparse.ArgumentParser(description="Train and evaluate a DQN cryptocurrency portfolio allocator.")
    p.add_argument("--data", default="data/crypto_portfolio.csv")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--lookback", type=int, default=180)
    p.add_argument("--rebalance", type=int, default=30)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--transaction-cost", type=float, default=0.001, help="One-way cost rate, e.g. 0.001 = 10 bps")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--demo", action="store_true", help="Mark this run as synthetic/demo if using generated demo prices.")
    p.add_argument("--output", default="outputs")
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    prices = load_prices(args.data)
    n = len(prices)
    split = int(n * args.train_ratio)
    if split <= args.lookback + args.rebalance or n - split < args.rebalance:
        raise ValueError(f"Not enough data for lookback={args.lookback}. Rows={n}, split={split}. Use a longer series or smaller lookback.")
    # Training transitions are collected only before the split. Start after lookback.
    agent = DQNAgent(prices.shape[1], seed=args.seed)
    train_returns, _, losses = backtest(
        prices, agent, args.lookback, args.rebalance, args.lookback, split,
        args.transaction_cost, strategy="dqn", train=True
    )

    for ep in range(max(0, args.episodes - 1)):
        _, _, ep_losses = backtest(
            prices, agent, args.lookback, args.rebalance, args.lookback, split,
            args.transaction_cost, strategy="dqn", train=True
        )
        losses.extend(ep_losses)

    test_start = split
    dqn_ret, allocations, _ = backtest(prices, agent, args.lookback, args.rebalance,
                                        test_start, n, args.transaction_cost, strategy="dqn", train=False)
    equal_ret, _, _ = backtest(prices, None, args.lookback, args.rebalance,
                                test_start, n, args.transaction_cost, strategy="equal", train=False)
    hold_ret, _, _ = backtest(prices, None, args.lookback, args.rebalance,
                               test_start, n, args.transaction_cost, strategy="buy_hold", train=False)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    results = {"dqn": dqn_ret, "equal_weight": equal_ret, "buy_and_hold_equal_initial": hold_ret}
    metrics = []
    for name, series in results.items():
        turnover = float(allocations["turnover"].mean()) if name == "dqn" and not allocations.empty else 0.0
        metrics.append({"strategy": name, **annualized_metrics(series, turnover)})
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame({k: v.reset_index(drop=True) for k, v in results.items()}).to_csv(out / "equity_curves.csv", index=False)
    allocations.to_csv(out / "allocations.csv", index=False)
    pd.DataFrame({"loss": losses}).to_csv(out / "training_losses.csv", index=False)
    save_plots(results, allocations, out)
    agent.save(out / "dqn_model.pt", {"lookback": args.lookback, "rebalance": args.rebalance,
                                     "assets": list(prices.columns), "seed": args.seed})
    meta = vars(args) | {"rows": n, "assets": list(prices.columns), "split_date": str(prices.index[split]),
                         "train_end": str(prices.index[split-1]), "test_start": str(prices.index[split]),
                         "synthetic_demo": bool(args.demo), "warning": "Backtest results are historical and not investment advice."}
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print("\nOut-of-sample metrics:")
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"\nSaved artifacts to: {out.resolve()}")
    if args.demo:
        print("WARNING: This was marked as a demo run. Do not report synthetic-data metrics as real market performance.")

if __name__ == "__main__":
    main()
