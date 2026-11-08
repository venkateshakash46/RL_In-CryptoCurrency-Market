from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt

ACTIONS = {"HOLD": 0, "BUY": 1, "SELL": 2}


def load_prices(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() in ("date", "datetime", "timestamp")), None)
    if date_col is None:

        date_col = df.columns[0]
    dates = pd.to_datetime(df[date_col], errors="coerce")
    df = df.drop(columns=[date_col])
    df.index = dates
    df = df.loc[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if df.empty or df.shape[1] < 2:
        raise ValueError("CSV must contain a date column and at least two numeric price columns.")
    if (df <= 0).any().any():
        raise ValueError("Prices must be strictly positive.")
    if len(df) < 30:
        raise ValueError("Not enough rows. Provide at least 30 daily price observations.")
    return df.astype(float)


def covariance_state(prices: pd.DataFrame, t: int, lookback: int) -> np.ndarray:

    if t < lookback:
        raise ValueError(f"Need t >= lookback ({lookback}); received t={t}.")
    hist_prices = prices.iloc[t-lookback:t]
    returns = hist_prices.pct_change().dropna()
    cov = returns.cov().to_numpy(dtype=np.float32)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

    return ((cov + cov.T) / 2).astype(np.float32)


def annualized_metrics(daily_returns: pd.Series | np.ndarray, turnover: float = 0.0) -> Dict[str, float]:
    r = pd.Series(np.asarray(daily_returns, dtype=float)).dropna()
    if len(r) == 0:
        return {"total_return": 0.0, "annualized_volatility": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "calmar": 0.0, "turnover": float(turnover)}
    equity = (1.0 + r).cumprod()
    total = float(equity.iloc[-1] - 1.0)
    vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else 0.0
    mean = float(r.mean() * 252)
    sharpe = mean / vol if vol > 1e-12 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    mdd = float(drawdown.min())
    annualized_return = float(equity.iloc[-1] ** (252 / max(len(r), 1)) - 1.0) if equity.iloc[-1] > 0 else -1.0
    calmar = annualized_return / abs(mdd) if abs(mdd) > 1e-12 else 0.0
    return {"total_return": total, "annualized_volatility": vol, "sharpe": sharpe,
            "max_drawdown": mdd, "calmar": calmar, "turnover": float(turnover)}


def portfolio_period_reward(daily_asset_returns: np.ndarray, weights: np.ndarray,
                            transaction_cost: float = 0.001) -> Tuple[float, np.ndarray]:

    if daily_asset_returns.ndim != 2:
        raise ValueError("daily_asset_returns must be a (days, assets) matrix.")
    gross = daily_asset_returns @ weights

    net = gross.copy()
    if len(net):
        net[0] -= transaction_cost
    if len(net) < 2 or np.std(net, ddof=1) < 1e-10:
        reward = float(np.clip(np.mean(net) * 252, -5, 5)) if len(net) else 0.0
    else:
        reward = float(np.clip(np.mean(net) / (np.std(net, ddof=1) + 1e-8) * np.sqrt(252), -5, 5))
    return reward, net


class QNetwork(nn.Module):

    def __init__(self, n_assets: int):
        super().__init__()
        self.n_assets = n_assets
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_assets * n_assets, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU()
        )
        self.heads = nn.ModuleList([nn.Linear(64, 3) for _ in range(n_assets)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        return torch.stack([head(z) for head in self.heads], dim=1)  # batch, assets, actions


@dataclass
class Transition:
    state: np.ndarray
    actions: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class DQNAgent:
    def __init__(self, n_assets: int, gamma: float = 0.95, lr: float = 1e-3,
                 epsilon: float = 1.0, epsilon_min: float = 0.03,
                 epsilon_decay: float = 0.97, seed: int = 7, device: str | None = None):
        self.n_assets = n_assets
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(seed)
        self.model = QNetwork(n_assets).to(self.device)
        self.target = copy.deepcopy(self.model).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory: List[Transition] = []
        self.steps = 0

    def select_actions(self, state: np.ndarray, explore: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if explore and self.rng.random() < self.epsilon:
            actions = self.rng.integers(0, 3, size=self.n_assets)
            with torch.no_grad():
                q = self.model(torch.tensor(state[None, ...], dtype=torch.float32, device=self.device))[0].cpu().numpy()
            return actions, q
        with torch.no_grad():
            q = self.model(torch.tensor(state[None, ...], dtype=torch.float32, device=self.device))[0].cpu().numpy()
        return np.argmax(q, axis=1).astype(int), q

    def actions_to_weights(self, actions: np.ndarray, q_values: np.ndarray) -> np.ndarray:

        buy = actions == ACTIONS["BUY"]
        if not buy.any():
            return np.full(self.n_assets, 1.0 / self.n_assets)
        advantage = np.maximum(q_values[:, 1] - q_values[:, 0], 1e-6)
        scores = np.where(buy, advantage, 0.0)
        if not np.isfinite(scores).all() or scores.sum() <= 1e-12:
            scores = buy.astype(float)
        return scores / scores.sum()

    def remember(self, transition: Transition):
        self.memory.append(transition)
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def train_step(self, batch_size: int = 32) -> float | None:
        if len(self.memory) < batch_size:
            return None
        idx = self.rng.choice(len(self.memory), size=batch_size, replace=False)
        batch = [self.memory[i] for i in idx]
        states = torch.tensor(np.stack([x.state for x in batch]), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack([x.next_state for x in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.stack([x.actions for x in batch]), dtype=torch.long, device=self.device)
        rewards = torch.tensor([x.reward for x in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([x.done for x in batch], dtype=torch.float32, device=self.device)

        q_all = self.model(states)  # B, N, 3
        chosen_q = q_all.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q = self.target(next_states).max(dim=2).values  # B, N
            target = rewards[:, None] + self.gamma * (1.0 - dones[:, None]) * next_q
        loss = self.loss_fn(chosen_q, target.expand_as(chosen_q))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.steps += 1
        if self.steps % 25 == 0:
            self.target.load_state_dict(self.model.state_dict())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def save(self, path: str | Path, extra: dict | None = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "n_assets": self.n_assets,
                    "epsilon": self.epsilon, "extra": extra or {}}, path)


def backtest(prices: pd.DataFrame, agent: DQNAgent | None, lookback: int,
             rebalance: int, start_idx: int, end_idx: int,
             transaction_cost: float = 0.001, strategy: str = "dqn",
             train: bool = False, batch_size: int = 32) -> Tuple[pd.Series, pd.DataFrame, list[float]]:

    returns = prices.pct_change().fillna(0.0).to_numpy()
    dates = prices.index
    n_assets = prices.shape[1]
    equity_returns, allocation_rows, losses = [], [], []
    previous_weights = np.full(n_assets, 1.0 / n_assets)
    t = max(start_idx, lookback)
    while t < end_idx:
        state = covariance_state(prices, t, lookback)
        next_t = min(t + rebalance, end_idx)
        if strategy == "equal":
            weights = np.full(n_assets, 1.0 / n_assets)
            actions = np.zeros(n_assets, dtype=int)
        elif strategy == "buy_hold":
            if not allocation_rows:
                weights = np.full(n_assets, 1.0 / n_assets)
            else:
                weights = previous_weights.copy()
            actions = np.zeros(n_assets, dtype=int)
        else:
            assert agent is not None
            actions, q = agent.select_actions(state, explore=train)
            weights = agent.actions_to_weights(actions, q)
        turnover = float(np.sum(np.abs(weights - previous_weights)))
        interval = returns[t:next_t].copy()
        if len(interval):
            if strategy == "buy_hold":

                daily_net = []
                drifting_weights = weights.copy()
                for day_i, asset_ret in enumerate(interval):
                    portfolio_ret = float(asset_ret @ drifting_weights)
                    if day_i == 0:
                        portfolio_ret -= transaction_cost * turnover
                    daily_net.append(portfolio_ret)
                    denom = max(1.0 + float(asset_ret @ drifting_weights), 1e-12)
                    drifting_weights = drifting_weights * (1.0 + asset_ret) / denom
                net = np.asarray(daily_net, dtype=float)
                weights_after_interval = drifting_weights
            else:
                gross = interval @ weights
                net = gross.copy()
                net[0] -= transaction_cost * turnover
                weights_after_interval = weights.copy()
            if train and agent is not None:

                if len(net) > 1 and np.std(net, ddof=1) > 1e-10:
                    reward = float(np.clip(np.mean(net) / (np.std(net, ddof=1) + 1e-8) * np.sqrt(252), -5, 5))
                else:
                    reward = float(np.clip(np.mean(net) * 252, -5, 5))
                next_state = covariance_state(prices, next_t, lookback) if next_t < len(prices) else state.copy()
                done = next_t >= end_idx
                agent.remember(Transition(state, actions, reward, next_state, done))
                loss = agent.train_step(batch_size)
                if loss is not None:
                    losses.append(loss)
            equity_returns.extend(net.tolist())
            allocation_rows.append({"date": dates[t], "turnover": turnover,
                                    **{str(c): float(w) for c, w in zip(prices.columns, weights)}})
            previous_weights = weights_after_interval.copy()
        t = next_t
    idx = dates[max(start_idx, lookback):max(start_idx, lookback) + len(equity_returns)]
    return pd.Series(equity_returns, index=idx[:len(equity_returns)], name=strategy), pd.DataFrame(allocation_rows), losses


def save_plots(returns: Dict[str, pd.Series], allocations: pd.DataFrame, out_dir: str | Path):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 5))
    for name, r in returns.items():
        if len(r):
            plt.plot(r.index, (1 + r).cumprod(), label=name.replace("_", " ").title())
    plt.title("Out-of-sample equity curves")
    plt.xlabel("Date"); plt.ylabel("Growth of $1"); plt.grid(alpha=.25); plt.legend()
    plt.tight_layout(); plt.savefig(out / "performance.png", dpi=160); plt.close()

    plt.figure(figsize=(11, 4))
    for name, r in returns.items():
        if len(r):
            equity = (1 + r).cumprod()
            dd = equity / equity.cummax() - 1
            plt.plot(dd.index, dd, label=name.replace("_", " ").title())
    plt.title("Out-of-sample drawdown")
    plt.xlabel("Date"); plt.ylabel("Drawdown"); plt.grid(alpha=.25); plt.legend()
    plt.tight_layout(); plt.savefig(out / "drawdown.png", dpi=160); plt.close()

    if not allocations.empty:
        weight_cols = [c for c in allocations.columns if c not in ("date", "turnover")]
        if weight_cols:
            plt.figure(figsize=(12, 5))
            plt.stackplot(pd.to_datetime(allocations["date"]), allocations[weight_cols].to_numpy().T,
                          labels=weight_cols, alpha=.85)
            plt.title("DQN portfolio allocation over time")
            plt.xlabel("Rebalance date"); plt.ylabel("Portfolio weight")
            plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7)
            plt.tight_layout(); plt.savefig(out / "allocations.png", dpi=160); plt.close()
