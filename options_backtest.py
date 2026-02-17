#!/usr/bin/env python3
"""Backtest a simple covered-call strategy using historical stock prices.

This uses Black-Scholes with historical volatility to estimate option premium.
It is a simplification (no dividends, no transaction costs, constant risk-free rate).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm


@dataclass
class BacktestConfig:
    ticker: str
    years: int = 10
    otm_pct: float = 0.05
    risk_free_rate: float = 0.01
    vol_lookback_days: int = 60
    min_vol: float = 0.10


def black_scholes_call_price(spot: float, strike: float, tau: float, r: float, vol: float) -> float:
    if tau <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        return max(0.0, spot - strike)
    d1 = (math.log(spot / strike) + (r + 0.5 * vol**2) * tau) / (vol * math.sqrt(tau))
    d2 = d1 - vol * math.sqrt(tau)
    return spot * norm.cdf(d1) - strike * math.exp(-r * tau) * norm.cdf(d2)


def compute_hist_vol(prices: pd.Series, lookback: int, min_vol: float) -> float:
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if len(log_returns) < lookback:
        return min_vol
    vol = log_returns.tail(lookback).std() * math.sqrt(252)
    return max(vol, min_vol)


def backtest_covered_call(prices: pd.Series, cfg: BacktestConfig) -> pd.DataFrame:
    month_ends = prices.resample("M").last().dropna()
    results = []

    equity = 1.0
    for i in range(len(month_ends) - 1):
        entry_date = month_ends.index[i]
        exit_date = month_ends.index[i + 1]
        s0 = month_ends.iloc[i]
        st = month_ends.iloc[i + 1]

        hist_prices = prices.loc[:entry_date]
        vol = compute_hist_vol(hist_prices, cfg.vol_lookback_days, cfg.min_vol)
        strike = s0 * (1.0 + cfg.otm_pct)
        tau = max((exit_date - entry_date).days / 365.0, 1 / 365.0)
        premium = black_scholes_call_price(s0, strike, tau, cfg.risk_free_rate, vol)

        payoff = min(st, strike) - s0 + premium
        monthly_return = payoff / s0
        equity *= 1.0 + monthly_return

        results.append(
            {
                "entry_date": entry_date,
                "exit_date": exit_date,
                "s0": s0,
                "st": st,
                "strike": strike,
                "vol": vol,
                "premium": premium,
                "monthly_return": monthly_return,
                "equity": equity,
            }
        )

    return pd.DataFrame(results)


def backtest_buy_hold(prices: pd.Series, start_date: pd.Timestamp) -> pd.Series:
    sliced = prices.loc[start_date:]
    start_price = sliced.iloc[0]
    return sliced / start_price


def fetch_prices(ticker: str, years: int) -> pd.Series:
    end = datetime.utcnow().date()
    start = end - timedelta(days=365 * years + 120)
    data = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False)
    if data.empty:
        raise ValueError(f"No data returned for {ticker}")
    return data["Close"].dropna()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest a covered-call strategy")
    parser.add_argument("ticker", help="Stock ticker, e.g. AAPL")
    parser.add_argument("--years", type=int, default=10)
    parser.add_argument("--otm", type=float, default=0.05, help="OTM strike percent, e.g. 0.05 = 5%%")
    parser.add_argument("--rf", type=float, default=0.01, help="Risk-free rate")
    parser.add_argument("--vol-lookback", type=int, default=60)
    args = parser.parse_args()

    cfg = BacktestConfig(
        ticker=args.ticker,
        years=args.years,
        otm_pct=args.otm,
        risk_free_rate=args.rf,
        vol_lookback_days=args.vol_lookback,
    )

    prices = fetch_prices(cfg.ticker, cfg.years)
    strategy = backtest_covered_call(prices, cfg)
    if strategy.empty:
        raise ValueError("Not enough data for backtest")

    start_date = strategy["entry_date"].iloc[0]
    bh = backtest_buy_hold(prices, start_date)

    total_return = strategy["equity"].iloc[-1] - 1.0
    bh_return = bh.iloc[-1] - 1.0

    print(f"Covered Call total return: {total_return:.2%}")
    print(f"Buy & Hold total return: {bh_return:.2%}")
    print("\nLast 5 months:")
    print(strategy.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
