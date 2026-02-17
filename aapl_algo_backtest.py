#!/usr/bin/env python3  # Use the system Python 3 interpreter.
"""Simple NVDA SMA backtest over 1 year with adjustable initial investment.

Strategy: SMA crossover (short vs long). Uses daily close prices.
"""  # Module docstring describing the script.

from __future__ import annotations  # Enable forward references in type hints.

from dataclasses import dataclass  # Create lightweight config class.
from datetime import datetime, timedelta, timezone  # Handle date ranges.

import matplotlib.pyplot as plt  # Plot equity curve.
import pandas as pd  # Time series data handling.
import yfinance as yf  # Download market data.


@dataclass  # Generate boilerplate for config storage.
class BacktestConfig:
    ticker: str = "NVDA"  # Stock ticker to trade.
    initial_cash: float = 10_000.0  # Starting cash amount.
    short_window: int = 20  # Short SMA window length.
    long_window: int = 50  # Long SMA window length.
    years: int = 1  # Lookback period in years.

# Fetches daily closing prices for the last year + buffer_days and convert to a 1D time series
def fetch_prices(ticker: str, years: int, buffer_days: int) -> pd.Series:
    end = datetime.now(timezone.utc).date()  # Use timezone-aware current UTC date.
    start = end - timedelta(days=365 * years + buffer_days)  # Add SMA warm-up buffer.
    data = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False)  # Pull data.
    if data.empty:
        raise ValueError(f"No data returned for {ticker}")  # Fail fast if no data.
    close = data["Close"]  # Extract close prices.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]  # Handle MultiIndex/single-column DataFrame output from yfinance.
    close = pd.to_numeric(close, errors="coerce").dropna()  # Ensure numeric series.
    if close.empty:
        raise ValueError(f"No close prices available for {ticker}")  # Validate usable data exists.
    return close  # Return cleaned close prices.


def generate_signals(prices: pd.Series, short_window: int, long_window: int) -> pd.Series:
    if short_window >= long_window:
        raise ValueError("short_window must be less than long_window")  # Ensure valid windows.
    short_sma = prices.rolling(window=short_window).mean()  # Compute short SMA.
    long_sma = prices.rolling(window=long_window).mean()  # Compute long SMA.
    signal = (short_sma > long_sma).astype(int)  # 1 when trend up, 0 otherwise.
    return signal.fillna(0)  # Treat warm-up period as no-position.


def run_backtest(prices: pd.Series, signal: pd.Series, initial_cash: float) -> pd.DataFrame:
    cash = initial_cash  # Track available cash.
    shares = 0.0  # Track shares held.
    holding = "CASH"  # Current position state.
    equity_curve = []  # Store daily portfolio snapshots.

    for date, price in prices.items():  # Iterate over each trading day.
        signal_value = int(signal.loc[date])  # Daily SMA signal: 1 = invested, 0 = cash.
        if signal_value == 1:
            desired_state = "INVESTED"  # short_sma > long_sma, so hold the stock.
        elif signal_value == 0:
            desired_state = "CASH"  # short_sma <= long_sma, so cash out.
        else:
            raise ValueError(f"Error signal value on {date}: {signal_value}")

        if holding != desired_state:
            if holding == "INVESTED":
                cash = shares * price  # Sell NVDA at today's close.
                shares = 0.0
            if desired_state == "INVESTED":
                shares = cash / price  # Buy NVDA with all capital.
                cash = 0.0
            holding = desired_state

        equity = cash if holding == "CASH" else shares * price  # Total portfolio value.
        equity_curve.append(
            {"date": date, "equity": equity, "cash": cash, "shares": shares, "holding": holding}
        )  # Log state.

    return pd.DataFrame(equity_curve).set_index("date")  # Return time series.


def main() -> None:
    cfg = BacktestConfig()  # Edit BacktestConfig defaults above to change inputs.

    buffer_days = cfg.long_window + 5  # Ensure long SMA has enough lookback.
    prices_all = fetch_prices(cfg.ticker, cfg.years, buffer_days)  # Load NVDA price data with buffer.
    signal_all = generate_signals(prices_all, cfg.short_window, cfg.long_window)  # Create SMA signal.
    cutoff = prices_all.index.max() - pd.DateOffset(years=cfg.years)  # Calculate explicit backtest start.
    prices = prices_all.loc[prices_all.index >= cutoff]  # Keep only last N years.
    signal = signal_all.loc[prices.index]  # Align signals to backtest window.
    equity = run_backtest(prices=prices, signal=signal, initial_cash=cfg.initial_cash)  # Simulate long-or-cash.

    strategy_final_value = equity["equity"].iloc[-1]  # Final strategy portfolio value.
    total_return = strategy_final_value / cfg.initial_cash - 1.0  # Strategy return.
    bh_equity = cfg.initial_cash * (prices / prices.iloc[0])  # Buy & hold curve.
    bh_final_value = bh_equity.iloc[-1]  # Final buy & hold portfolio value.
    bh_return = bh_final_value / cfg.initial_cash - 1.0  # Buy & hold return.

    print(f"{cfg.ticker} SMA long-or-cash ({cfg.short_window}/{cfg.long_window})")  # Name.
    print(f"Initial cash: ${cfg.initial_cash:,.2f}")  # Print initial cash.
    print(f"SMA strategy ({cfg.years}Y): return={total_return:.2%}, final value=${strategy_final_value:,.2f}")  # Print strategy summary.
    print(f"{cfg.ticker} buy & hold ({cfg.years}Y): return={bh_return:.2%}, final value=${bh_final_value:,.2f}")  # Print benchmark summary.


if __name__ == "__main__":
    main()  # Run the script if executed directly.
