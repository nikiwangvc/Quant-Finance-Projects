
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import pandas as pd
import yfinance as yf

class BacktestConfig:
    ticker: str = 'AAPL'
    initial_cash: float = 10_000.0
    short_window: int = 20
    long_window: int = 50
    years: int = 1

# Fetch daily closing prices from Yahoo Finance
def fetch_prices(ticker: str, years: int, buffer_days: int) -> pd.Series:
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=365 * years + buffer_days)
    data = yf.download(ticker, start=start.isoformat(), end=end.isoformat(), progress=False)
    if data.empty:
        raise ValueError(f"No data returned for {ticker}")
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    if close.empty:
        raise ValueError(f"No close prices available for {ticker}")
    return close


# Generate long-or-cash trading signals from short and long moving averages.
def generate_signals(prices: pd.Series, short_window: int, long_window: int) -> pd.Series:
    if short_window >= long_window:
        raise ValueError("short_window must be less than long_window")
    short_sma = prices.rolling(window=short_window).mean()
    long_sma = prices.rolling(window=long_window).mean()
    signal = (short_sma > long_sma).astype(int)
    return signal.fillna(0)


# Simulate portfolio value by switching either 100% stock or 100% cash.
def run_backtest(prices: pd.Series, signal: pd.Series, initial_cash: float) -> pd.DataFrame:
    cash = initial_cash
    shares = 0.0
    holding = "CASH"
    equity_curve = []

    for date, price in prices.items():
        signal_value = int(signal.loc[date])
        if signal_value == 1:
            desired_state = "INVESTED"
        elif signal_value == 0:
            desired_state = "CASH"
        else:
            raise ValueError(f"Error signal value on {date}: {signal_value}")

        if holding != desired_state:
            if holding == "INVESTED":
                cash = shares * price
                shares = 0.0
            if desired_state == "INVESTED":
                shares = cash / price
                cash = 0.0
            holding = desired_state

        equity = cash if holding == "CASH" else shares * price
        equity_curve.append(
            {"date": date, "equity": equity, "cash": cash, "shares": shares, "holding": holding}
        )

    return pd.DataFrame(equity_curve).set_index("date")


# Run the backtest and print summary metrics.
def main() -> None:
    cfg = BacktestConfig()

    buffer_days = cfg.long_window + 5
    prices_all = fetch_prices(cfg.ticker, cfg.years, buffer_days)
    signal_all = generate_signals(prices_all, cfg.short_window, cfg.long_window)
    cutoff = prices_all.index.max() - pd.DateOffset(years=cfg.years)
    prices = prices_all.loc[prices_all.index >= cutoff]
    signal = signal_all.loc[prices.index]
    equity = run_backtest(prices=prices, signal=signal, initial_cash=cfg.initial_cash)

    strategy_final_value = equity["equity"].iloc[-1]
    total_return = strategy_final_value / cfg.initial_cash - 1.0
    bh_equity = cfg.initial_cash * (prices / prices.iloc[0])
    bh_final_value = bh_equity.iloc[-1]
    bh_return = bh_final_value / cfg.initial_cash - 1.0

    print(f"{cfg.ticker} SMA long-or-cash ({cfg.short_window}/{cfg.long_window})")
    print(f"Initial cash: ${cfg.initial_cash:,.2f}")
    print(f"SMA strategy ({cfg.years}Y): return={total_return:.2%}, final value=${strategy_final_value:,.2f}")
    print(f"{cfg.ticker} buy & hold ({cfg.years}Y): return={bh_return:.2%}, final value=${bh_final_value:,.2f}")


if __name__ == "__main__":
    main()
