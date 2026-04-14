import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import requests
from io import StringIO

# Get all current stocks in SP500 from Wikipedia
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    table = pd.read_html(StringIO(response.text))[0]

    return [ticker.replace(".", "-") for ticker in table["Symbol"]]

# Download closing prices from yfinance
def download_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]

    return data.dropna(how="all")

# Compute the compounded annual return from a series of returns periodically
def annualized_return(returns, periods_per_year=12):
    if len(returns) == 0:
        return np.nan

    total_growth = (1 + returns).prod()
    computed_annualized_return = total_growth ** (periods_per_year / len(returns)) - 1
    return computed_annualized_return

# Compute the sharpe ratio with risk-free rate of 0.02
def sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=12):
    if len(returns) < 2:
        return np.nan
    excess_returns = returns - (risk_free_rate / periods_per_year)
    std = excess_returns.std()
    if std == 0 or np.isnan(std):
        return np.nan

    computed_sharpe_ratio = (excess_returns.mean() / std) * np.sqrt(periods_per_year)
    return computed_sharpe_ratio

# Rebalance monthly into the top momentum stocks trading above their SMA
# Hold them equally for one period
# Compare the results to SPY.
def run_strategy(prices, stocks, benchmark, momentum_lookback, top_percent,
                 stock_sma_window=100):
    # Get the dates of the last row of a month end stock data
    rebalance_dates = prices.resample("ME").last().index
    strategy_returns = []
    spy_returns = []

    for i in range(1, len(rebalance_dates)):
        rebalance_date = rebalance_dates[i - 1]
        next_date = rebalance_dates[i]

        hist = prices.loc[:rebalance_date]

        spy_period = prices.loc[rebalance_date:next_date, benchmark].dropna()

        # Compute SPY returns
        if len(spy_period) >= 2:
            # Return = end / start - 1
            spy_ret = (spy_period.iloc[-1] / spy_period.iloc[0]) - 1
            spy_returns.append(spy_ret)

        scores = {}

        for stock in stocks:
            stock_hist = hist[stock].dropna()

            if len(stock_hist) >= max(momentum_lookback, stock_sma_window):
                current_price = stock_hist.iloc[-1]
                sma100 = stock_hist.rolling(stock_sma_window).mean().iloc[-1]

                if current_price > sma100:
                    # Compute momentum = (current_price / past_price) - 1
                    past_price = stock_hist.iloc[-momentum_lookback]
                    momentum = (current_price / past_price) - 1
                    scores[stock] = momentum

        if not scores:
            portfolio_return = 0.0
        else:
            # Select top performing stocks
            ranked = sorted(scores, key=scores.get, reverse=True)
            top_n = max(1, int(len(ranked) * top_percent))
            selected = ranked[:top_n]

            period_prices = prices.loc[rebalance_date:next_date, selected].dropna(axis=1, how="any")
            # Our table cannot be less than 2 rows or no column
            if period_prices.shape[0] < 2 or period_prices.shape[1] == 0:
                portfolio_return = 0.0
            else:
                start_prices = period_prices.iloc[0]
                end_prices = period_prices.iloc[-1]

                stock_returns = (end_prices / start_prices) - 1
                portfolio_return = stock_returns.mean()

        strategy_returns.append(portfolio_return)

    strategy_returns = pd.Series(strategy_returns)
    spy_returns = pd.Series(spy_returns[:len(strategy_returns)])

    return {
        "strategy_total_return": (1 + strategy_returns).prod() - 1,
        "strategy_annual_return": annualized_return(strategy_returns, periods_per_year=12),
        "strategy_sharpe": sharpe_ratio(strategy_returns, periods_per_year=12),
        "spy_total_return": (1 + spy_returns).prod() - 1,
        "spy_annual_return": annualized_return(spy_returns, periods_per_year=12),
        "spy_sharpe": sharpe_ratio(spy_returns, periods_per_year=12),
    }

if __name__ == "__main__":
    stocks = get_sp500_tickers()

    start = dt.datetime(2020, 1, 1)
    end = dt.datetime(2025, 1, 1)

    all_tickers = ["SPY"] + stocks
    prices = download_prices(all_tickers, start, end)
    prices.index = pd.to_datetime(prices.index)

    prices = prices.dropna(axis=1, how="all")
    stocks = [ticker for ticker in stocks if ticker in prices.columns]

    tests = [
        {"name": "Top 20%, 3-month momentum", "momentum_lookback": 63, "top_percent": 0.20},
        {"name": "Top 10%, 3-month momentum", "momentum_lookback": 63, "top_percent": 0.10},
        {"name": "Top 20%, 6-month momentum", "momentum_lookback": 126, "top_percent": 0.20},
        {"name": "Top 10%, 6-month momentum", "momentum_lookback": 126, "top_percent": 0.10},
        {"name": "Top 20%, 12-month momentum", "momentum_lookback": 252, "top_percent": 0.20},
        {"name": "Top 10%, 12-month momentum", "momentum_lookback": 252, "top_percent": 0.10},
    ]

    results_list = []
    for test in tests:
        results = run_strategy(
            prices=prices,
            stocks=stocks,
            benchmark="SPY",
            momentum_lookback=test["momentum_lookback"],
            top_percent=test["top_percent"]
        )

        print("-" * 50)
        print(test["name"])
        print("Strategy Results")
        print(f"Total Return:   {results['strategy_total_return'] * 100:.2f}%")
        print(f"Annual Return:  {results['strategy_annual_return'] * 100:.2f}%")
        print(f"Sharpe Ratio:   {results['strategy_sharpe']:.2f}")
        print()
        print("SPY Buy-and-Hold Results")
        print(f"Total Return:   {results['spy_total_return'] * 100:.2f}%")
        print(f"Annual Return:  {results['spy_annual_return'] * 100:.2f}%")
        print(f"Sharpe Ratio:   {results['spy_sharpe']:.2f}")
        if results["strategy_sharpe"] > results["spy_sharpe"]:
            print("Strategy returns > Buy_and_hold_returns")
        print()
