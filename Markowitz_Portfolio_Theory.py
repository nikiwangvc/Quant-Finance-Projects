import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as optimization

TOTAL_TRADING_DAYS = 252
TOTAL_PORTFOLIOS = 10000
RISK_FREE_RATE = 0.02

stocks = ['AAPL', 'AMZN', 'MSFT', 'COST', 'CTAS']
TOTAL_STOCKS = len(stocks)

start_date = '2015-01-01'
end_date = '2025-01-01'

# Download adjusted closing-price history for the selected stocks and align dates across all series.
def download_price_data():
    stock_data = {}
    for stock in stocks:
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date, auto_adjust=True)['Close']
    return pd.DataFrame(stock_data).dropna()


# Plot the historical prices to compare visually in a line chart.
def plot_price_data(price_data):
    price_data.plot(figsize=(8, 4))
    plt.show()


# Converts prices into daily log returns: log(today_price/yesterday_price).
def compute_log_returns(price_data):
    log_returns = np.log(price_data / price_data.shift(1))
    return log_returns.dropna()


# Annualize each asset's expected return and covariance matrix from daily return data.
def compute_annualized_asset_metrics(returns):
    annualized_returns = returns.mean() * TOTAL_TRADING_DAYS
    annualized_cov = returns.cov() * TOTAL_TRADING_DAYS
    return annualized_returns, annualized_cov


# Compute a portfolio's expected annual return and annualized volatility for a given weight vector.
def compute_portfolio_return_and_volatility(returns, weights):
    portfolio_annual_return = np.dot(weights, returns.mean()) * TOTAL_TRADING_DAYS
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(returns.cov() * TOTAL_TRADING_DAYS, weights))
    )
    return portfolio_annual_return, portfolio_volatility


# Generate many random portfolios.
# Monte Carlo Portfolio Simulation.
def simulate_random_portfolios(returns):
    portfolio_returns = []
    portfolio_volatilities = []
    portfolio_weights = []

    # Normalize weights so the sum is equal to 1.
    # Compute return and vol for each one.
    for _ in range(TOTAL_PORTFOLIOS):
        w = np.random.random(TOTAL_STOCKS)
        w = w / np.sum(w)

        portfolio_weights.append(w)

        annual_return, annual_volatility = compute_portfolio_return_and_volatility(
            returns, w
        )
        portfolio_returns.append(annual_return)
        portfolio_volatilities.append(annual_volatility)

    return (
        np.array(portfolio_returns),
        np.array(portfolio_volatilities),
        np.array(portfolio_weights),
    )


# Summarize portfolio with return, volatility, sharpe ratio.
def compute_portfolio_statistics(weights, returns):
    portfolio_return, portfolio_volatility = compute_portfolio_return_and_volatility(
        returns, weights
    )
    sharpe_ratio = (portfolio_return - RISK_FREE_RATE) / portfolio_volatility
    return np.array([portfolio_return, portfolio_volatility, sharpe_ratio])


# Negate the Sharpe ratio to maximize it by minimizing this objective using Scipy.
def negative_sharpe_ratio(weights, returns):
    return -compute_portfolio_statistics(weights, returns)[2]


# Solve for the portfolio with the Max Sharpe ratio.
def find_max_sharpe_portfolio(candidate_weights, returns):
    constraint = {
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1
    }

    bounds = tuple((0, 1) for _ in range(TOTAL_STOCKS))

    return optimization.minimize(
        fun=negative_sharpe_ratio,
        x0=candidate_weights[0],
        args=(returns,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraint
    )


# Print the optimized portfolio weights and metrics.
def print_optimal_portfolio_summary(optimum, returns, stocks):
    print("Optimal portfolio weights:")
    for stock, weight in zip(stocks, optimum['x']):
        print(f"{stock}: {weight:.3f}")

    stats = compute_portfolio_statistics(optimum['x'], returns)
    print(f"Expected annual return: {stats[0]:.3f}")
    print(f"Expected annual volatility: {stats[1]:.3f}")
    print(f"Sharpe ratio: {stats[2]:.3f}")


# Plot all simulated portfolios and the Max Sharpe Ratio.
def plot_portfolio_scatter(simulated_returns, simulated_volatilities, optimum, returns):
    plt.figure(figsize=(8, 4))
    plt.scatter(
        simulated_volatilities,
        simulated_returns,
        c=simulated_returns / simulated_volatilities,
        marker='o'
    )
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')

    optimal_stats = compute_portfolio_statistics(optimum['x'], returns)
    optimal_return = optimal_stats[0]
    optimal_volatility = optimal_stats[1]
    optimal_sharpe = optimal_stats[2]

    plt.plot(
        optimal_volatility,
        optimal_return,
        'gx',
        markersize=20,
        label='Optimal Portfolio'
    )

    plt.annotate(
        f"Max Sharpe Ratio\nx={optimal_volatility:.3f}, y={optimal_return:.3f}\nSR={optimal_sharpe:.3f}",
        (optimal_volatility, optimal_return),
        xytext=(10, 10),
        textcoords='offset points'
    )

    print(f"Optimal portfolio point -> x (volatility): {optimal_volatility:.3f}")
    print(f"Optimal portfolio point -> y (return): {optimal_return:.3f}")
    print(f"Optimal Sharpe ratio: {optimal_sharpe:.3f}")

    plt.legend()
    plt.show()

if __name__ == '__main__':
    price_data = download_price_data()
    #plot_price_data(price_data)

    log_returns = compute_log_returns(price_data)

    annual_returns, annual_cov = compute_annualized_asset_metrics(log_returns)

    print(f"Annualized Returns:\n {annual_returns.round(3)}\n")
    print(f"Annualized Covariance Matrix:\n {annual_cov.round(3)}\n")

    simulated_returns, simulated_volatilities, portfolio_weights = simulate_random_portfolios(log_returns)

    optimum = find_max_sharpe_portfolio(portfolio_weights, log_returns)

    print_optimal_portfolio_summary(optimum, log_returns, stocks)
    plot_portfolio_scatter(simulated_returns, simulated_volatilities, optimum, log_returns)
