import numpy as np
import yfinance as yf
import datetime as dt
from scipy.stats import norm

# Download historical closing prices for a stock
def download_data(stock, start, end):
    tickers = yf.download(
        stock,
        start=start,
        end=end,
        auto_adjust=True,
        multi_level_index=False
    )
    return tickers[['Close']].rename(columns={'Close': stock})

# Compare VaR from Monte Carlo Simulations and Formula
class VaRComparison:
    def __init__(self, portfolio_value, mu, sigma, confidence_level, days, iterations):
        # Current dollar value of the portfolio
        self.portfolio_value = portfolio_value

        # Average daily return from historical data
        self.mu = mu

        # Daily standard deviation of returns from historical data
        self.sigma = sigma

        # Confidence level for VaR, for example 0.95 means 95%
        self.confidence_level = confidence_level

        # Number of days ahead for the VaR calculation
        self.days = days

        # Number of Monte Carlo simulations
        self.iterations = iterations

    # Calculate Monte Carlo VaR using simple normally distributed returns
    def monte_carlo_var(self):

        rand = np.random.normal(0, 1, self.iterations)

        simulated_values = self.portfolio_value * np.exp(
            self.days * (self.mu - 0.5 * self.sigma ** 2)
            + self.sigma * np.sqrt(self.days) * rand
        )

        # Find the lowest percentile of possible portfolio values
        percentile = np.percentile(
            simulated_values,
            (1 - self.confidence_level) * 100
        )

        # VaR = today's portfolio value - the lower-tail portfolio value
        monte_carlo_var = self.portfolio_value - percentile
        return monte_carlo_var

    # Calculate Formula-based VaR using the normal distribution
    def formula_var(self):
        # use the left-tail z-score directly
        z = norm.ppf(1 - self.confidence_level)

        # compute the log-return quantile
        log_return = (
            self.days * (self.mu - 0.5 * self.sigma ** 2)
            + self.sigma * np.sqrt(self.days) * z
        )

        # convert that quantile into a portfolio-value VaR
        formula_var = self.portfolio_value * (1 - np.exp(log_return))
        return formula_var


if __name__ == '__main__':

    portfolio_value = 1e6
    confidence_level = 0.95
    days = 100
    iterations = 100000

    start = dt.datetime(2020, 1, 1)
    end = dt.datetime(2021, 1, 1)

    # Download Apple historical prices
    apple = download_data('AAPL', start, end)

    apple['returns'] = np.log(apple['AAPL'] / apple['AAPL'].shift(1))
    apple = apple.dropna()

    # Estimate average daily return and daily volatility from history
    mu = np.mean(apple['returns'])
    sigma = np.std(apple['returns'])

    model = VaRComparison(portfolio_value, mu, sigma, confidence_level, days, iterations)

    print('Monte Carlo VaR: $%.2f' % model.monte_carlo_var())
    print('Formula-based VaR: $%.2f' % model.formula_var())