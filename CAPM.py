import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

RISK_FREE_RATE = 0.05
# Explore CAPM model and calculate beta value from formula and linear regression model
class CAPM:
    def __init__(self, stock, start_date, end_date):
        self.data = None
        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self):
        data = {}
        for stock in self.stock:
            ticker = yf.download(stock, start=self.start_date, end=self.end_date, auto_adjust=False)
            data[stock] = ticker["Adj Close"].squeeze()

        return pd.DataFrame(data)

    def initialize(self):
        stock_data = self.download_data().dropna()
        self.data = stock_data.resample("ME").last()
        self.data = pd.DataFrame({'stock_adj_close': stock_data[self.stock[0]],
                                'market_adj_close': stock_data[self.stock[1]]})

        # calculate log returns
        self.data[['stock_return','market_return']] = np.log(self.data[['stock_adj_close','market_adj_close']]/self.data[['stock_adj_close','market_adj_close']].shift(1))
        self.data = self.data[1:]
        # print(self.data)

    # Using formula to calculate beta = cova(stock,market) divided by var(market)
    def calculate_beta_by_formula(self):
        cov_matrix = np.cov((self.data['stock_return'],self.data['market_return']))

        beta = cov_matrix[0,1] / cov_matrix[1,1]
        print('Beta from the formula =', beta)
        return beta
    # Using linear regression to calculate slope = beta
    def calculate_beta_by_regression(self):
        beta, alpha = np.polyfit(self.data['market_return'], self.data['stock_return'], deg=1)
        print('Beta from the regression =', beta)
        # Annual expected returns
        expected_returns = RISK_FREE_RATE + beta * (self.data['market_return'].mean() * 12 - RISK_FREE_RATE)
        print('Expected return =', expected_returns)
        self.plot_regression(alpha, beta)

    def plot_regression(self, alpha, beta):
        fig, axis = plt.subplots(1, figsize = (10,5))
        axis.scatter(self.data['market_return'], self.data['stock_return'], label='Data Points')
        axis.plot(self.data['market_return'], beta * self.data['market_return'] + alpha,
                  c='red', label='CAPM line')
        plt.title('CAPM Model Linear Regression')
        plt.xlabel('Market Return $R_m$', fontsize = 18)
        plt.ylabel('Stock Return $R_a$')
        axis.text(
            0.05, 0.95,
            f'$\\beta = {beta:.2f}$\n$\\alpha = {alpha:.4f}$',
            transform=axis.transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8)
        )
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    capm = CAPM(['AAPL', '^GSPC'], '2020-01-01', '2025-01-01')
    capm.initialize()
    capm.calculate_beta_by_formula()
    capm.calculate_beta_by_regression()