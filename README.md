# Quant Finance Projects

A collection of Python projects implementing core topics in quantitative finance, including Monte Carlo Simulation, Black-Scholes option pricing, Capital Asset Pricing Model (CAPM), Markowitz portfolio optimization, Value at Risk, and algorithmic trading backtesting. The scripts use historical data and Python libraries, such as `yfinance`, `numpy`, `pandas`, `matplotlib`, and `scipy`.

## List of Projects
- [CAPM](#capm) — [`CAPM.py`](./CAPM.py)
- [Markowitz Portfolio Optimization](#markowitz-portfolio-optimization) — [`Markowitz_Portfolio_Theory.py`](./Markowitz_Portfolio_Theory.py)
- [Black-Scholes Option Pricing](#black-scholes-option-pricing) — [`monte_carlo_black_scholes_option_pricing.py`](./monte_carlo_black_scholes_option_pricing.py)
- [Value at Risk](#value-at-risk) — [`Value_at_risk_comparison.py`](./Value_at_risk_comparison.py)
- [Momentum_And_Simple_Moving_Average_Trading_Backtest](#momentum-simple-moving-average-trading-backtest) — [`momentum_moving_avg.py`](./momentum_moving_avg.py)
- [SMA Algorithmic Trading Backtest](#sma-algorithmic-trading-backtest) — [`aapl_algo_backtest.py`](./aapl_algo_backtest.py)

## Projects Overview

### CAPM
This script implements the Capital Asset Pricing Model (CAPM) using Apple (`AAPL`) as the stock and the S&P 500 index (`^GSPC`) as the market benchmark over the 2020–2025 period. It downloads adjusted close data, computes log returns, estimates beta value in two ways, and plots the regression line for visualization.

**What it does**
- Implements the Capital Asset Pricing Model (CAPM) for AAPL using the S&P 500 (^GSPC) as an example
- Downloads historical adjusted closing prices from `yfinance`
- Calculates log returns for both the stock and the market return
- Computes beta, which measures the stock’s systematic risk relative to the overall market in two ways:
  1. using the covariance/variance formula
  2. using linear regression
- Uses the regression beta to calculate the stock’s expected return with the CAPM equation
- Plots the relationship between market returns and stock returns with a CAPM regression line
- Shows the estimated alpha and beta on the graph for interpretation (stock_return = α + β⋅market_return)
  
Inputs:
```
capm = CAPM(['AAPL', '^GSPC'], '2020-01-01', '2025-01-01')
```

Outputs:

<img width="373" height="69" alt="Screenshot 2026-03-25 at 1 54 27 PM" src="https://github.com/user-attachments/assets/e93b9925-aa3a-4284-99b2-a78179a8686d" />
<img width="956" height="485" alt="Screenshot 2026-03-25 at 1 54 08 PM" src="https://github.com/user-attachments/assets/44c9368d-c180-4fff-bf43-322416d0e31b" />

---

### Markowitz Portfolio Optimization
This script implements a Markowitz portfolio optimization workflow for five stocks: `AAPL`, `AMZN`, `MSFT`, `COST`, and `CTAS`, using historical data from yfinance from 2015 to 2025. It utilizes Monte Carlo portfolio simulation to find the maximum-Sharpe portfolio. 

**What it does**
- Downloads adjusted historical stock prices for AAPL, AMZN, MSFT, COST, and CTAS from `yfinance`
- Computes daily log returns and annualized return/covariance metrics
- Simulates 10,000 random portfolios using Monte Carlo simulation
- Calculates portfolio return, volatility, and Sharpe ratio
- Uses SciPy SLSQP optimization to find the maximum-Sharpe portfolio
- Prints optimal allocation and performance metrics
- Visualizes the simulated portfolio set and highlights the optimal portfolio
  
Inputs:
```
TOTAL_TRADING_DAYS = 252
TOTAL_PORTFOLIOS = 10000
RISK_FREE_RATE = 0.02

stocks = ['AAPL', 'AMZN', 'MSFT', 'COST', 'CTAS']

start_date = '2015-01-01'
end_date = '2025-01-01'
```

Outputs:

<img width="174" height="156" alt="Screenshot 2026-03-25 at 1 44 49 PM" src="https://github.com/user-attachments/assets/1308efe4-d3ed-42a1-9998-e3d305f734d0" />
<img width="321" height="158" alt="Screenshot 2026-03-25 at 1 45 24 PM" src="https://github.com/user-attachments/assets/9de9ed37-8d95-4c79-9755-e52babad8c7a" />
<img width="396" height="276" alt="Screenshot 2026-03-25 at 1 45 35 PM" src="https://github.com/user-attachments/assets/4fbabb1c-6a5c-4f4b-a61f-d910ead952b0" />
<img width="764" height="380" alt="Screenshot 2026-03-25 at 1 37 04 PM" src="https://github.com/user-attachments/assets/658b3051-840d-4bc7-95a5-bd04d23d8270" />

---
### Black-Scholes Option Pricing
This script computes both call and put option prices using Monte Carlo simulation and compares them with the risk-neutral Black-Scholes formula. 
- Built a Python option pricing model that compares Monte Carlo simulation with the Black-Scholes formula
- Used `NumPy` and `SciPy` to handle random simulation and normal distribution calculations
- Simulated future stock prices using geometric Brownian motion
- Calculated present-day call and put option prices from discounted expected payoffs
- Implemented Black-Scholes pricing using `d1` and `d2` parameter calculations
- Compared simulation-based and analytical prices to validate the model implementation

Inputs:
- `S0 = 100` → stock price is $100
- `E = 100` → strike price is $100
- `T = 1` → 1 year to expiration
- `rf = 0.05` → risk-free rate 5%
- `sigma = 0.2` → 20% volatility
- `iterations = 100000` → run 100,000 simulations

Outputs:

<img width="1058" height="154" alt="Screenshot 2026-03-25 at 2 33 54 PM" src="https://github.com/user-attachments/assets/243ebb13-b33a-4ed4-a23a-d2dd9bd04616" />

---
### Value at Risk
This script compares Value at Risk (VaR) calculated in two ways: Monte Carlo simulation and the lognormal VaR formula. It uses historical Apple prices from 2020 to 2021, estimates log-return parameters, and then computes the portfolio VaR. 

**What it does**
- Downloads Apple price data from `yfinance`
- Computes daily log returns from the historical series
- Estimates the mean and volatility of log returns
- Calculate Monte Carlo VaR using simple normally distributed returns from the lower-tail percentile of simulated portfolio values using `scipy` and `numpy`
- Computes a formula-based VaR using the log-return quantile `scipy` and `numpy`

Inputs
```
    portfolio_value = 1e6
    confidence_level = 0.95
    days = 100
    iterations = 100000

    start = dt.datetime(2020, 1, 1)
    end = dt.datetime(2021, 1, 1)
```
Outputs:

<img width="949" height="88" alt="Screenshot 2026-03-25 at 2 39 00 PM" src="https://github.com/user-attachments/assets/c74e172d-f756-4aa3-b0a8-9e7a404b8428" />

---

### Momentum_SMA_Trading_Strategy
This script backtests a monthly momentum strategy on top-percent S&P 500 stocks and compares it to `SPY` buy-and-hold strategy using total returns, annualized returns, and sharpe ratios. The backtest tests historical data of multiple combinations of 3, 6, and 12 month momentum as well as top 10% and top 20% of ranked stocks from 2022 to 2026. 

**What it does***
- Pulls the current S&P 500 stock list from Wikipedia
- Downloads historical adjusted closing prices using `yfinance
- Updates portfolio monthly
- Chooses stocks trading above their simple moving average
- Ranks eligible stocks by momentum scores
- Buys the top percentage of those stocks equally
- Holds the portfolio until the next rebalance date
- Compares this strategy performance to SPY buy and hold strategy

Inputs:
```
    stocks = get_sp500_tickers() # current S&P 500 stock list from Wikipedia

    start = dt.datetime(2022, 1, 1) # year 2020 to 2026
    end = dt.datetime(2026, 1, 1)

tests = [
        {"name": "Top 20%, 12-month momentum", "momentum_lookback": 252, "top_percent": 0.20},
        {"name": "Top 10%, 12-month momentum", "momentum_lookback": 252, "top_percent": 0.10},
        {"name": "Top 20%, 6-month momentum", "momentum_lookback": 126, "top_percent": 0.20},
        {"name": "Top 10%, 6-month momentum", "momentum_lookback": 126, "top_percent": 0.10},
        {"name": "Top 20%, 3-month momentum", "momentum_lookback": 63, "top_percent": 0.20},
        {"name": "Top 10%, 3-month momentum", "momentum_lookback": 63, "top_percent": 0.10},
    ]
```

Outputs:
<img width="419" height="563" alt="Screenshot 2026-04-14 at 5 05 39 PM" src="https://github.com/user-attachments/assets/c7698d22-96f1-432e-bb04-e7c1484946ed" />
<img width="435" height="591" alt="Screenshot 2026-04-14 at 5 06 12 PM" src="https://github.com/user-attachments/assets/02932703-801a-4aa6-b38d-87075acc9d28" />
<img width="409" height="571" alt="Screenshot 2026-04-14 at 5 06 41 PM" src="https://github.com/user-attachments/assets/5b1f5349-608f-41a0-86b5-c86db4a6d4b2" />


---

### SMA Algorithmic Trading Backtest
This script backtests a 20-day vs. 50-day simple moving average crossover strategy on `AAPL` over the past year by downloading daily price data with yfinance, creating long-or-cash trading signals, and simulating portfolio performance. It uses `pandas` for time-series handling and moving-average calculations, then compares the strategy’s return to a buy-and-hold value.

**What it does**
- Pulls `AAPL` historical daily close prices from `yfinance`
- Generates trading signals based on 20-day short and 50-day long-term moving average crossovers
- Simulates a portfolio that fully allocates to stock or cash
- Tracks portfolio positions through time
- Compares strategy performance against buy-and-hold over the past year

Inputs:
```
    ticker: str = 'AAPL'
    initial_cash: float = 10_000.0
    short_window: int = 20
    long_window: int = 50
    years: int = 1
```
Outputs:

<img width="914" height="113" alt="Screenshot 2026-03-25 at 3 09 12 PM" src="https://github.com/user-attachments/assets/1fb30d15-5c98-42b0-b9a0-a4160f7f9b65" />
