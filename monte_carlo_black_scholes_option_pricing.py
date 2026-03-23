import pandas as pd
import numpy as np
from math import log, sqrt, exp
from scipy import stats

# Option pricing with Monte Carlo simulation and Black-Scholes
class CompareOptionPricing():
    def __init__(self, S0, E, T, rf, sigma, iterations):
        self.S0 = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations

    # Monte Carlo Simulations to calculate call option pricing
    def call_option_monte_carlo_simulate(self):
        option_data = np.zeros([self.iterations, 2])

        rand = np.random.normal(0, 1, self.iterations)

        stock_price = self.S0 * np.exp(
            self.T * (self.rf - 0.5 * self.sigma ** 2)
            + self.sigma * np.sqrt(self.T) * rand
        )

        option_data[:, 1] = stock_price - self.E
        # calculate call option payoff - max(S-E,0)
        payoffs = np.amax(option_data, axis=1)
        average_payoff = np.mean(payoffs)
        # need to apply exp(-rT) factor to calculate present day value
        present_value = average_payoff * np.exp(-1 * self.T * self.rf)

        return present_value

    # Monte Carlo Simulations to calculate put option pricing
    def put_option_monte_carlo_simulate(self):
        option_data = np.zeros([self.iterations, 2])

        rand = np.random.normal(0, 1, self.iterations)

        stock_price = self.S0 * np.exp(
            self.T * (self.rf - 0.5 * self.sigma ** 2)
            + self.sigma * np.sqrt(self.T) * rand
        )
        # calculate put option payoff - max(E-S,0)
        option_data[:, 1] = self.E - stock_price
        payoffs = np.amax(option_data, axis=1)
        average_payoff = np.mean(payoffs)
        present_value = average_payoff * np.exp(-1 * self.T * self.rf)

        return present_value
    # Calculate d1 parameter
    def calculate_d1(self):
        d1 = (log(self.S0 / self.E) +
              (self.rf + self.sigma * self.sigma / 2.0) * self.T) / (self.sigma * sqrt(self.T))

        return d1

    # Calculate d2 parameter
    def calculate_d2(self):
        d2 = self.calculate_d1() - self.sigma * sqrt(self.T)

        return d2

    # Black-Scholes formula to calculate call option pricing
    def call_option_black_scholes_price(self):
        d1 = self.calculate_d1()
        d2 = self.calculate_d2()

        return self.S0 * stats.norm.cdf(d1) - self.E * exp(-self.rf * self.T) * stats.norm.cdf(d2)

    # Black-Scholes formula to calculate put option pricing
    def put_option_black_scholes_price(self):
        d1 = self.calculate_d1()
        d2 = self.calculate_d2()

        return -self.S0 * stats.norm.cdf(-d1) + self.E * exp(-self.rf * self.T) * stats.norm.cdf(-d2)


if __name__ == '__main__':
    model = CompareOptionPricing(100, 100, 1, 0.05, 0.2, 100000)

    mc_call = model.call_option_monte_carlo_simulate()
    mc_put = model.put_option_monte_carlo_simulate()
    d1 = model.calculate_d1()
    d2 = model.calculate_d2()
    print('The d1 parameter for call and put options is: %s' % (d1))
    print('The d2 parameter for call and put options is: %s' % (d2))
    bs_call = model.call_option_black_scholes_price()
    bs_put = model.put_option_black_scholes_price()
    print('The call option price with Monte Carlo is:', mc_call)
    print('The call option price according to Black-Scholes model:', bs_call)

    print('The put option price with Monte Carlo is:', mc_put)
    print('The put option price according to Black-Scholes model:', bs_put)