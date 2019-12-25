import math

import numpy as np
from scipy.stats import norm


def zero_or_max(x):
    if x < 0:
        return 0
    else:
        return x


def is_almost_zero(x, presize=1, proxmity=None):
    if proxmity:
        return math.fabs(x) < math.fabs(proxmity)

    return math.fabs(x) < math.pow(10, -presize)


def analytic_call_option_price(S, K, r, sigma, T, t):
    dt = T - t
    if is_almost_zero(dt, 8):
        return lower_call_option_price(S, K)

    dt_sq = math.sqrt(dt)
    d1 = (math.log(S / K) + r * dt) / (sigma * dt_sq) + 0.5 * sigma * dt_sq
    d2 = d1 - (sigma * dt_sq)

    return S * norm.cdf(d1) - K * math.exp(-r * dt) * norm.cdf(d2)


def analytic_put_option_price(S, K, r, sigma, T, t):
    dt = T - t
    if is_almost_zero(dt, 8):
        return lower_put_option_price(S, K)

    dt_sq = math.sqrt(dt)
    d1 = (math.log(S / K) + r * dt) / (sigma * dt_sq) + 0.5 * sigma * dt_sq
    d2 = d1 - (sigma * dt_sq)

    return -S * norm.cdf(-d1) + K * math.exp(-r * dt) * norm.cdf(-d2)


def monte_carlo_call_option_price_gen(sims=5000):
    def monte_carlo_call_option_price(S, K, r, sigma, T, t):
        dt = T - t
        if is_almost_zero(dt, 8):
            return lower_call_option_price(S, K)

        dt_sq = math.sqrt(dt)

        R = (r - 0.5 * math.pow(sigma, 2)) * dt
        SD = sigma * dt_sq

        sum_payoffs = 0
        for i in range(sims):
            S_T = S * math.exp(R + SD * np.random.normal())
            sum_payoffs += lower_call_option_price(S_T, K)

        return math.exp(-r * dt) * (sum_payoffs / sims)

    return monte_carlo_call_option_price


def binomial_call_option_price_gen(steps=100):
    def binomial_call_option_price(S, K, r, sigma, T, t):
        dt = T - t
        if is_almost_zero(dt, 8):
            return lower_call_option_price(S, K)

        R = math.exp(r * (dt / steps))
        Rinv = 1 / R
        u = math.exp(sigma * math.sqrt(dt / steps))
        d = 1 / u
        p_up = (R - d) / (u - d)
        p_down = 1 - p_up

        prices = [0.0] * (steps + 1)
        prices[0] = S * pow(d, steps)
        uu = u * u
        for i in range(1, steps + 1):
            prices[i] = uu * prices[i - 1]

        call_values = [0.0] * (steps + 1)
        for i in range(steps + 1):
            call_values[i] = lower_call_option_price(prices[i], K)

        for step in range(steps - 1, -1, -1):
            for i in range(step + 1):
                call_values[i] = (p_up * call_values[i + 1] + p_down * call_values[i]) * Rinv
                prices[i] = d * prices[i + 1]
                call_values[i] = max(call_values[i], prices[i] - K)

        return call_values[0]

    return binomial_call_option_price


def lower_put_option_price(S, K, r=0, sigma=0, T=1, t=0):
    return zero_or_max(K - S)


def lower_call_option_price(S, K, r=0, sigma=0, T=1, t=0):
    return zero_or_max(S - K)
