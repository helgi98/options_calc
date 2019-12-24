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
    if is_almost_zero(dt):
        zero_or_max(K - S)

    dt_sq = math.sqrt(dt)
    d1 = (math.log(S / K) + r * dt) / (sigma * dt_sq) + 0.5 * sigma * dt_sq
    d2 = d1 - (sigma * dt_sq)

    return S * norm.cdf(d1) - K * math.exp(-r * dt) * norm.cdf(d2)


def analytic_put_option_price(S, K, r, sigma, T, t):
    dt = T - t
    if is_almost_zero(dt):
        zero_or_max(K - S)

    dt_sq = math.sqrt(dt)
    d1 = (math.log(S / K) + r * dt) / (sigma * dt_sq) + 0.5 * sigma * dt_sq
    d2 = d1 - (sigma * dt_sq)

    return -S * norm.cdf(-d1) + K * math.exp(-r * dt) * norm.cdf(-d2)


def monte_carlo_call_option_price(S, K, r, sigma, T, t, sims=1000):
    dt = T - t
    if is_almost_zero(dt):
        zero_or_max(K - S)

    dt_sq = math.sqrt(dt)

    R = (r - 0.5 * math.pow(sigma, 2)) * dt
    SD = sigma * dt_sq

    sum_payoffs = 0
    for i in range(sims):
        S_T = S * math.exp(R + SD * np.random.normal())
        sum_payoffs += zero_or_max(S_T - K)

    return math.exp(-r * dt) * (sum_payoffs / sims)


def lower_put_option_price(S, K, r=0, sigma=0, T=1, t=0):
    return zero_or_max(K - S)


def lower_call_option_price(S, K, r=0, sigma=0, T=1, t=0):
    return zero_or_max(S - K)
