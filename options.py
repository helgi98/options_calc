import math

from scipy.stats import norm


def zero_or_max(x):
    if x < 0:
        return 0
    else:
        return x


def is_almost_zero(x, presize=1):
    return math.fabs(x) < math.pow(10, -presize)


def analytic_put_option_price(S, K, r, sigma, T, t):
    dt = T - t
    if is_almost_zero(dt):
        zero_or_max(K - S)

    dt_sq = math.sqrt(dt)
    d1 = (math.log(S / K) + r * dt) / (sigma * dt_sq) + 0.5 * sigma * dt_sq
    d2 = d1 - (sigma * dt_sq)

    # Vc = S * norm.cdf(d1) - K * math.exp(-r * dt) * norm.cdf(d2)
    Vp = -S * norm.cdf(-d1) + K * math.exp(-r * dt) * norm.cdf(-d2)

    return Vp


def lower_put_option_price(S, K, r=0, sigma=0, T=1, t=0):
    return zero_or_max(K - S)
