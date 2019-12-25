import math

import matplotlib

import options

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

S = 100
T = 1

K = 100
r = 0
sigma = 0.3

n = 50

dT = T / n
R = math.exp(r * dT)
Rinv = 1 / R
u = math.exp(sigma * math.sqrt(dT))
d = 1 / u
p_up = (R - d) / (u - d)
p_down = 1 - p_up

uu = u * u

prices = np.zeros(n + 1)
prices[0] = S * math.pow(d, n)

for i in range(1, n + 1):
    prices[i] = uu * prices[i - 1]

call_values = np.zeros((n + 1, n + 1))
for i in range(0, n + 1):
    call_values[n][i] = max(0.0, (prices[i] - K))

for step in range(n - 1, -1, -1):
    for i in range(0, step + 1):
        call_values[step][i] = (p_up * call_values[step + 1][i + 1] + p_down * call_values[step + 1][i]) * Rinv

print(call_values[0][0])

t = []
S = []
V = []

for i in range(0, n + 1):
    for j in range(0, i + 1):
        t += [i * dT]
        S += [prices[j]]
        V += [call_values[i][j]]

Va = []
for i in range(0, n + 1):
    for j in range(0, i + 1):
        Va += [options.analytic_call_option_price(S[len(Va)], K, r, sigma, t[len(Va)], 0)]

Vl = []
for i in range(0, n + 1):
    for j in range(0, i + 1):
        Vl += [options.lower_call_option_price(prices[j], K)]

Vd = np.array(Va) - np.array(Vl)

fig = plt.figure()
ax = Axes3D(fig)

# ax.scatter(t, S, V, s=1, color='red')
# ax.scatter(t, S, Va, s=1, color='blue')
# ax.scatter(t, S, Vl, s=1, color='black')
ax.scatter(t, S, Vd, s=1, c='black')

ax.set_xlabel('t')
ax.set_ylabel('S')
ax.set_zlabel('dV')

plt.show()
