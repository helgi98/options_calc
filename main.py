import options as opts

# Task data
S_end = 20
t_end = 1

K = 10
r = 0.06
sigma = 0.3

n = 40

# ALl visualization functionality goes here
S_data = [(S_end / n) * (i + 1) for i in range(n)]
t_data = [((t_end / n) * i) for i in range(n)]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from mpl_toolkits.mplot3d import Axes3D

matplotlib.use("Qt5Agg")


def build_option_surface(fun, S_mesh, t_mesh):
    res = []
    for i in range(S_mesh.shape[0]):
        res.append([])
        for j in range(S_mesh.shape[1]):
            res[i].append(fun(S_mesh[i][j], K, r, sigma, t_end, t_mesh[i][j]))

    return np.array(res)


def build_lower_option_value_surface(fun, S_mesh, t_mesh):
    res = []
    for i in range(S_mesh.shape[0]):
        res.append([])
        for j in range(S_mesh.shape[1]):
            res[i].append(fun(S_mesh[i][j], K, r, sigma, t_end, t_mesh[i, j]))

    return np.array(res)


def build_upper_curve(S, t, Lower_value, Value):
    res = [[], [], []]

    for i in range(len(t)):
        # We search for the lowest Sj where Vij is close to LVij
        j = len(S) - 2
        while j > 0 and opts.is_almost_zero(Lower_value[i, j] - Value[i, j]):
            j -= 1
        j += 1

        # We shouldn't go lower than that
        if S[j] < K:
            res[0].append(K)
            res[1].append(t[i])
            res[2].append(0)
        else:
            res[0].append(S[j])
            res[1].append(t[i])
            res[2].append(Value[i, j])

    return res


def build_lower_curve(S, t, Value):
    res = [[], [], []]

    for i in range(len(t)):
        # We search for the biggest Si where Vij = 0
        j = 1
        while j < len(S) and opts.is_almost_zero(Value[i, j]):
            j += 1
        j -= 1

        res[0].append(S[j])
        res[1].append(t[i])
        res[2].append(Value[i, j])

    return res


x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

S_mesh, t_mesh = np.meshgrid(S_data, t_data)

Values_calc = build_option_surface(opts.analytic_put_option_price, S_mesh, t_mesh)
Values_lower = build_lower_option_value_surface(opts.lower_put_option_price, S_mesh, t_mesh)

upper_curve = build_upper_curve(S_data, t_data, Values_lower, Values_calc)
lower_curve = build_lower_curve(S_data, t_data, Values_calc)

fig = plt.figure()
ax = Axes3D(fig)

ax.plot(upper_curve[0], upper_curve[1], upper_curve[2], color='red', marker='o')
ax.plot(lower_curve[0], lower_curve[1], lower_curve[2], color='red', marker='o')

ax.plot_surface(S_mesh, t_mesh, Values_calc, rstride=1, cstride=1,
                color='blue', edgecolor='none', alpha=0.5)

ax.plot_surface(S_mesh, t_mesh, Values_lower, rstride=1, cstride=1,
                color='grey', edgecolor='none', alpha=0.5)

ax.set_xlabel('S')
ax.set_ylabel('t')
ax.set_zlabel('V')

plt.show()
