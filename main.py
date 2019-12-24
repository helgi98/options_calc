import options as opts

# Task data
S_end = 150
t_end = 1

K = 40
r = 0
sigma = 0.5

n = 30

# ALl visualization functionality goes here
S_data = [(S_end / n) * (i + 1) for i in range(n)]
t_data = [((t_end / n) * i) for i in range(n)]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from mpl_toolkits.mplot3d import Axes3D

# Need to select Qt5Agg backend before importing plotting functions
matplotlib.use("Qt5Agg")

print(opts.analytic_call_option_price(40, 40, r, sigma, 1, 0))


def build_option_surface(fun, S_mesh, t_mesh):
    res = []
    for i in range(S_mesh.shape[0]):
        res.append([])
        for j in range(S_mesh.shape[1]):
            res[i].append(fun(S_mesh[i][j], K, r, sigma, t_end, t_mesh[i][j]))

    return np.array(res)


def build_lower_option_value_surface(fun, S_mesh):
    res = []
    for i in range(S_mesh.shape[0]):
        res.append([])
        for j in range(S_mesh.shape[1]):
            res[i].append(fun(S_mesh[i][j], K))

    return np.array(res)


def build_upper_curve(S_mesh, t_mesh, Lower_value, Value):
    res_c_x = []
    res_c_y = []
    res_c_z = []
    n = t_mesh.shape[0]

    def find_min(i):
        j = n - 1
        while opts.is_almost_zero(Lower_value[i, j] - Value[i, j]) and j >= 0:
            j -= 1

        return j - 1

    for i in range(n):
        t_i = t_mesh[i, 0]

        j = find_min(i)
        res_c_x.append(S_mesh[i, j])
        res_c_y.append(t_i)
        res_c_z.append(Value[i, j])

    return [res_c_x, res_c_y, res_c_z]


def build_lower_curve(S_mesh, t_mesh, Value):
    res_c_x = []
    res_c_y = []
    res_c_z = []
    n = t_mesh.shape[0]

    def find_min(i):
        j = 0
        while opts.is_almost_zero(Value[i, j]) and j < n:
            j += 1

        return j - 1

    for i in range(n):
        t_i = t_mesh[i, 0]
        j = find_min(i)
        res_c_x.append(S_mesh[i, j])
        res_c_y.append(t_i)
        res_c_z.append(Value[i, j])

    return [res_c_x, res_c_y, res_c_z]


x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

S_mesh, t_mesh = np.meshgrid(S_data, t_data)
Values_calc = build_option_surface(opts.monte_carlo_call_option_price, S_mesh, t_mesh)
Values_lower = build_lower_option_value_surface(opts.lower_call_option_price, S_mesh)

upper_curve = build_upper_curve(S_mesh, t_mesh, Values_lower, Values_calc)
lower_curve = build_lower_curve(S_mesh, t_mesh, Values_calc)

fig = plt.figure()
ax = Axes3D(fig)

# ax.plot(upper_curve[0], upper_curve[1], upper_curve[2], color='red', marker='o')
# ax.plot(lower_curve[0], lower_curve[1], lower_curve[2], color='red', marker='o')

ax.plot_surface(S_mesh, t_mesh, Values_calc, rstride=1, cstride=1,
                color='blue', edgecolor='none', alpha=0.5)

ax.plot_surface(S_mesh, t_mesh, Values_lower, rstride=1, cstride=1,
                color='grey', edgecolor='none', alpha=0.5)

ax.set_xlabel('S')
ax.set_ylabel('t')
ax.set_zlabel('V')

plt.show()
