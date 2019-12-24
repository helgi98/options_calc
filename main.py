import options as opts

S_end = 100
t_end = 1

K = 40
r = 0
sigma = 0.4

n = 100

S_data = [(S_end / n) * (i + 1) for i in range(n)]
t_data = [((t_end / n) * i) for i in range(n)]

print(opts.analytic_put_option_price(25, 50, r, sigma, 1, 0))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from mpl_toolkits.mplot3d import Axes3D

# Need to select Qt5Agg backend before importing plotting functions
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
            res[i].append(fun(S_mesh[i][j], K))

    return np.array(res)


def build_upper_curve(S_mesh, t_mesh, Lower_value, Value):
    res_c_x = []
    res_c_y = []
    res_c_z = []
    n = t_mesh.shape[0]

    def find_minimizing_S(i):
        was_zero = False
        for j in range(0, n):
            if opts.is_almost_zero(Lower_value[i, j] - Value[i, j], 2):
                was_zero = True
            elif was_zero:
                return j - 1

        return 0

    for i in range(n):
        t_i = t_mesh[i, 0]
        j = find_minimizing_S(i)
        res_c_x.append(S_mesh[i, j])
        res_c_y.append(t_i)
        res_c_z.append(Value[i, j])

    return [res_c_x, res_c_y, res_c_z]


def build_lower_curve(S_mesh, t_mesh, Value_mesh):
    res_c_x = []
    res_c_y = []
    res_c_z = []
    n = t_mesh.shape[0]

    def find_minimizing_S(i):
        was_zero = False
        for j in range(n - 1, -1, -1):
            if opts.is_almost_zero(Value_mesh[i, j]):
                was_zero = True
            elif was_zero:
                return j - 1

        return 0

    for i in range(n):
        t_i = t_mesh[i, 0]
        j = find_minimizing_S(i)
        res_c_x.append(S_mesh[i, j])
        res_c_y.append(t_i)
        res_c_z.append(Value_mesh[i, j])

    return [res_c_x, res_c_y, res_c_z]


x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

S_mesh, t_mesh = np.meshgrid(S_data, t_data)
Values_calc = build_option_surface(opts.analytic_put_option_price, S_mesh, t_mesh)
Values_lower = build_lower_option_value_surface(opts.lower_put_option_price, S_mesh, t_mesh)

upper_curve = build_upper_curve(S_mesh, t_mesh, Values_lower, Values_calc)
lower_curve = build_lower_curve(S_mesh, t_mesh, Values_calc)

fig = plt.figure()
ax = Axes3D(fig)

ax.plot(upper_curve[0], upper_curve[1], upper_curve[2], marker='o', color='red')
ax.plot(lower_curve[0], lower_curve[1], lower_curve[2], marker='o', color='red')

ax.plot_surface(S_mesh, t_mesh, Values_calc, rstride=1, cstride=1,
                color='blue', edgecolor='none', alpha=0.5)

ax.plot_surface(S_mesh, t_mesh, Values_lower, rstride=1, cstride=1,
                color='grey', edgecolor='none', alpha=0.5)

ax.set_xlabel('S')
ax.set_ylabel('t')
ax.set_zlabel('V')

plt.show()
