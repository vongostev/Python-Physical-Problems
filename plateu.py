# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 20:34:11 2021

@author: vonGostev
"""

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

t_max = 1.5
t = np.linspace(0, t_max, 100)
v0 = 10
g = 9.8
alpha = np.deg2rad(45)

x0 = -1
y0 = 0
h = 0.6


def calc_x(v0, alpha, t, x0):
    return x0 + v0 * np.cos(alpha) * t


def calc_y(v0, alpha, t):
    return v0 * np.sin(alpha) * t - g * t ** 2 / 2


@np.vectorize
def plateau(h, x):
    return int(x > 0) * h


def opt_distance(params, v0, h, t_max):
    alpha, x0 = params

    def root_fun(t):
        return calc_y(v0, alpha, t) - h

    t_h1 = opt.root(root_fun, 0, method='hybr').x[0]
    t_h2 = opt.root(root_fun, t_max, method='hybr').x[0]

    x1 = calc_x(v0, alpha, t_h1, x0)
    x2 = calc_x(v0, alpha, t_h2, x0)

    if x1 > 0:
        return x1

    return -x2


res = opt.minimize(opt_distance, [alpha, x0], args=(v0, h, t_max), method='Nelder-Mead', tol=1e-12,
                   options=dict(disp=1))


X = calc_x(v0, alpha, t, x0)

plt.plot(X, calc_y(v0, alpha, t))
plt.plot(X, plateau(h, calc_x(v0, alpha, t, x0)))

alpha, x0 = res.x
X = calc_x(v0, alpha, t, x0)
plt.plot(X, calc_y(v0, alpha, t))

plt.ylim((0, None))
plt.show()

print(res)

eta = v0 ** 2 / 2 / g / h
alpha_t = 1 / 2 * np.arccos(- 1 / eta)
x0_t = h * (1 - eta + np.sqrt(eta ** 2 - 1))

print(alpha, -x0)
print(alpha_t, x0_t)

alphas = np.linspace(0, np.pi / 2, 100).reshape((-1, 1))
x0s = np.linspace(-4, 0, 100)


@np.vectorize
def show_x2(a, x):
    return opt_distance([a, x], v0, h, t_max)


data = show_x2(alphas, x0s)
data[data > 0] = 0
plt.imshow(data)
plt.show()
