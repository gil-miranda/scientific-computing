import numpy as np

## Foward euler using numpy
def forward_euler(F, t0, y0, tf, n):
    h = (tf - t0) / n
    t = np.linspace(t0, tf, n + 1)
    y = [y0]
    for t_i in t:
        y.append(y, y[-1] + h * F(t_i, y[-1]))
    return t, np.array(y)

## Runge-Kutta 4th order
def runge_kutta4(F, t0, y0, tf, n, h = False):
    if not h:
        h = (tf - t0) / n
    t = np.linspace(t0, tf, n + 1)
    y = [y0]
    for t_i in t:
        k1 = h * F(t_i, y[-1])
        k2 = h * F(t_i + h / 2, y[-1] + k1 / 2)
        k3 = h * F(t_i + h / 2, y[-1] + k2 / 2)
        k4 = h * F(t_i + h, y[-1] + k3)
        y.append(y[-1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6)
    return t, np.array(y)

def newton_raphson(F, t0, y0, tf, n):
    h = (tf - t0) / n
    t = np.linspace(t0, tf, n + 1)
    y = [y0]
    for t_i in t:
        y.append(y[-1] - (y[-1] - F(t_i, y[-1])) / (F(t_i, y[-1]) - F(t_i, y[-1] - h)))
    return t, np.array(y)