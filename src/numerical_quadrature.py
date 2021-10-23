import numpy as np
from typing import *


def quad_mittelpunkt(f: Callable[[float], float], a: float, b: float, n: int):
    y = np.zeros(n)
    h = (b - a) / n
    start = a
    for i in range(n):
        y[i] = f(start + h / 2) * h
        start += h
    return y.sum()


def quad_trapez(f: Callable[[float], float], a: float, b: float, n: int):
    weights = np.array([1 / 2, 1 / 2])
    return quad_newton_cotes(f, a, b, n, weights)


def quad_simpson(f: Callable[[float], float], a: float, b: float, n: int):
    weights = np.array([1, 4, 1]) / 6
    return quad_newton_cotes(f, a, b, n, weights)


def quad_milne(f: Callable[[float], float], a: float, b: float, n: int):
    weights = np.array([7, 32, 12, 32, 7]) / 90
    return quad_newton_cotes(f, a, b, n, weights)


def quad_weddle(f: Callable[[float], float], a: float, b: float, n: int):
    weights = np.array([41, 216, 27, 272, 27, 216, 41]) / 840
    return quad_newton_cotes(f, a, b, n, weights)


def quad_newton_cotes(f: Callable[[float], float], a: float, b: float, n: int, weights: np.array):
    w_unique = weights.shape[0] - 1
    y = np.zeros(w_unique * n + 1)
    x_step = (b - a) / (y.shape[0] - 1)
    for i in range(y.shape[0]):
        y[i] = f(a + x_step * i)

    h = (b - a) / n
    sq = np.zeros(n)

    for i in range(n):
        s = 0
        for ii, w in enumerate(weights):
            s += w * y[i * w_unique + ii]
        sq[i] = h * s

    return sq.sum()


if __name__ == '__main__':
    def f(x: float) -> float:
        return x * x


    a = 0
    b = 2
    n = 1

    real_value = 1 / 3 * b * b * b - 1 / 3 * a * a * a

    mittel = quad_mittelpunkt(f, a, b, n)
    trap = quad_trapez(f, a, b, n)
    simps = quad_simpson(f, a, b, n)
    milne = quad_milne(f, a, b, n)
    weddle = quad_weddle(f, a, b, n)

    print(f"real value: {real_value}\n"
          f"mittel: {mittel}, {real_value - mittel}\n"
          f"trap: {trap}, {real_value - trap}\n"
          f"simps: {simps}, {real_value - simps}\n"
          f"milne: {milne}, {real_value - milne}\n"
          f"weddle: {weddle}, {real_value - weddle}\n"
          )
