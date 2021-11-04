from numpy.typing import *
import numpy as np
from pprint import pprint
from typing import *
from fractions import Fraction


def fdm_solver(
        k: Callable[[float], float],
        r: Callable[[float], float],
        q: Callable[[float], float],
        f: Callable[[Union[float, ArrayLike]], float],
        h: Union[float, Fraction],
        interval: ArrayLike
) -> ArrayLike:
    """
    :param r:
    :param k:
    :param q:
    :param f: right side
    :param h: Step size
    :type interval: should be filled with the interval boarders
    """
    # location grid

    x = np.arange(start=interval[0],
                  stop=interval[1] + h,  # + h to include interval border
                  step=h)
    f_v = f(x)
    # generate A
    N = x.shape[0] - 1
    a, b, c = gen_A_matrix(x, k, r, q, N, h)
    pprint(a)
    pprint(b)
    pprint(c)


def gen_A_matrix(x: ArrayLike,
                 k: Callable[[float], float],
                 r: Callable[[float], float],
                 q: Callable[[float], float],
                 N: int,
                 h: Union[float, Fraction]) -> ArrayLike:
    a = np.zeros(shape=[N - 1])
    b = np.ones(shape=[N])
    c = np.zeros(shape=[N - 1])

    for i in range(N):
        # fill a_i
        if i > 1:
            a[i - 1] = -k(x[i + 1] - h / 2) / (h * h) - r(x[i + 1]) / (2 * h)

        # fill b_i
        if 0 < i < N-1:
            b[i] = k(x[i + 1] + h / 2) / (h * h) + k(x[i + 1] - h / 2) / (h * h) + q(x[i + 1])
        # fill c_i
        if i < N - 2:
            c[i] = -k(x[i + 1] + h / 2) / (h * h) + r(x[i + 1]) / (2 * h)

    return a, b, c


if __name__ == '__main__':
    n = 16


    def k(x: float):
        return 1


    def q(x: float):
        return 0


    def f(x: float):
        return np.exp(x) - np.exp(-x) + 1


    interval = [0, 1]
    h = (interval[1] - interval[0]) / n
    x = np.arange(start=interval[0],
                  stop=interval[1] + h,  # + h to include interval border
                  step=h)

    # generate A
    r = q
    fdm_solver(k, r, q, f, h, interval)
