from numpy.typing import *
import numpy as np
from typing import *
from fractions import Fraction


def fdm_solver(
        k: Callable[[float], float],
        r: Callable[[float], float],
        q: Callable[[float], float],
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

    # generate A
    N = x.shape[0] - 1
    A = np.zeros(shape=[N - 1, N - 1])

    for i in range(N - 1):
        # fill a_i
        if i > 0:
            A[i][i - 1] = - k((x[i + 1] - h) / 2) / (h * h) - r(x[i + 1]) / (2 * h)

        # fill b_i
        A[i][i] = k((x[i + 1] + h) / 2) / (h * h) + k((x[i + 1] + h) / 2) / (h * h)

        # fill c_i
        if i < N - 2:
            A[i][i + 1] = -k((x[i + 1] + h / 2) / 2) / (h * h) + r(x[i + 1]) / 2 * h + q(x[i + 1])


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
    x = np.arange(start=interval[0], stop=interval[1], step=h)
    r = q
    fdm_solver(k, r, q, h, interval)
