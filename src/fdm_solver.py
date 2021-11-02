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
    A = gen_A_matrix(x, k, r, q, N, h)
    print(A)


def gen_A_matrix(x: ArrayLike,
                 k: Callable[[float], float],
                 r: Callable[[float], float],
                 q: Callable[[float], float],
                 N: int,
                 h: Union[float, Fraction]) -> ArrayLike:
    A = np.zeros(shape=[N - 1, N - 1])

    for i in range(N - 1):
        # fill a_i
        if i > 0:
            A[i][i - 1] = -k(x[i + 1] - h / 2) / (h * h) - r(x[i + 1]) / (2 * h)

        # fill b_i
        A[i][i] = k(x[i + 1] + h / 2) / (h * h) + k(x[i + 1] - h / 2) / (h * h) + q(x[i + 1])

        # fill c_i
        if i < N - 2:
            A[i][i + 1] = -k(x[i + 1] + h / 2) / (h * h) + r(x[i + 1]) / (2 * h)
    return A


if __name__ == '__main__':
    pass
