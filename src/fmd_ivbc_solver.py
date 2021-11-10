from dataclasses import dataclass
from fractions import Fraction
from typing import *

import numpy as np
from numpy.typing import *

from src.util import thomas_algorithm, calc_h


def fdm_ivbc_solver(
        space: Tuple[float, float, int],
        time: Tuple[float, float, float],
        k: Callable[[float], float],
        q: Callable[[float], float],
        f: Callable[[float, float], float],
        mu_a: Callable[[float], float],
        mu_b: Callable[[float], float],
        phi: Callable[[float], float],
        sigma: float
) -> (ArrayLike, ArrayLike, ArrayLike):
    """

    :param space:
        0: a
        1: b
        2: N + 1
    :param time:
        0: t_0
        1: t_end
        2: M + 1
    :param k:
    :param q:
    :param f:
    :param mu_a:
    :param mu_b:
    :param phi:
    :param sigma: weight for integration
        0: Explicit Euler
        1/2: Trapezoid method
        1: Euler
    """
    n = space[-1] - 1
    h = calc_h(space[:-1], n)

    x = np.arange(start=space[0],
                  stop=space[1],
                  step=h)

    a, b, c = gen_A_vectors(x, k, q, n, h)


    pass


def ArwpFdm1d(ort, zeit, k, q, f, mu_a, mu_b, phi, sigma) -> (ArrayLike, ArrayLike, ArrayLike):
    pass


def gen_A_vectors(x: ArrayLike,
                  k: Callable[[float], float],
                  q: Callable[[float], float],
                  N: int,
                  h: Union[float, Fraction]) -> ArrayLike:
    a = np.zeros(shape=[N - 1])

    b = np.ones(shape=[N - 1])

    c = np.zeros(shape=[N - 1])

    for i in range(N - 1):
        # fill b_i
        b[i] = k(x[i] + h / 2) / (h * h) + k(x[i] - h / 2) / (h * h) + q(x[i])
        # fill a_i
        a[i] = -k(x[i] - h / 2) / (h * h)
        # fill c_i
        c[i] = -k(x[i] + h / 2) / (h * h)

    return a, b, c


if __name__ == '__main__':
    pass
