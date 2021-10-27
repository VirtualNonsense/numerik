from numpy.typing import *
import numpy as np
from typing import *

def fdm_solver(
        u: Callable[[float], float],
        u_: Callable[[float], float],
        k: Callable[[float, float]],
        ku__: Callable[[float], float],
        q: Callable[[float], float],
        r: Callable[[float], float],
        f: Callable[[float], float],
        h: float,
        interval: ArrayLike
) -> ArrayLike:
    """
    :param r:
    :param u: Function in question
    :param u_: First derivation of function
    :param k:
    :param ku__:
    :param q:
    :param f: right side
    :param h: Step size
    :type interval: should be filled with the interval boarders
    """
    # location grid
    x = np.arange(start=interval[0], stop=interval[1], step=h)

    # generate A
    N = x.shape[0] - 1
    A = np.Zeros(shape=[N - 1, N - 1])

    # TODO: INDICES FEEL WRONG
    for i in range(N - 1):
        # fill a_i
        A[i][i] = - k((x[i+1] - h)/2) / (h*h) - r(x[i+1]) / (2*h)
        # fill b_i
        if i < N-2:
            A[i][i+1] = k((x[i+1]+h)/2) / (h*h) + k((x[i+1]+h)/2) / (h*h)

        # fill c_i
        if i > 0:
            A[i][i-1] = -k((x[i+1] + h/2)/2) / (h*h) + r(x+1) / 2 * h
