from numpy.typing import *
import numpy as np
from pprint import pprint
from typing import *
from fractions import Fraction
from src.util import thomas_algorithm

# dirichlet boundary-condition
# u(x) = 1
dirichlet = "dirichlet"

# neumann boundary-condition
# u'(x) = 2
neumann = "neumann"

# robin boundary-condition
# u(x) + 2u'(x) = 132
robin = "robin"


def fdm_solver(
        k: Callable[[float], float],
        r: Callable[[float], float],
        q: Callable[[float], float],
        f: Callable[[Union[float, ArrayLike]], Union[float, ArrayLike]],
        h: Union[float, Fraction],
        border: Dict[float, Tuple[float, str]],
        interval: ArrayLike
) -> ArrayLike:
    """
    :param r:
    :param k:
    :param q:
    :param f: right side
    :param h: Step size
    :param border:
        key: boundary value
        value: a tuple of the value at the border an the kind of boundary condition
    :type interval: should be filled with the interval boarders
    """

    x = np.arange(start=interval[0],
                  stop=interval[1] + h,  # + h to include interval border
                  step=h)

    # generate A
    N = x.shape[0]
    a, b, c = gen_A_vectors(x, k, r, q, N, h)

    f_v = f(x)
    # f_v[0] = border[x[0]][0]
    f_v = np.array([border[x[0]][0], *f_v])
    f_v[-1] = border[x[-1]][0]
    # according for boundary conditions
    if border[x[0]][1] == dirichlet:
        a = np.array([*a, 0])
        b = np.array([1, *b])

    if border[x[-1]][1] == dirichlet:
        b = np.array([*b, 1])
        c = np.array([0, *c])

    # solving l
    return thomas_algorithm(a, b, c, f_v)[1:]


def gen_A_vectors(x: ArrayLike,
                  k: Callable[[float], float],
                  r: Callable[[float], float],
                  q: Callable[[float], float],
                  N: int,
                  h: Union[float, Fraction]) -> ArrayLike:

    # TODO: check if that fits other bc than dirichlet
    a = np.zeros(shape=[N - 1])

    b = np.ones(shape=[N - 1])

    # TODO: check if that fits other bc than dirichlet
    c = np.zeros(shape=[N - 1])

    for i in range(N - 1):

        # fill b_i
        b[i] = k(x[i] + h / 2) / (h * h) + k(x[i] - h / 2) / (h * h) + q(x[i])
        if i < N - 1:
            # fill a_i
            a[i] = -k(x[i + 1] - h / 2) / (h * h) - r(x[i + 1]) / (2 * h)
            # fill c_i
            c[i] = -k(x[i] + h / 2) / (h * h) + r(x[i]) / (2 * h)

    return a, b, c


def calc_h(interval, n):
    return (interval[1] - interval[0]) / (n - 1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n = 4


    def u(x: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
        return np.exp(-x) + np.exp(x) + 1


    def k(x: float):
        return 1


    def q(x: float):
        return 0


    def f(x: float):
        return np.exp(x) - np.exp(-x) + 1


    interval = [0, 1]
    h = calc_h(interval, n)
    x = np.arange(start=interval[0], stop=interval[1] + h, step=h)
    boundary = {
        interval[0]: (3, dirichlet),
        interval[1]: (np.exp(1) + 1 / np.exp(1) + 1, dirichlet)
    }

    solution = u(x)
    u = fdm_solver(k=k,
                   r=k,
                   q=k,
                   f=f,
                   h=h,
                   border=boundary,
                   interval=interval)

    plt.plot(x, u, label="Aprox. solution", color="r")
    plt.plot(x, solution, label="Solution", color="b")
    plt.show()
