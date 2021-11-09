from numpy.typing import *
import numpy as np
from pprint import pprint
from typing import *
from fractions import Fraction
from src.util import thomas_algorithm
from dataclasses import dataclass


@dataclass
class BoundaryCondition:
    location: float


@dataclass
class DirichletBoundaryCondition(BoundaryCondition):
    mu: float


@dataclass
class RobinBoundaryCondition(BoundaryCondition):
    kappa: float
    mu: float


def fdm_solver(
        k: Callable[[float], float],
        r: Callable[[float], float],
        q: Callable[[float], float],
        f: Callable[[Union[float, ArrayLike]], Union[float, ArrayLike]],
        h: Union[float, Fraction],
        bc: Tuple[BoundaryCondition, BoundaryCondition],
        interval: ArrayLike
) -> ArrayLike:
    """
    :param r:
    :param k:
    :param q:
    :param f: right side
    :param h: Step size
    :param bc: boundary conditions
        key: boundary value
        value: a tuple of the value at the border an the kind of boundary condition
    :type interval: should be filled with the interval boarders
    """

    x = np.arange(start=interval[0],
                  stop=interval[1],
                  step=h)

    # generate A
    N = x.shape[0]
    a, b, c = gen_A_vectors(x, k, r, q, N, h)

    f_v = f(x)
    if isinstance(bc[0], DirichletBoundaryCondition):
        b = np.array([1, *b])
        c = np.array([0, *c])
        f_v = np.array([bc[0].mu, *f_v])

    if isinstance(bc[0], RobinBoundaryCondition):
        b_0 = 2 * k(interval[0] + h / 2) / (h * h) \
              + q(interval[0]) \
              + bc[0].kappa * (2 / h + r(interval[0]) / k(interval[0] + h / 2))

        c_0 = -2 * k(interval[0] + h / 2) / (h * h)
        f_0 = f(interval[0]) \
              + bc[0].mu * (2 / h + r(interval[0]) / k(interval[0] + h / 2))

        b = np.array([b_0, *b])
        c = np.array([c_0, *c])
        f_v = np.array([f_0, *f_v])

    if isinstance(bc[-1], DirichletBoundaryCondition):
        a = np.array([*a, 0])
        b = np.array([*b, 1])
        f_v[-1] = bc[-1].mu

    if isinstance(bc[-1], RobinBoundaryCondition):
        a_n = - 2 * k(interval[-1] - h / 2) / (h * h)
        b_n = 2 * k(interval[-1] - h / 2) / (h * h) + q(interval[-1]) \
              + bc[-1].kappa * (2 / h - r(interval[-1]) / k(interval[-1] - h / 2))
        f_n = f(interval[-1]) + bc[-1].mu * (2 / h - r(interval[-1]) / k(interval[-1] - h / 2))
        f_v[-1] = f_n
        a = np.array([*a, a_n])
        b = np.array([*b, b_n])

    # solving l
    return thomas_algorithm(a, b, c, f_v)[:-1]


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

    n = 16


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
    x = np.arange(start=interval[0], stop=interval[1], step=h)
    boundary = (
        DirichletBoundaryCondition(location=interval[0], mu=3),
        RobinBoundaryCondition(location=interval[1], mu=2 * np.exp(1) + 1, kappa=1),
    )

    solution = u(x)
    u = fdm_solver(
        k=k,
        r=k,
        q=k,
        f=f,
        h=h,
        bc=boundary,
        interval=interval
    )

    plt.plot(x, u, label="Aprox. solution", color="r")
    plt.plot(x, solution, label="Solution", color="b")
    plt.show()
