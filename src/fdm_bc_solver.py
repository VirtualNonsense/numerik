from dataclasses import dataclass
from fractions import Fraction
from typing import *

import numpy as np
from numpy.typing import *

from src.util import thomas_algorithm


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
    :param r: convection equation
    :param k: diffusion equation
    :param q: reaction equation
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

    N = x.shape[0]
    # generate A as vectors
    # b c 0 0
    # a b c 0
    # 0 a b c
    # 0 0 a b
    a, b, c = gen_A_vectors(x, k, r, q, N, h)

    f_v = f(x)
    # Account for left boundary condition
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

    # Account for right boundary condition
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
    a = np.zeros(shape=[N - 1])

    b = np.ones(shape=[N - 1])

    c = np.zeros(shape=[N - 1])

    for i in range(N - 1):

        # fill b_i
        b[i] = k(x[i] + h / 2) / (h * h) + k(x[i] - h / 2) / (h * h) + q(x[i])
        if i < N - 1:
            # fill a_i
            a[i] = -k(x[i] - h / 2) / (h * h) - r(x[i]) / (2 * h)
            # fill c_i
            c[i] = -k(x[i] + h / 2) / (h * h) + r(x[i]) / (2 * h)

    return a, b, c


def calc_h(interval, n):
    return (interval[1] - interval[0]) / (n - 1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n = 200

    interval_abc = [0, 1]
    h_abc = calc_h(interval_abc, n)


    def u(x: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
        return np.exp(-x) + np.exp(x) + 1


    x_abc = np.arange(start=interval_abc[0], stop=interval_abc[1], step=h_abc)


    def k_ab(x: float):
        return 1


    def f_ab(x: float):
        return np.exp(x) - np.exp(-x) + 1


    def k_cd(x: float):
        return 1 + x


    def f_cd(x: float):
        return x * (np.exp(x) - np.exp(-x)) + x + 1

    solution_abc = u(x_abc)

    # a ###########################################################################################################33333

    boundary_a = (
        DirichletBoundaryCondition(location=interval_abc[0], mu=3),
        DirichletBoundaryCondition(location=interval_abc[1], mu=np.exp(1) + 1 / np.exp(1) + 1),
    )

    u_a = fdm_solver(k=k_ab,
                     r=k_ab,
                     q=k_ab,
                     f=f_ab,
                     h=h_abc,
                     bc=boundary_a,
                     interval=interval_abc)
    # b ###########################################################################################################33333

    boundary_b = (
        DirichletBoundaryCondition(location=interval_abc[0], mu=3),
        RobinBoundaryCondition(location=interval_abc[1], mu=2 * np.exp(1) + 1, kappa=1),
    )
    u_b = fdm_solver(k=k_ab,
                     r=k_ab,
                     q=k_ab,
                     f=f_ab,
                     h=h_abc,
                     bc=boundary_b,
                     interval=interval_abc)
    # c ###########################################################################################################33333
    boundary_c = (
        DirichletBoundaryCondition(location=interval_abc[0], mu=3),
        RobinBoundaryCondition(
            location=interval_abc[1],
            mu=3 * np.exp(1) - 1 / np.exp(1) + 1,
            kappa=1),
    )
    u_c = fdm_solver(k=k_cd,
                     r=k_cd,
                     q=k_cd,
                     f=f_cd,
                     h=h_abc,
                     bc=boundary_c,
                     interval=interval_abc)

    # d ###########################################################################################################33333
    interval_d = [-1 / 2, 1 / 2]
    h_d = calc_h(interval_d, n)
    x_d = np.arange(start=interval_d[0], stop=interval_d[1], step=h_d)

    boundary_d = (
        RobinBoundaryCondition(
            location=interval_d[0],
            mu=(np.exp(1 / 2) - np.exp(-1 / 2)) / 2,
            kappa=0),
        RobinBoundaryCondition(
            location=interval_d[1],
            mu=(5 * np.exp(1 / 2) - np.exp(-1 / 2)) / 2 + 1,
            kappa=1),
    )

    solution_d = u(x_d)
    u_d = fdm_solver(
        k=k_cd,
        r=k_cd,
        q=k_cd,
        f=f_cd,
        h=h_d,
        bc=boundary_d,
        interval=interval_d
    )

    plt.plot(x_abc, u_a, "ro", label="Aprox. A")
    plt.plot(x_abc, u_b, "go", label="Aprox. B")
    plt.plot(x_abc, u_c, "bo", label="Aprox. C")
    plt.plot(x_d, u_d, "mo", label="Aprox. D")
    plt.plot(x_abc, solution_abc, "k-", label="Solution abc")
    plt.plot(x_d, solution_d, "k-", label="Solution d")
    plt.legend()
    plt.show()
