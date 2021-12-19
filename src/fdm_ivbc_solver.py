from dataclasses import dataclass
from fractions import Fraction
from typing import *

import numpy as np
from numpy.typing import *

from src.util import thomas_algorithm, calc_h


def fdm_ivbc_solver(
        space: Tuple[float, float, int],
        time: Tuple[float, float, int],
        k: Callable[[Union[float, ArrayLike]], float],
        q: Callable[[Union[float, ArrayLike]], float],
        f: Callable[[Union[float, ArrayLike], Union[float, ArrayLike]], float],
        mu_a: Callable[[Union[float, ArrayLike]], float],
        mu_b: Callable[[Union[float, ArrayLike]], float],
        phi: Callable[[Union[float, ArrayLike]], float],
        sigma: float) -> (ArrayLike, ArrayLike, ArrayLike):
    """
    :param space:
        0: a
        1: b
        2: N
    :param time:
        0: t_0
        1: t_end
        2: M
    :param k:
    :param q:
    :param f:
        first argument should be space
        second argument should be time
    :param mu_a: function the describes the boundary condition over time (dirichlet)
    :param mu_b:function the describes the boundary condition over time (dirichlet)
    :param phi:
    :param sigma: weight for integration
        0: Explicit Euler
        1/2: Trapezoid method
        1: implicit Euler
    """
    n = space[-1] + 1
    m = time[-1] + 1
    x = np.linspace(space[0],
                    space[1],
                    n,
                    endpoint=True)
    t = np.linspace(time[0],
                    time[1],
                    m,
                    endpoint=True)
    h_x = x[1] - x[0]
    tau = t[1] - t[0]

    a, b, c = gen_A_vectors(x, k, q, n - 1, h_x)
    # building matrix
    a_h = np.zeros(shape=[b.shape[0], b.shape[0]])
    I_h = np.zeros(shape=a_h.shape)
    for i, b_i in enumerate(b):
        if i > 0:
            a_h[i][i - 1] = a[i - 1]
        if i < b.shape[0] - 1:
            a_h[i][i + 1] = c[i]
        a_h[i][i] = b[i]
        I_h[i][i] = 1
    matrix = np.zeros(shape=[t.shape[0], x.shape[0]])
    matrix[:, 0] = mu_a(t)
    matrix[:, -1] = mu_b(t)
    matrix[0, :] = phi(x)

    A = I_h + sigma * tau * a_h
    tmp = I_h - tau * (1 - sigma) * a_h
    for i, t_j in enumerate(t):
        if i == 0:
            continue
        b = tmp @ matrix[i - 1, :] + tau * (sigma * f(x, t_j) + (1 - sigma) * f(x, t[i - 1]))
        matrix[i, 1:-1] = np.linalg.solve(A, b)[1:-1]
    return [matrix, x, t]


def ArwpFdm1d(
        ort: Tuple[float, float, int],
        zeit: Tuple[float, float, float],
        k: Callable[[float], float],
        q: Callable[[float], float],
        f: Callable[[float, float], float],
        mu_a: Callable[[float], float],
        mu_b: Callable[[float], float],
        phi: Callable[[float], float],
        sigma: float
) -> (ArrayLike, ArrayLike, ArrayLike):
    """

    :param ort:
        0: a
        1: b
        2: N + 1
    :param zeit:
        0: t_0
        1: t_end
        2: M + 1
    :param k:
    :param q:
    :param f:
        first argument should be space
        second argument should be time
    :param mu_a: function the describes the boundary condition over time (dirichlet)
    :param mu_b:function the describes the boundary condition over time (dirichlet)
    :param phi:
    :param sigma: weight for integration
        0: Explicit Euler
        1/2: Trapezoid method
        1: implicit Euler
    """


def gen_A_vectors(x: ArrayLike,
                  k: Callable[[float], float],
                  q: Callable[[float], float],
                  N: int,
                  h: Union[float, Fraction]) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    a = np.zeros(shape=[N - 1])

    b = np.ones(shape=[N - 1])

    c = np.zeros(shape=[N - 1])

    for i in range(N - 1):
        # fill a_i
        a[i] = -k(x[i] - h / 2) / np.square(h)
        # fill c_i
        c[i] = -k(x[i] + h / 2) / np.square(h)
        # fill b_i
        b[i] = -a[i] - c[i] + q(x[i])

    a = np.array([*a, 0])
    b = np.array([1, *b, 1])
    c = np.array([0, *c])

    return a, b, c


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

    m = 80
    n = 7
    t_start = 0
    t_end = 1

    a = 0
    b = 1

    t = (t_start, t_end, m)
    x = (a, b, n)

    sigma = 0
    u = lambda x, t: np.exp(-2 * t) * np.cos(np.pi * x)

    k = lambda x: 1 + 2 * np.power(x, 2)
    q = lambda x: x * (1 - x)

    mu_a = lambda t: np.exp(-2 * t)
    mu_b = lambda t: - np.exp(-2 * t)

    phi = lambda x: np.cos(np.pi * x)

    f = lambda x, t: (np.exp(-2 * t) *
                      (np.cos(np.pi * x) *
                       ((2 * np.power(np.pi, 2) - 1) *
                        np.power(x, 2) + x + np.power(np.pi, 2) - 2)
                       - np.sin(np.pi * x) * (4 * np.pi * x)))

    explicit = fdm_ivbc_solver(space=x,
                               time=t,
                               k=k,
                               q=q,
                               f=f,
                               mu_a=mu_a,
                               mu_b=mu_b,
                               phi=phi,
                               sigma=0)

    crank_nicolson = fdm_ivbc_solver(space=x,
                                     time=t,
                                     k=k,
                                     q=q,
                                     f=f,
                                     mu_a=mu_a,
                                     mu_b=mu_b,
                                     phi=phi,
                                     sigma=1 / 2)

    implicit = fdm_ivbc_solver(space=x,
                               time=t,
                               k=k,
                               q=q,
                               f=f,
                               mu_a=mu_a,
                               mu_b=mu_b,
                               phi=phi,
                               sigma=1)

    X = explicit[1]
    T = explicit[2]
    xx, tt = np.meshgrid(X, T)
    approx = implicit[0]
    solution = u(xx, tt)

    fig: Figure = plt.figure()
    ax: Axes = fig.add_subplot(2, 2, 1, projection='3d')
    ax.set_title("solution")
    ax.set_xlabel("space")
    ax.set_ylabel("time")
    print(np.abs(solution - approx).max())
    print(approx)
    ax.plot_surface(xx, tt, solution, label="solution")
    ax: Axes = fig.add_subplot(2, 2, 2, projection='3d')
    ax.set_title("approx")
    ax.set_xlabel("space")
    ax.set_ylabel("time")
    ax.plot_surface(xx, tt, approx, label="Approx. using explicit euler")
    ax: Axes = fig.add_subplot(2, 2, 3, projection='3d')
    ax.set_title("solution - approx")
    ax.set_xlabel("space")
    ax.set_ylabel("time")
    ax.plot_surface(xx, tt, solution - approx, label="Approx. using explicit euler")

    plt.show()
