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
        first argument should be space
        second argument should be time
    :param mu_a: function the describes the boundary condition over time (dirichlet)
    :param mu_b:function the describes the boundary condition over time (dirichlet)
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

    return [0, 0, 0]


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
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

    m_p_1 = 20
    n_p_1 = 20
    t_start = 0
    t_end = 1

    a = 0
    b = 1

    t = (t_start, t_end, m_p_1)
    x = (a, b, n_p_1)

    sigma = 0
    u = lambda x, t: np.sin(x) * np.cos(t)

    k = lambda x: 1
    q = lambda x: k(x)
    mu_a = lambda t: 0
    mu_b = lambda t: np.sin(1) * np.cos(t)

    phi = lambda x: np.sin(x)

    f = lambda x, t: np.sin(x) * (2 * np.cos(t) - np.sin(t))

    # explicit = fdm_ivbc_solver(space=x,
    #                            time=t,
    #                            k=k,
    #                            q=q,
    #                            f=f,
    #                            mu_a=mu_a,
    #                            mu_b=mu_b,
    #                            phi=phi,
    #                            sigma=0)

    # crank_nicolson = fdm_ivbc_solver(space=x,
    #                                  time=t,
    #                                  k=k,
    #                                  q=q,
    #                                  f=f,
    #                                  mu_a=mu_a,
    #                                  mu_b=mu_b,
    #                                  phi=phi,
    #                                  sigma=1 / 2)
    #
    # implicit = fdm_ivbc_solver(space=x,
    #                            time=t,
    #                            k=k,
    #                            q=q,
    #                            f=f,
    #                            mu_a=mu_a,
    #                            mu_b=mu_b,
    #                            phi=phi,
    #                            sigma=1)

    # plotting
    h_x = calc_h([a, b], n_p_1 - 2)
    h_t = calc_h([t_start, t_end], m_p_1 - 2)

    X = np.arange(a, b, h_x)
    T = np.arange(t_start, t_end, h_t)
    xx, tt = np.meshgrid(X, T)
    solution = u(xx, tt)

    fig: Figure = plt.figure()
    ax: Axes = plt.axes(projection='3d')
    ax.set_xlabel("space")
    ax.set_ylabel("time")
    ax.plot_surface(xx, tt, solution, label="solution")
    # ax.plot_surface(explicit[1], explicit[2], explicit[0], label="Approx. using explicit euler")
    plt.show()
