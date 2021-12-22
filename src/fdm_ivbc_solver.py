from fractions import Fraction
from typing import *

import numpy as np
from numpy.typing import *


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
        0: a. the "left" boundary in space
        1: b. the "right" boundary in space
        2: N + 1. basically the output dimensions
    :param time:
        0: t_0. simulation start in time.
        1: t_end. simulation end in time
        2: M + 1. basically the output dimensions
    :param k: diffusion equation
    :param q: reaction equation
    :param f: right side
        first argument should be space
        second argument should be time
    :param mu_a: function the describes the situation in a over time (dirichlet)
    :param mu_b: function the describes the situation in b over time (dirichlet)
    :param phi: function that describes the initial state at t = 0
    :param sigma: weight for integration
        0: Explicit Euler
        1/2: Trapezoid method
        1: implicit Euler
    """
    # setting up variables
    n = space[-1]
    m = time[-1]
    x = np.linspace(space[0],
                    space[1],
                    n,
                    endpoint=True)
    t = np.linspace(time[0],
                    time[1],
                    m,
                    endpoint=True)
    # space delta
    h_x = x[1] - x[0]

    # time delta
    tau = t[1] - t[0]
    # grabbing matrix vectors
    a, b, c = gen_A_vectors(x, k, q, n - 1, h_x)

    # building A_h and 1 matrix from vectors to utilize fast matrix routines
    a_h = np.zeros(shape=[b.shape[0], b.shape[0]])
    I_h = np.zeros(shape=a_h.shape)
    for i, b_i in enumerate(b):
        if i > 0:
            a_h[i][i - 1] = a[i - 1]
        if i < b.shape[0] - 1:
            a_h[i][i + 1] = c[i]
        a_h[i][i] = b[i]
        I_h[i][i] = 1

    # initializing empty matrix
    matrix = np.zeros(shape=[t.shape[0], x.shape[0]])

    # left boundary conditions
    matrix[:, 0] = mu_a(t)

    # right boundary condition
    matrix[:, -1] = mu_b(t)

    # initial condition for t = 0
    matrix[0, 1:-1] = phi(x)[1:-1]

    # calculating left side of LGS
    A = I_h + sigma * tau * a_h

    # preparing constant part of right side
    tmp = I_h - tau * (1 - sigma) * a_h

    # solving for every t
    for i, t_j in enumerate(t):
        # skipping t_0.
        # This could be done by slicing t, but it would lead to more confusing indizes.
        if i == 0:
            continue
        # calculating the right side
        b = tmp @ matrix[i - 1, :] + tau * (sigma * f(x, t_j) + (1 - sigma) * f(x, t[i - 1]))

        # placing solution within the free spaces of the matrix
        matrix[i, 1:-1] = np.linalg.solve(A, b)[1:-1]
    return [matrix, x, t]


def ArwpFdm1d(
        ort: Tuple[float, float, int],
        zeit: Tuple[float, float, int],
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
        0: a. the "left" boundary in space
        1: b. the "right" boundary in space
        2: N + 1. basically the output dimensions
    :param zeit:
        0: t_0. simulation start in time.
        1: t_end. simulation end in time
        2: M + 1. basically the output dimensions
    :param k: diffusion equation
    :param q: reaction equation
    :param f: right side
        first argument should be space
        second argument should be time
    :param mu_a: function the describes the situation in a over time (dirichlet)
    :param mu_b: function the describes the situation in b over time (dirichlet)
    :param phi: function that describes the initial state at t = 0
    :param sigma: weight for integration
        0: Explicit Euler
        1/2: Trapezoid method
        1: implicit Euler
    """
    return fdm_ivbc_solver(space=ort,
                           time=zeit,
                           k=k,
                           q=q,
                           f=f,
                           mu_a=mu_a,
                           mu_b=mu_b,
                           phi=phi,
                           sigma=sigma)


def gen_A_vectors(x: ArrayLike,
                  k: Callable[[float], float],
                  q: Callable[[float], float],
                  N: int,
                  h: Union[float, Fraction]) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    # init empty vectors
    a = np.zeros(shape=[N - 1])

    b = np.ones(shape=[N - 1])

    c = np.zeros(shape=[N - 1])

    # Iterating over space
    for i in range(N - 1):
        # fill a_i
        a[i] = -k(x[i] - h / 2) / np.square(h)
        # fill c_i
        c[i] = -k(x[i] + h / 2) / np.square(h)
        # fill b_i
        b[i] = -a[i] - c[i] + q(x[i])

    # expanding vectors to account for dirichlet bc
    a = np.array([*a, 0])
    b = np.array([1, *b, 1])
    c = np.array([0, *c])

    return a, b, c


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

    ####################################################################################################################
    # settings
    ####################################################################################################################
    # m + 1
    m_p1 = 801

    # n + 1
    n_p1 = 21

    t_start = 0
    t_end = 1

    a = 0
    b = 1

    t = (t_start, t_end, m_p1)
    x = (a, b, n_p1)

    draw_solutions = False
    draw_difference = True

    ####################################################################################################################
    # problem
    ####################################################################################################################
    u = lambda x, t: np.sin(x) * np.cos(t)
    u_dt = lambda x, t: -np.sin(x) * np.sin(t)
    u_dx = lambda x, t: np.cos(x) * np.cos(t)
    u_dx2 = lambda x, t: -np.sin(x) * np.cos(t)

    k = lambda x: np.power(x, 3)
    k_dx = lambda x: 3 * np.power(x, 2)
    q = lambda x: 1

    mu_a = lambda t: u(a, t)
    mu_b = lambda t: u(b, t)

    phi = lambda x: u(x, 0)

    f = lambda x, t: u_dt(x, t) - (k_dx(x) * u_dx(x, t) + k(x) * u_dx2(x, t)) + q(x) * u(x, t)

    ####################################################################################################################
    # solve
    ####################################################################################################################
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
    solutions = {"explicit": explicit, "crank nicolson": crank_nicolson, "implicit": implicit}
    ####################################################################################################################
    # plot
    ####################################################################################################################
    for i, (key, item) in enumerate(solutions.items()):
        X = item[1]
        T = item[2]
        xx, tt = np.meshgrid(X, T)
        approx = implicit[0]
        solution = u(xx, tt)
        if draw_solutions:
            fig: Figure = plt.figure()
            fig.suptitle(key)
            fig.canvas.set_window_title(f"{key} - solutions")
            ax: Axes = fig.add_subplot(1, 2, 1, projection='3d')
            ax.set_title("solution")
            ax.set_xlabel("space")
            ax.set_ylabel("time")
            print(np.abs(solution - approx).max())
            print(approx)
            ax.plot_surface(xx, tt, solution, label="solution")
            ax: Axes = fig.add_subplot(1, 2, 2, projection='3d')
            ax.set_title("approx")
            ax.set_xlabel("space")
            ax.set_ylabel("time")
            ax.plot_surface(xx, tt, approx, label="Approx. using explicit euler")
        if draw_difference:
            fig: Figure = plt.figure()
            fig.suptitle(key)
            fig.canvas.set_window_title(f"{key} - solution - approx")
            ax: Axes = fig.add_subplot(1, 1, 1, projection='3d')
            ax.set_title("solution - approx")
            ax.set_xlabel("space")
            ax.set_ylabel("time")
            ax.plot_surface(xx, tt, solution - approx, label="Approx. using explicit euler")

    plt.show()
