import numpy as np
from typing import *
from numpy.typing import *
from fem_solver import rwp_fem_1d
from scipy.integrate import quad


def _adaptivity_solver(
        x_grid: ArrayLike,
        k: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        r: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        q: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        f: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        rba: Tuple[int, float, float],
        rbb: Tuple[int, float, float],
        el_typ: int,
        in_typ: int,
        epsilon: float,
        alpha: float,
        m_e_max: int = 200):
    m_e = x_grid.size - 1
    u_approx = 0
    x_e = 0
    while m_e <= m_e_max:
        u_approx, x_e = rwp_fem_1d(x_grid=x_grid,
                                   k=k,
                                   r=r,
                                   q=q,
                                   f=f,
                                   rba=rba,
                                   rbb=rbb,
                                   el_typ=el_typ,
                                   in_typ=in_typ)
        eta = 0
        etas = np.zeros(m_e)

        # calculate errors
        for i in range(m_e):
            t_l = x_grid[i]
            t_r = x_grid[i + 1]
            h = t_r - t_l
            fun = lambda x: np.square(2 * f(x))
            etas[i] = h * np.sqrt(quad(fun, t_l, t_r)[0])
            eta += np.square(etas[i])
        eta = np.sqrt(eta)
        eta_max = np.max(etas)
        # return current solution if it's accurate enough
        if eta <= epsilon:
            return u_approx, x_e

        # improve grid
        tmp_grid = [*x_grid]
        for i in range(m_e):
            if etas[i] > alpha * eta_max:
                t_l = x_grid[i]
                t_r = x_grid[i + 1]
                h = t_r - t_l
                new_value = h / 2 + t_l
                tmp_grid = [*tmp_grid, new_value]
        tmp_grid.sort()
        x_grid = np.array(tmp_grid)
        m_e = x_grid.size - 1
    return u_approx, x_e


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from scipy.special import erf

    ####################################################################################################################
    # shared by a and b
    x_grid = np.linspace(0, 1, num=10, endpoint=True)
    k = lambda x: 1
    r = lambda x: 0
    q = lambda x: 0
    rba = (1, 0, 0)
    rbb = (1, 0, 0)
    el_typ = 1
    in_typ = 0
    epsilon = 1e-4
    alpha = .8
    ####################################################################################################################
    # a
    c_a = 100
    f = lambda x: (2 * np.power(c_a, 3) * (x - .5)) / np.square(1 + np.square(c_a) * np.square(x - .5))
    u_a = lambda x: np.arctan(c_a * (x - .5)) + np.arctan(c_a / 2) * (1 - 2 * x)

    u_adap_approx_a, x_adap_a = _adaptivity_solver(x_grid, k, r, q, f, rba, rbb, el_typ, in_typ, epsilon, alpha)
    u_approx_a, x_a = rwp_fem_1d(x_grid, k, r, q, f, rba, rbb, el_typ, in_typ)

    ####################################################################################################################
    # b
    c_b = 1e3
    u_b = lambda x: -1 / (4 * c_b) * (np.sqrt(np.pi * c_b) * ((2 * x - 1) * erf(np.sqrt(c_b) * (x - .5))
                                                              - erf(np.sqrt(c_b) / 2))
                                      + 2 * np.exp(-c_b * np.square(x - .5))
                                      - 2 * np.exp(-c_b / 4))
    f = lambda x: np.exp(-c_b * np.square(x - .5))

    u_adap_approx_b, x_adap_b = _adaptivity_solver(x_grid, k, r, q, f, rba, rbb, el_typ, in_typ, epsilon, alpha)
    u_approx_b, x_b = rwp_fem_1d(x_grid, k, r, q, f, rba, rbb, el_typ, in_typ)

    fig: Figure = plt.figure("2.3 III")
    fig.suptitle("Adaptivity solver test")

    ax: Axes = fig.add_subplot(2, 1, 1)
    ax.plot(x_a, u_approx_a, marker="o", linestyle="", label=f"start grid ({u_approx_a.size} Points)")
    ax.plot(x_adap_a, u_adap_approx_a, marker="x", linestyle="",
            label=f"adaptivity solver ({u_adap_approx_a.size} Points)")
    ax.plot(x_adap_a, u_a(x_adap_a), label="solution")
    ax.legend()

    ax: Axes = fig.add_subplot(2, 1, 2)
    ax.plot(x_b, u_approx_b, marker="o", linestyle="", label=f"start grid ({u_approx_b.size} Points)")
    ax.plot(x_adap_b, u_adap_approx_b, marker="x", linestyle="",
            label=f"adaptivity solver ({u_adap_approx_b.size} Points)")
    ax.plot(x_adap_b, u_b(x_adap_b), label="solution")
    ax.legend()
    plt.show()
