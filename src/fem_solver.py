import numpy as np
from dataclasses import dataclass

from typing import *
from scipy.integrate import quad
from numerical_quadrature import quad_gauss
from numpy.typing import *
from numpy.linalg import solve

from boundary_condition import *


@dataclass
class FEMProblem:
    label: str
    interval: Tuple[float, float]
    u: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]]
    u_dx: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]]
    u_dx2: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]]

    k: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]]
    k_dx: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]]
    r: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]]
    q: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]]

    k_udx_dx: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]]

    boundary: Tuple[BoundaryCondition, BoundaryCondition]

    def f(self, x: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
        return -self.k_udx_dx(x) + self.r(x) * self.u_dx(x) + self.q(x) * self.u(x)


def lin_elem(k, q, f, rbr, rbl, in_typ, n_e) -> Tuple[ArrayLike, ArrayLike]:
    """

    :param k:
    :param q:
    :param f:
    :param rbr:
    :param rbl:
    :param in_typ:
    :param n_e:
    :return:
    """

    def phi(x_i, index):
        if index == 0:
            return 1 - x_i
        return x_i

    k_i = np.zeros(shape=[n_e, n_e])
    f_i = np.zeros(n_e)
    F = lambda x_i: (rbr - rbl) * x_i + rbl
    phi_2 = [-1, 1]
    hi = abs(rbr - rbl)
    integrate = lambda function: quad_gauss(function, -1, 1, n=in_typ)
    if in_typ == 0:
        integrate = lambda function: quad(function, 0, 1)
    for alpha in range(n_e):
        fun_2 = lambda x_i: f(F(x_i)) * phi(x_i, alpha)
        f_i[alpha] = hi * integrate(fun_2)
        for beta in range(n_e):
            fun = lambda x_i: k(F(x_i)) / np.square(hi) * phi_2[alpha] * phi_2[beta] + q(F(x_i)) * phi(x_i, alpha) * phi(x_i, beta)
            k_i[alpha, beta] = hi * integrate(fun)
    return k_i, f_i


def quad_elem(k, q, f, rbr, rbl, in_typ, n_e) -> Tuple[ArrayLike, ArrayLike]:
    """

    :param k:
    :param q:
    :param f:
    :param rbr:
    :param rbl:
    :param in_typ:
    :param n_e:
    :return:
    """

    def phi(x_i: float, index: int) -> float:
        if index == 0:
            return (2 * x_i - 1) * (x_i - 1)
        if index == 1:
            return 4 * x_i * (1 - x_i)
        return x_i * (2 * x_i - 1)

    def phi_2(x_i: float, index: int) -> float:
        if index == 0:
            return 4 * x_i - 3
        if index == 1:
            return 4 - 8 * x_i
        return 4 * x_i - 1

    k_i = np.zeros(shape=[n_e, n_e])
    f_i = np.zeros(n_e)
    F = lambda x_i: (rbr - rbl) * x_i + rbl
    h_i = np.abs(rbr - rbl)
    integrate = lambda function: quad_gauss(function, -1, 1, n=in_typ)
    if in_typ == 0:
        integrate = lambda function: quad(function, 0, 1)
    for alpha in range(n_e):
        fun_2 = lambda x_i: f(F(x_i)) * phi(x_i, alpha)
        f_i[alpha] = h_i * integrate(fun_2)
        for beta in range(n_e):
            fun = lambda x_i: k(F(x_i)) / np.square(h_i) * phi_2(x_i, alpha) * phi_2(x_i, beta) + q(F(x_i)) * phi(x_i,
                                                                                                           alpha) * phi(
                x_i, beta)
            k_i[alpha, beta] = h_i * integrate(fun)
    return k_i, f_i


def rwp_fem_1d(
        x_git: ArrayLike,
        k: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        r: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        q: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        f: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        rba: Tuple[int, float, float],
        rbb: Tuple[int, float, float],
        el_typ: int,
        in_typ: int) -> Tuple[ArrayLike, ArrayLike]:
    """
    1 dimensional fem solver vor boundary condition problems

    :param x_git: Grid points [x0, ..., xN]
    :param k: diffusion equation
    :param r: convection equation
    :param q: reaction equation
    :param f: right side
    :param rba: [type, beta, value]
        type == 1: Dirchlet-RB
        type == 2: Neumann-RB
        type == 3: Robin-RB
    :param rbb: [type, beta, value]
        type == 1: Dirchlet-RB
        type == 2: Neumann-RB
        type == 3: Robin-RB
    :param el_typ:
        1: linear approach
        2: cubic approach
    :param in_typ:
        integration typ:
            0: use default integration from numpy
            1 - 3 use quad_gauss
    :return:
        0: u_kno: approx. solution
        1: x_kno: node values
            el_typ == 1: u_kno will be equal to x_grid
            el_typ == 2: u_know will contain more points than x_grid
    """
    # preparing variables

    """amount of finite elements (amount of spaces between grid nodes)"""
    m_e = len(x_git) - 1

    """Dimension of K_h. Amount of p nodes"""
    n_g = el_typ * m_e + 1
    x_kno = np.zeros(n_g)
    u_kno = np.zeros(n_g)

    """
    Matrix that will be used to solve for u (K_h * u_h = f_h). shape: n_g, n_g
    """
    k_h: np.ndarray = np.zeros(shape=[n_g, n_g])

    """
    Vector that will be used to solve for u(K_h * u_h = f_h)
    """
    f_h = np.zeros(n_g)

    """
    this table contains the global indexes of 
    columns:
    0            1            2            3            4
    el_typ       in_typ       P^e = 1      P^e = 2      P^e = 3 (will remain empty if el_typ = 1)
    """
    kn_el = np.zeros(shape=[m_e, 5])
    kn_el[:, 0] = el_typ
    kn_el[:, 1] = in_typ

    # setting up node table and nodes
    if el_typ == 1:
        # returning x_git in this case
        x_kno = x_git
        for i in range(m_e):
            kn_el[i, 2] = i
            kn_el[i, 3] = i + 1
    elif el_typ == 2:
        kn_el[:, 2] = np.array([i for i in range(1, n_g - 2, 2)])
        kn_el[:, 3] = np.array([i for i in range(2, n_g - 1, 2)])
        kn_el[:, 4] = np.array([i for i in range(3, n_g, 2)])
        j = 0
        for i in range(m_e):
            x_kno[j] = x_git[i]
            x_kno[j + 1] = (x_git[i + 1] - x_git[i]) / 2 + x_git[i]
            x_kno[j + 2] = x_git[i + 1]
            j += 2

    # calculating k_h and f_h
    n_e = el_typ + 1

    # configuration for main loop
    elem = lambda rbr, lbr: quad_elem(k, q, f, rbr, lbr, kn_el[0, 1], n_e)
    get_rb = lambda i: (x_kno[kn_el[i, 4]], x_kno[kn_el[i, 2]])
    if el_typ == 1:
        get_rb = lambda i: (x_kno[kn_el[i, 3]], x_kno[kn_el[i, 2]])
        elem = lambda rbr, lbr: lin_elem(k, q, f, rbr, lbr, kn_el[0, 1], n_e)

    for i in range(n_e):
        rbr, lbr = get_rb(i)
        k_i, f_i = elem(rbr, lbr)
        for a in range(n_e):
            r = kn_el[i, 2 + a]
            f_h[r] += f_i[a]

            for b in range(n_e):
                s = kn_el[i, 2 + b]
                k_h[r, s] += k_i[a, b]

    # handle robin bc
    if rba[0] == 3:
        k_h[0, 0] += rba[1]
        f_h[0] += rba[2]
    if rbb[0] == 3:
        k_h[n_g, n_g] += rbb[1]
        f_h[n_g] += rbb[2]

    # handle neumann bc
    if rba[0] == 2:
        f_h[0] += rba[2]
    if rbb[0] == 2:
        f_h[n_g] += rbb[2]

    # handle dirichlet bc
    if rba[0] == 1:
        u_kno[0] = rba[3]
        k_h2 = k_h
        k_h2[0, :] = 0
        f_h2 = f_h
        f_h2[0] = 0
        f_h = f_h2 - k_h2@u_kno

        f_h[0] = rba[2]
        k_h[0, 0] = 1
        k_h[1:n_g, 0] = 0
        k_h[0, 1:n_g] = 0

    if rbb[0] == 1:
        u_kno[n_g] = rbb[3]
        tmp = u_kno[0]
        u_kno[0] = 0
        k_h2 = k_h
        k_h2[n_g, :] = 0
        f_h2 = f_h
        f_h2[n_g] = 0
        f_h = f_h2 - k_h2@u_kno

        f_h[n_g] = rba[2]
        k_h[n_g, n_g] = 1
        k_h[1:n_g, n_g] = 0
        k_h[n_g, 1:n_g] = 0
        u_kno[0] = tmp
    u_kno = solve(k_h, f_h)
    return u_kno, x_kno


if __name__ == '__main__':
    from pprint import pprint

    # ##################################################################################################################
    # settings
    # ##################################################################################################################

    # ##################################################################################################################
    # problems
    # ##################################################################################################################
    # 2.1
    problems = []
    # 2.1.a
    label = "2.1.a"
    interval = (0, 1)
    u = lambda x: np.exp(-x) + np.exp(x) + 1
    u_dx = lambda x: -np.exp(-x) + np.exp(x)
    u_dx2 = lambda x: np.exp(-x) + np.exp(x)
    k = lambda x: 1
    k_dx = lambda x: 0
    r = lambda x: 1
    q = lambda x: 1

    k_udx_dx = lambda x: k_dx(x) * u_dx(x) + k(x) * u_dx2(x)
    boundary = (
        DirichletBoundaryCondition(
            location=interval[0],
            mu=u(interval[0])
        ),
        DirichletBoundaryCondition(
            location=interval[1],
            mu=u(interval[1])
        ),
    )

    problems.append(
        FEMProblem(
            label=label,
            interval=interval,
            u=u,
            u_dx=u_dx,
            u_dx2=u_dx2,
            k=k,
            k_dx=k_dx,
            r=r,
            q=q,
            k_udx_dx=k_udx_dx,
            boundary=boundary
        )
    )

    # 2.1.b
    label = "2.1.b"
    kappa = 1
    boundary = (
        DirichletBoundaryCondition(
            location=interval[0],
            mu=u(interval[0])
        ),
        RobinBoundaryCondition(
            location=interval[1],
            mu=k(interval[1]) * u_dx(interval[1]) + kappa * u(interval[1]),
            kappa=kappa),
    )
    problems.append(
        FEMProblem(
            label=label,
            interval=interval,
            u=u,
            u_dx=u_dx,
            u_dx2=u_dx2,
            k=k,
            k_dx=k_dx,
            r=r,
            q=q,
            k_udx_dx=k_udx_dx,
            boundary=boundary
        )
    )

    # 2.1.c
    label = "2.1.c"

    k = lambda x: 1 + x
    k_dx = lambda x: 1
    r = lambda x: k(x)
    q = lambda x: k(x)

    k_udx_dx = lambda x: k_dx(x) * u_dx(x) + k(x) * u_dx2(x)
    boundary = (
        DirichletBoundaryCondition(location=interval[0], mu=u(interval[0])),
        RobinBoundaryCondition(
            location=interval[1],
            mu=k(interval[1]) * u_dx(interval[1]) + kappa * u(interval[1]),
            kappa=kappa),
    )

    problems.append(
        FEMProblem(
            label=label,
            interval=interval,
            u=u,
            u_dx=u_dx,
            u_dx2=u_dx2,
            k=k,
            k_dx=k_dx,
            r=r,
            q=q,
            k_udx_dx=k_udx_dx,
            boundary=boundary
        )
    )

    # 2.1.d
    label = "2.1.d"
    interval = (-1 / 2, 1 / 2)

    kappa_a = 1
    kappa_b = 1
    boundary = (
        RobinBoundaryCondition(
            location=interval[0],
            mu=-k(interval[0]) * u_dx(interval[0]) + kappa_a * u(interval[0]),
            kappa=kappa_a),
        RobinBoundaryCondition(
            location=interval[1],
            mu=k(interval[1]) * u_dx(interval[1]) + kappa_b * u(interval[1]),
            kappa=kappa_b),
    )

    problems.append(
        FEMProblem(
            label=label,
            interval=interval,
            u=u,
            u_dx=u_dx,
            u_dx2=u_dx2,
            k=k,
            k_dx=k_dx,
            r=r,
            q=q,
            k_udx_dx=k_udx_dx,
            boundary=boundary
        )
    )

    # ##################################################################################################################
    # 2.2
    # 2.2.e
    label = "2.2.e"
    interval = (0, 1)
    alpha = 13 / 4
    u = lambda x: np.power(x, alpha)
    u_dx = lambda x: alpha * np.power(x, alpha - 1)
    u_dx2 = lambda x: (alpha - 1) * alpha * np.power(x, alpha - 2)
    u_dx3 = lambda x: (alpha - 2) * (alpha - 1) * alpha * np.power(x, alpha - 3)
    k = lambda x: 1
    k_dx = lambda x: 0
    r = lambda x: 0
    q = lambda x: 1

    k_udx_dx = lambda x: k_dx(x) * u_dx(x) + k(x) * u_dx2(x)
    boundary = (
        DirichletBoundaryCondition(
            location=interval[0],
            mu=u(interval[0])
        ),
        DirichletBoundaryCondition(
            location=interval[1],
            mu=u(interval[1])
        ),
    )

    problems.append(
        FEMProblem(
            label=label,
            interval=interval,
            u=u,
            u_dx=u_dx,
            u_dx2=u_dx2,
            k=k,
            k_dx=k_dx,
            r=r,
            q=q,
            k_udx_dx=k_udx_dx,
            boundary=boundary
        )
    )

    # ##################################################################################################################
    # solve loop
    # ##################################################################################################################
    for problem in problems:
        print(problem)
