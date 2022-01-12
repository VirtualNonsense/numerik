import numpy as np
from dataclasses import dataclass

from typing import *
from scipy.integrate import quad
from numerical_quadrature import quad_gauss
from numpy.typing import *

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


def RwpFem1d(
        xGit,
        k: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        r: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        q: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        f: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        rba: Tuple[int, float, float],
        rbb: Tuple[int, float, float],
        eltyp: int,
        intyp: int) -> Tuple[ArrayLike, ArrayLike]:
    """
    :param xGit: Grid points [x0, ..., xN]
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
    :param eltyp:
        1: linear approach
        2: cubic approach
    :param intyp:
        amount of nodes
    :return:
    """
    pass


def lin_elem(k, q, f, rbr, rbl, in_typ, n_e) -> Tuple[ArrayLike, ArrayLike]:
    def phi(x_i, index):
        if index == 0:
            return 1 - x_i
        return x_i

    k_i = np.zeros(shape=[n_e, n_e])
    f_i = np.zeros(n_e)
    F = lambda x_i: (rbr - rbl) * x_i + rbl
    tmp = [-1, 1]
    hi = abs(rbr - rbl)
    if in_typ == 0:
        for a in range(n_e):
            fun_2 = lambda x_i: f(F(x_i)) * phi(x_i, a)
            f_i[a] = hi * quad(fun_2, 0, 1)

            for b in range(n_e):
                fun = lambda x_i: k(F(x_i)) / np.square(hi) * tmp[a] * tmp[b] + q(F(x_i)) * phi(x_i, a) * phi(x_i, b)
                k_i[a, b] = hi * quad(fun, 0, 1)
        return k_i, f_i
    for a in range(n_e):
        fun_2 = lambda x_i: f(F(x_i)) * phi(x_i, a)
        f_i[a] = hi * quad_gauss(fun_2, -1, 1, n=in_typ)
        for b in range(n_e):
            fun = lambda x_i: k(F(x_i)) / np.square(hi) * tmp[a] * tmp[b] + q(F(x_i)) * phi(x_i, a) * phi(x_i, b)
            k_i[a, b] = hi * quad_gauss(fun, -1, 1, in_typ)



def fem_bc_solver(
        x_git: ArrayLike,
        k: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        r: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        q: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        f: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        boundaries: Tuple[BoundaryCondition, BoundaryCondition],
        el_typ: int,
        in_typ: int) -> Tuple[ArrayLike, ArrayLike]:
    """
    1 dimensional fem solver vor boundary condition problems

    :param x_git: Grid points [x0, ..., xN]
    :param k: diffusion equation
    :param r: convection equation
    :param q: reaction equation
    :param f: right side
    :param boundaries: Boundary conditions
        0: left boundary
        1: right boundary
        see subclasses for details
    :param el_typ:
        1: linear approach
        2: cubic approach
    :param in_typ:
        amount of nodes
    :return:
        0:
    """
    # preparing variables
    m_e = len(x_git) - 1
    n_g = el_typ * m_e + 1
    x_kno = np.zeros(n_g)
    u_kno = np.zeros(n_g)
    k_h = np.zeros(shape=[n_g, n_g])
    f_h = np.zeros(n_g)

    kn_el = np.zeros(shape=[m_e, 5])
    kn_el[:, 0] = el_typ
    kn_el[:, 1] = in_typ

    if el_typ == 1:
        x_kno = x_git
        for i in range(m_e):
            kn_el[i, 2] = i
            kn_el[i, 3] = i + 1
    elif el_typ == 2:
        kn_el[:, 3] = np.array([i for i in range(1, n_g - 2, 2)])
        kn_el[:, 3] = np.array([i for i in range(2, n_g - 1, 2)])
        kn_el[:, 3] = np.array([i for i in range(3, n_g, 2)])
        j = 0
        for i in range(m_e):
            x_kno[j] = x_git[i]
            x_kno[j + 1] = (x_git[i + 1] - x_git[i]) / 2 + x_git[i]
            x_kno[j + 2] = x_git[i + 1]
            j += 2
    for i in range(m_e):
        pass


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
