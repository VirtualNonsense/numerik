import numpy as np
from dataclasses import dataclass

from typing import *
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
