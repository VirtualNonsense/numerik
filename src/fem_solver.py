import numpy as np

from typing import *
from scipy.integrate import quad
from numerical_quadrature import quad_gauss
from numpy.typing import *
from numpy.linalg import solve


def _elem(
        phi: Callable[[float, int], float],
        phi_dx: Callable[[float, int], float],
        k: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        r: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        q: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        f: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        right_boundary,
        left_boundary,
        in_typ,
        n_e):
    # init return tensors
    k_i = np.zeros(shape=[n_e, n_e])
    f_i = np.zeros(n_e)

    """function used to transfer xi element of [0, 1] into [x_i_1, x_i]]"""
    F = lambda xi: (right_boundary - left_boundary) * xi + left_boundary

    """distance between right boundary and left boundary"""
    h_i = np.abs(right_boundary - left_boundary)

    # setting integration method
    integrate = lambda function: quad_gauss(function, 0, 1, n=in_typ)
    if in_typ == 0:
        integrate = lambda function: quad(function, 0, 1)
    for row in range(n_e):
        fun_2 = lambda xi: f(F(xi)) * phi(xi, row)
        f_i[row] = h_i * integrate(fun_2)
        for column in range(n_e):
            fun = lambda xi: k(F(xi)) / np.square(h_i) * phi_dx(xi, row) * phi_dx(xi, column) \
                             + r(f(xi)) * phi_dx(xi, column) * phi(xi, row) * 1 / h_i \
                             + q(F(xi)) * phi(xi, row) * phi(xi, column)
            k_i[row, column] = h_i * integrate(fun)
    return k_i, f_i


def lin_elem(
        k: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        r: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        q: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        f: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        right_boundary,
        left_boundary,
        in_typ,
        n_e) -> Tuple[ArrayLike, ArrayLike]:
    """
    Constructing block matrix k_i and vector f_i using the linear approach
    :param k: diffusion equation
    :param r: convection equation
    :param q: reaction equation
    :param f: right side
    :param right_boundary: x_i, right boundary of current element
    :param left_boundary: x_(i-1), left boundary of current element
    :param in_typ:
    :param n_e:
    :return: K_i, f_i
    """

    def phi(xi: float, index: int) -> float:
        """
        linear approach (for reference interval [0, 1])
        :param xi:
        :param index:
        :return:
        """
        if index == 0:
            return 1 - xi
        return xi

    def phi_dx(xi: float, index: int) -> float:
        """
        weak derivation of phi
        :param xi:
        :param index:
        :return:
        """
        if index == 0:
            return -1
        return 1

    return _elem(phi, phi_dx, k, r, q, f, right_boundary, left_boundary, in_typ, n_e)


def quad_elem(
        k: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        r: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        q: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        f: Callable[[Union[ArrayLike, float]], Union[ArrayLike, float]],
        right_boundary,
        left_boundary,
        in_typ,
        n_e) -> Tuple[ArrayLike, ArrayLike]:
    """
    Constructing block matrix k_i and vector f_i using the linear approach
    :param k: diffusion equation
    :param r: convection equation
    :param q: reaction equation
    :param f: right side
    :param right_boundary: x_i, right boundary of current element
    :param left_boundary: x_(i-1), left boundary of current element
    :param in_typ:
    :param n_e:
    :return: K_i, f_i
    """

    def phi(xi: float, index: int) -> float:
        """
        cubic approach (for reference interval [0, 1])
        :param xi:
        :param index:
        :return:
        """
        if index == 0:
            return (2 * xi - 1) * (xi - 1)
        if index == 1:
            return 4 * xi * (1 - xi)
        return xi * (2 * xi - 1)

    def phi_dx(xi: float, index: int) -> float:
        """
        weak derivation of phi
        :param xi:
        :param index:
        :return:
        """
        if index == 0:
            return 4 * xi - 3
        if index == 1:
            return 4 - 8 * xi
        return 4 * xi - 1

    return _elem(phi, phi_dx, k, r, q, f, right_boundary, left_boundary, in_typ, n_e)


def rwp_fem_1d(
        x_grid: ArrayLike,
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

    :param x_grid: Grid points [x0, ..., xN]
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
    ####################################################################################################################
    # preparing variables

    """amount of finite elements (amount of spaces between grid nodes)"""
    m_e = len(x_grid) - 1

    """Dimension of K_h. Amount of p nodes"""
    n_g = el_typ * m_e + 1

    """"""
    x_kno = np.zeros(n_g)

    """"""
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
    kn_el = np.zeros(shape=[m_e, 5], dtype=int)
    kn_el[:, 0] = el_typ
    kn_el[:, 1] = in_typ

    ####################################################################################################################
    # setting up node table and nodes
    if el_typ == 1:
        # returning x_git in this case
        x_kno = x_grid
        for i in range(m_e):
            kn_el[i, 2] = i
            kn_el[i, 3] = i + 1
    elif el_typ == 2:
        kn_el[:, 2] = np.array([i for i in range(0, n_g - 2, 2)], dtype=int)
        kn_el[:, 3] = np.array([i for i in range(1, n_g - 1, 2)], dtype=int)
        kn_el[:, 4] = np.array([i for i in range(2, n_g, 2)], dtype=int)

        # calculating "in between grid points"
        j = 0
        for i in range(m_e):
            x_kno[j] = x_grid[i]
            x_kno[j + 1] = (x_grid[i + 1] - x_grid[i]) / 2 + x_grid[i]
            x_kno[j + 2] = x_grid[i + 1]
            j += 2

    ####################################################################################################################
    # calculating k_h and f_h
    """
    dimension of K^i_h
    el_typ == 1: 2
    el_typ == 2: 3
    """
    n_e = el_typ + 1

    elem = lambda x_i, x_i_1: quad_elem(k, r, q, f, x_i, x_i_1, kn_el[0, 1], n_e)
    get_rb = lambda element_index: (x_kno[kn_el[element_index, 4]], x_kno[kn_el[element_index, 2]])
    if el_typ == 1:
        get_rb = lambda element_index: (x_kno[kn_el[element_index, 3]], x_kno[kn_el[element_index, 2]])
        elem = lambda x_i, x_i_1: lin_elem(k, r, q, f, x_i, x_i_1, kn_el[0, 1], n_e)

    for i in range(m_e):
        # get boundaries
        x_i, x_i_1 = get_rb(i)

        # get block matrix and right side
        k_i, f_i = elem(x_i, x_i_1)

        # patch matrix and vector into f_h and k_h
        for row in range(n_e):
            # extracting global row index from look up table (+2 to account for first and second column)
            global_row = kn_el[i, 2 + row]
            f_h[global_row] += f_i[row]

            for column in range(n_e):
                # extracting global column index from look up table (+2 to account for first and second column)
                global_column = kn_el[i, 2 + column]
                k_h[global_row, global_column] += k_i[row, column]

    ####################################################################################################################
    # boundary conditions

    # dirichlet bc
    if rba[0] == 1:
        u_kno[0] = rba[2]
        k_h2 = k_h
        k_h2[0, :] = 0
        f_h2 = f_h
        f_h2[0] = 0
        f_h = f_h2 - k_h2 @ u_kno

        f_h[0] = rba[2]
        k_h[:, 0] = 0
        k_h[0, :] = 0
        k_h[0, 0] = 1

    if rbb[0] == 1:
        u_kno[-1] = rbb[2]
        tmp = u_kno[0]
        u_kno[0] = 0
        k_h2 = k_h
        k_h2[-1, :] = 0
        f_h2 = f_h
        f_h2[-1] = 0
        f_h = f_h2 - k_h2 @ u_kno

        f_h[-1] = rba[2]
        k_h[:, -1] = 0
        k_h[-1, :] = 0
        k_h[-1, -1] = 1
        u_kno[0] = tmp

    # neumann bc
    if rba[0] == 2:
        f_h[0] += rba[2]
    if rbb[0] == 2:
        f_h[-1] += rbb[2]

    # robin bc
    if rba[0] == 3:
        k_h[0, 0] += rba[1]
        f_h[0] += rba[2]
    if rbb[0] == 3:
        k_h[-1, -1] += rbb[1]
        f_h[-1] += rbb[2]

    ####################################################################################################################
    # solve equations
    u_kno = solve(k_h, f_h)
    return u_kno, x_kno
