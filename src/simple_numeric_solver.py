from typing import *
import numpy as np
from numpy.typing import *
from numba import njit


def ex_euler(f: Callable[[float, ArrayLike], ArrayLike],
             t0: float,
             y0: ArrayLike,
             t_max: float,
             n: int,
             ) -> Tuple[ArrayLike, ArrayLike]:
    """

    :param f: Differential equation
    :param t0: initial time
    :param y0: initial value
    :param t_max: ending time
    :param n: amount steps
    """
    t, y = __ex_euler(f, t0, y0, t_max, n)
    return t, y.squeeze()


@njit()
def __ex_euler(f: Callable[[float, ArrayLike], ArrayLike],
               t0: float,
               y0: ArrayLike,
               t_max: float,
               n: int,
               ) -> Tuple[ArrayLike, ArrayLike]:
    t = np.linspace(t0, t_max, n)
    y = np.zeros((n, y0.shape[0]))
    y[0] = y0
    h = (t_max - t0)/n
    for i in range(1, n):
        y[i] = y[i - 1] + h * f(t[i - 1], y[i - 1])
    return t, y


def newton_solver(f, f_, y, t, tol):
    y_next = y - f(t, y) / f_(t, y)
    if np.linalg.norm(y_next - y) < tol:
        return y_next
    return newton_solver(f, f_, y_next, t, tol)


def im_euler(f: Callable[[float, ArrayLike], ArrayLike],
             t0: float,
             y0: ArrayLike,
             t_max: float,
             n: int,
             f_: Callable[[float, ArrayLike], ArrayLike],
             tol: float,
             ) -> Tuple[ArrayLike, ArrayLike]:
    """

    :param f: Differential equation
    :param t0: initial time
    :param y0: initial value
    :param t_max: ending time
    :param n: amount steps
    :param f_: df/dy. Derivative for newton procedure
    :param tol: tolerance. it will be used as threshold to abort the iteration.
    """

    t = np.linspace(t0, t_max, n)
    y = np.zeros((n, y0.shape[0]))
    y[0] = y0

    for i in range(1, n):
        h = t[i] - t[i - 1]
        q = lambda t, x: x - y[i - 1] - h * f(t, x)
        q_ = lambda t, x: 1 - h * f_(t, x)
        y[i] = newton_solver(q, q_, y[i - 1], t[i], tol)
    return t, y.squeeze()


def trapez(f: Callable[[float, ArrayLike], ArrayLike],
           t0: float,
           y0: ArrayLike,
           t_max: float,
           n: int,
           f_: Callable[[float, ArrayLike], ArrayLike],
           tol: float,
           ) -> Tuple[ArrayLike, ArrayLike]:
    """

    :param f: Differential equation
    :param t0: initial time
    :param y0: initial value
    :param t_max: ending time
    :param n: amount steps
    :param f_: df/dy. Derivative for newton procedure
    :param tol: tolerance. it will be used as threshold to abort the iteration.
    """

    t_a = np.linspace(t0, t_max, n)
    y = np.zeros((n, y0.shape[0]))
    y[0] = y0

    for i in range(1, n):
        h = t_a[i] - t_a[i - 1]
        q = lambda t, x: y[i - 1] + h / 2 * (f(t_a[i - 1], y[i - 1]) + f(t, x)) - x
        q_ = lambda t, x: h / 2 * f_(t, x) - 1
        y[i] = newton_solver(q, q_, y[i - 1], t_a[i], tol)
    return t_a, y.squeeze()


def veb_euler(f: Callable[[float, ArrayLike], ArrayLike],
              t0: float,
              y0: ArrayLike,
              t_max: float,
              n: int,
              ) -> Tuple[ArrayLike, ArrayLike]:
    """
    :param f: Differential equation
    :param t0: initial time
    :param y0: initial value
    :param t_max: ending time
    :param n: amount steps
    """
    t, y = __veb_euler(f, t0, y0, t_max, n)
    return t, y.squeeze()


@njit()
def __veb_euler(f: Callable[[float, ArrayLike], ArrayLike],
                t0: float,
                y0: ArrayLike,
                t_max: float,
                n: int,
                ) -> Tuple[ArrayLike, ArrayLike]:
    t = np.linspace(t0, t_max, n)
    y = np.zeros((n, y0.shape[0]))
    y[0] = y0
    h = (t_max-t0)/n
    for i in range(1, n):
        y[i] = y[i - 1] + h * f(
            t[i - 1] + 1 / 2 * h,
            y[i - 1] + 1 / 2 * h * f(t[i - 1], y[i - 1])
        )
    return t, y


def runge_kutta(A: ArrayLike,
                b: ArrayLike,
                c: ArrayLike,
                f: Callable[[float, ArrayLike], ArrayLike],
                t0: float,
                y0: ArrayLike,
                t_max: float,
                n: int,
                f_: Optional[Callable[[float, ArrayLike], ArrayLike]],
                tol: Optional[float]) -> Tuple[ArrayLike, ArrayLike]:
    """

    :param A: matrix values from butcher tableau
    :param b: bottom row of butcher tableau
    :param c: left column of butcher tableau
    :param f: Differential equation
    :param t0: initial time
    :param y0: initial value
    :param t_max: ending time
    :param n: amount steps
    :param f_: df/dy. Derivative for newton procedure. Only used for implicit procedures
    :param tol: tolerance. it will be used as threshold to abort the iteration. Only used for implicit procedures
    """
    t = np.linspace(t0, t_max, n)
    y = np.zeros((n, y0.shape[0]))
    y[0] = y0
    s = b.shape[0]

    for z in range(1, n):
        # for some reason it's important to calculate it this way instead of using (t_max - t0)/n. doing it the other
        # way around messes with the order of consistency. not sure why
        h = t[z] - t[z - 1]
        # init new k vector in order to save all results between iterations
        k = np.zeros(s)
        for i in range(s):
            ak = 0
            for j in range(i):
                ak += A[i - 1][j] * k[j]
            k[i] = f(t[z - 1] + c[i] * h, y[z - 1] + h * ak)
        y[z] = y[z - 1] + h * np.dot(b, k)

    return t, y.squeeze()


def runge_kutta_heun(
        f: Callable[[float, ArrayLike], ArrayLike],
        t0: float,
        y0: ArrayLike,
        t_max: float,
        n: int,
        p: int = 2) -> Tuple[ArrayLike, ArrayLike]:
    """

    :param f: Differential equation
    :param t0: initial time
    :param y0: initial value
    :param t_max: ending time
    :param n: amount steps
    :param p: Order of consistency
    """
    if p == 2:
        A = np.array([[1]])
        b = np.array([1 / 2, 1 / 2])
        c = np.array([0, 1])
        return runge_kutta(A, b, c, f, t0, y0, t_max, n, None, None)
    if p == 3:
        A = np.array([[1 / 3, 0],
                      [0, 2 / 3]])
        b = np.array([1 / 4, 0, 3 / 4])
        c = np.array([0, 1 / 3, 2 / 3])
        return runge_kutta(A, b, c, f, t0, y0, t_max, n, None, None)
    raise Exception("p should be 2 or 3")


def runge_kutta_ex_euler(f: Callable[[float, ArrayLike], ArrayLike],
                         t0: float,
                         y0: ArrayLike,
                         t_max: float,
                         n: int) -> Tuple[ArrayLike, ArrayLike]:
    """

    :param f: Differential equation
    :param t0: initial time
    :param y0: initial value
    :param t_max: ending time
    :param n: amount steps
    """
    A = np.array([])
    b = np.array([1])
    c = np.array([0])
    return runge_kutta(A, b, c, f, t0, y0, t_max, n, None, None)


def runge_kutta_veb_euler(f: Callable[[float, ArrayLike], ArrayLike],
                          t0: float,
                          y0: ArrayLike,
                          t_max: float,
                          n: int) -> Tuple[ArrayLike, ArrayLike]:
    A = np.array([[1 / 2]])
    b = np.array([0, 1])
    c = np.array([0, 1 / 2])
    return runge_kutta(A, b, c, f, t0, y0, t_max, n, None, None)


def runge_kutta_runge(f: Callable[[float, ArrayLike], ArrayLike],
                      t0: float,
                      y0: ArrayLike,
                      t_max: float,
                      n: int) -> Tuple[ArrayLike, ArrayLike]:
    """

    :param f: Differential equation
    :param t0: initial time
    :param y0: initial value
    :param t_max: ending time
    :param n: amount steps
    """
    A = np.array([[1 / 2, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1],
                  ])
    b = np.array([1 / 6, 2 / 3, 0, 1 / 6])
    c = np.array([0, 1 / 2, 1, 1])
    return runge_kutta(A, b, c, f, t0, y0, t_max, n, None, None)


def runge_kutta_classic(f: Callable[[float, ArrayLike], ArrayLike],
                        t0: float,
                        y0: ArrayLike,
                        t_max: float,
                        n: int) -> Tuple[ArrayLike, ArrayLike]:
    """

    :param f: Differential equation
    :param t0: initial time
    :param y0: initial value
    :param t_max: ending time
    :param n: amount steps
    """
    A = np.array([[1 / 2, 0, 0],
                  [0, 1 / 2, 0],
                  [0, 0, 1],
                  ])
    b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
    c = np.array([0, 1 / 2, 1 / 2, 1])
    return runge_kutta(A, b, c, f, t0, y0, t_max, n, None, None)


def runge_kutta_3_8(f: Callable[[float, ArrayLike], ArrayLike],
                    t0: float,
                    y0: ArrayLike,
                    t_max: float,
                    n: int) -> Tuple[ArrayLike, ArrayLike]:
    """

    :param f: Differential equation
    :param t0: initial time
    :param y0: initial value
    :param t_max: ending time
    :param n: amount steps
    """
    A = np.array([[1 / 3, 0, 0],
                  [-1 / 3, 1, 0],
                  [1, -1, 1],
                  ])
    b = np.array([1 / 8, 3 / 8, 3 / 8, 1 / 8])
    c = np.array([0, 1 / 3, 2 / 3, 1])
    return runge_kutta(A, b, c, f, t0, y0, t_max, n, None, None)


if __name__ == '__main__':
    m_0 = 3
    t_end = 10
    tol = 1e-2
    t0 = 0
    y0 = np.array([0])
    n = 100000


    @njit()
    def example_dgl(t: float, y: ArrayLike, m=m_0):
        y = y[0]
        return -m * (y - np.cos(t))


    @njit()
    def example_dgl_(t: float, y: ArrayLike, m=m_0):
        return -m * np.ones(y.shape[0])


    def solution(time, m=m_0):
        return 1 / (m * m + 1) * (-(m * m) * np.exp(-m * time) + m * np.sin(time) + m * m * np.cos(time))


    y_exakt = solution(t_end)
    t_ex_eul, y_ex_eul = ex_euler(f=example_dgl, t0=t0, y0=y0, t_max=t_end, n=n)
    t_rk_ex_eul, y_rk_ex_eul = runge_kutta_ex_euler(f=example_dgl, t0=t0, y0=y0, t_max=t_end, n=n)
    t_im_eul, y_im_eul = im_euler(f=example_dgl, f_=example_dgl_, t0=t0, y0=y0, t_max=t_end, n=n, tol=tol)
    t_trap, y_trap = trapez(f=example_dgl, f_=example_dgl_, t0=t0, y0=y0, t_max=t_end, n=n, tol=tol)
    t_veb_eul, y_veb_eul = veb_euler(f=example_dgl, t0=t0, y0=y0, t_max=t_end, n=n)
    t_rk_veb_eul, y_rk_veb_eul = runge_kutta_veb_euler(f=example_dgl, t0=t0, y0=y0, t_max=t_end, n=n)
    t_rk_heun_2, y_rk_heun_2 = runge_kutta_heun(f=example_dgl, t0=t0, y0=y0, t_max=t_end, n=n)
    t_rk_heun_3, y_rk_heun_3 = runge_kutta_heun(f=example_dgl, t0=t0, y0=y0, t_max=t_end, n=n, p=3)
    t_rk_runge, y_rk_runge = runge_kutta_runge(f=example_dgl, t0=t0, y0=y0, t_max=t_end, n=n)
    t_rk_classic, y_rk_classic = runge_kutta_classic(f=example_dgl, t0=t0, y0=y0, t_max=t_end, n=n)
    t_rk_3_8, y_rk_3_8 = runge_kutta_3_8(f=example_dgl, t0=t0, y0=y0, t_max=t_end, n=n)

    toString = lambda value: f"{value[-1]} {value[-1] - y_exakt}"

    print(f"y = {y_exakt}\n"
          f"y_ex_euler = {toString(y_ex_eul)}\n"
          f"y_rk_ex_euler = {toString(y_rk_ex_eul)} {y_ex_eul[-1] - y_rk_ex_eul[-1]}\n"
          f"y_im_euler = {toString(y_im_eul)}\n"
          f"y_trapez = {toString(y_trap)}\n"
          f"y_veb_euler = {toString(y_veb_eul)}\n"
          f"y_rk_veb_euler = {toString(y_rk_veb_eul)} {y_veb_eul[-1] - y_rk_veb_eul[-1]}\n"
          f"y_rk_heun p=2 = {toString(y_rk_heun_2)}\n"
          f"y_rk_heun p=3 = {toString(y_rk_heun_3)}\n"
          f"y_rk_runge = {toString(y_rk_runge)}\n"
          f"y_rk_classic = {toString(y_rk_classic)}\n"
          f"y_rk_3_8 = {toString(y_rk_3_8)}\n"

          )
