from simple_numeric_solver import *
from numpy.typing import *

def exper_konv_ord(method: Callable[[int], ArrayLike],
                   n0,
                   m=1) -> ArrayLike:
    """
    :param method: should be a preconfigured solver. The only allowed parameter is the amount of iteration. the return value
    is expected to be the last iteration step
    :param n0: initial amount of iterations
    :param m: iterations. the initial iterations will be doubled after each iteration
    """
    p = np.zeros(m)
    log_e_2 = np.log(2)
    for i in range(m):
        y_1 = method(n0)
        y_2 = method(2 * n0)
        y_3 = method(4 * n0)
        dy_12 = np.linalg.norm(y_1 - y_2)
        dy_23 = np.linalg.norm(y_2 - y_3)
        dy = dy_12/dy_23
        p[i] = np.log(dy) / log_e_2
        n0 *= 2
    return p


if __name__ == '__main__':
    m_0 = 3
    t_end = 10
    tol = 1e-2
    t0 = 0
    y0 = np.array([0])
    n0 = 1600
    m = 5


    @njit()
    def example_dgl(t: float, y: ArrayLike, m=m_0):
        y = y[0]
        return -m * (y - np.cos(t))


    @njit()
    def example_dgl_(t: float, y: ArrayLike, m=m_0):
        return -m * np.ones(y.shape[0])


    e_euler = lambda n: ex_euler(f=example_dgl_, t0=t0, y0=y0, t_max=t_end, n=n)[1][-1]
    i_euler = lambda n: im_euler(f=example_dgl_, f_=example_dgl_, t0=t0, y0=y0, t_max=t_end, n=n, tol=tol)[1][-1]
    trap = lambda n: trapez(f=example_dgl_, f_=example_dgl_, t0=t0, y0=y0, t_max=t_end, n=n, tol=tol)[1][-1]
    v_euler = lambda n: veb_euler(f=example_dgl_, t0=t0, y0=y0, t_max=t_end, n=n)[1][-1]
    rk_3_8 = lambda n: runge_kutta_3_8(f=example_dgl, t0=t0, y0=y0, t_max=t_end, n=n)[1][-1]
    rk_heun_2= lambda n: runge_kutta_heun(f=example_dgl, t0=t0, y0=y0, t_max=t_end, n=n, p=2)[1][-1]
    rk_heun_3= lambda n: runge_kutta_heun(f=example_dgl, t0=t0, y0=y0, t_max=t_end, n=n, p=3)[1][-1]
    rk_ex_euler= lambda n: runge_kutta_ex_euler(f=example_dgl, t0=t0, y0=y0, t_max=t_end, n=n)[1][-1]
    rk_veb_euler= lambda n: runge_kutta_veb_euler(f=example_dgl, t0=t0, y0=y0, t_max=t_end, n=n)[1][-1]

    e_euler_p = exper_konv_ord(e_euler, n0, m)
    im_euler_p = exper_konv_ord(i_euler, n0, m)
    trap_p = exper_konv_ord(trap, n0, m)
    v_euler_p = exper_konv_ord(v_euler, n0, m)
    rk_3_8_p = exper_konv_ord(rk_3_8, n0, m)
    rk_heun_2_p = exper_konv_ord(rk_heun_2, n0, m)
    rk_heun_3_p = exper_konv_ord(rk_heun_3, n0, m)
    rk_ex_euler_p = exper_konv_ord(rk_ex_euler, n0, m)
    rk_veb_euler_p = exper_konv_ord(rk_veb_euler, n0, m)


    def toString(array: ArrayLike):
        return ", ".join([str(k) for k in array])


    print(f"ex euler | 1 | {toString(e_euler_p)}")
    print(f"im euler | 1 | {toString(im_euler_p)}")
    print(f"trapez | 1 | {toString(trap_p)}")
    print(f"veb euler | 2 | {toString(v_euler_p)}")
    print(f"rk 3/8 | 4 | {toString(rk_3_8_p)}")
    print(f"rk heun | 2 | {toString(rk_heun_2_p)}")
    print(f"rk heun | 3 | {toString(rk_heun_3_p)}")
    print(f"rk ex euler | 1 | {toString(rk_ex_euler_p)}")
    print(f"rk veb euler | 2 | {toString(rk_veb_euler_p)}")
