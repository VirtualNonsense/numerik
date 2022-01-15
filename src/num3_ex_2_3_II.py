if __name__ == '__main__':
    import numpy as np
    from typing import *
    from numpy.typing import *
    from fem_solver import rwp_fem_1d
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

    el_typs = [1, 2]
    in_typs = [0, 1, 2, 3]

    alpha = 13 / 4
    u = lambda x: np.power(x, alpha)
    u_dx = lambda x: alpha * np.power(x, alpha - 1)
    u_dx2 = lambda x: alpha * (alpha - 1) * np.power(x, alpha - 2)

    k = lambda x: 1
    r = lambda x: 0
    q = lambda x: 1
    f = lambda x: u(x) - u_dx2(x)
    a = 0
    b = 1
    left_boundary = (1, 0, u(a))
    right_boundary = (1, 0, u(b))
    j_start = 1
    j_end = 8
    js = np.array([i for i in range(j_start, j_end)])
    figs: List[Figure] = []
    for el_typ_index, el_typ in enumerate(el_typs):
        figs.append(plt.figure(f"Ex 2_3 apriori fehler {el_typ}"))
        figs[-1].suptitle(f"exercise 2.3 using el_type: {el_typ}")
        axs: List[Axes] = figs[-1].subplots(len(in_typs), 1, sharex="col")
        axs[-1].set_xlabel("step width")
        for in_typ_index, in_typ in enumerate(in_typs):
            errors = np.zeros(js.shape)
            h_array = np.zeros(js.shape)
            for i, j in enumerate(js):
                h = np.power(1 / 2, j)
                x_grid = np.arange(start=a, stop=b + h, step=h)

                u_approx, x_e = rwp_fem_1d(x_grid=x_grid,
                                           k=k,
                                           r=r,
                                           q=q,
                                           f=f,
                                           rba=left_boundary,
                                           rbb=right_boundary,
                                           el_typ=el_typ,
                                           in_typ=in_typ)
                u_exact = u(x_e)
                errors[i] = np.abs(u_approx - u_exact).max()
                h_array[i] = h

            ax: Axes = axs[in_typ_index]
            ax.loglog(h_array, errors, linestyle="", marker="x", label=f"error:\nel_typ {el_typ}\nin_typ: {in_typ}\n")
            ax.set_ylabel("absolute max error")
            ax.legend()
    plt.show()
