import numpy as np

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from pprint import pprint
    from fem_solver import rwp_fem_1d
    import numpy as np

    ####################################################################################################################
    # problems from 2.2

    fig: Figure = plt.figure("Ex 2_2")
    u_abcd = lambda x: np.exp(-x) + np.exp(x) + 1
    u_abcd_dx = lambda x: -np.exp(-x) + np.exp(x)

    x_grid_abc = np.array([0, .1, .3, .4, .45, .6, .8, .88, .9, 1])
    el_type = 2
    in_type = 2

    k = lambda x: 1
    r = lambda x: 1
    q = lambda x: 1
    f = lambda x: np.exp(x) - np.exp(-x) + 1
    left_boundary = (1, 0, u_abcd(x_grid_abc[0]))
    right_boundary = (1, 0, u_abcd(x_grid_abc[-1]))
    u_a, x_a = rwp_fem_1d(x_grid=x_grid_abc,
                          k=k,
                          r=r,
                          q=q,
                          f=f,
                          rba=left_boundary,
                          rbb=right_boundary,
                          el_typ=el_type,
                          in_typ=in_type)

    ax: Axes = fig.add_subplot(4, 1, 1)
    ax.plot(x_a, u_a, linestyle="", marker="x", label="approx a")
    ax.plot(x_a, u_abcd(x_a), label="solution")
    ax.legend()

    right_boundary = (3, 1, k(x_grid_abc[-1]) * u_abcd_dx(x_grid_abc[-1]) + u_abcd(x_grid_abc[-1]))
    u_b, x_b = rwp_fem_1d(x_grid=x_grid_abc,
                          k=k,
                          r=r,
                          q=q,
                          f=f,
                          rba=left_boundary,
                          rbb=right_boundary,
                          el_typ=el_type,
                          in_typ=in_type)

    ax: Axes = fig.add_subplot(4, 1, 2)
    ax.plot(x_b, u_b, linestyle="", marker="x", label="approx b")
    ax.plot(x_b, u_abcd(x_b), label="solution")
    ax.legend()

    k = lambda x: 1 + x
    r = lambda x: k(x)
    q = lambda x: k(x)
    f = lambda x: x * (np.exp(x) - np.exp(-x)) + 1 + x
    right_boundary = (3, 1, k(x_grid_abc[-1]) * u_abcd_dx(x_grid_abc[-1]) + u_abcd(x_grid_abc[-1]))
    u_c, x_c = rwp_fem_1d(x_grid=x_grid_abc,
                          k=k,
                          r=r,
                          q=q,
                          f=f,
                          rba=left_boundary,
                          rbb=right_boundary,
                          el_typ=el_type,
                          in_typ=in_type)

    ax: Axes = fig.add_subplot(4, 1, 3)
    ax.plot(x_c, u_c, linestyle="", marker="x", label="approx c")
    ax.plot(x_c, u_abcd(x_c), label="solution")
    ax.legend()

    x_grid_d = x_grid_abc - 1 / 2
    left_boundary = (3, 0, -k(x_grid_d[0] * u_abcd_dx(x_grid_d[0])))
    right_boundary = (3, 1, k(x_grid_d[-1]) * u_abcd_dx(x_grid_d[-1]) + u_abcd(x_grid_d[-1]))
    u_d, x_d = rwp_fem_1d(x_grid=x_grid_d,
                          k=k,
                          r=r,
                          q=q,
                          f=f,
                          rba=left_boundary,
                          rbb=right_boundary,
                          el_typ=el_type,
                          in_typ=in_type)

    ax: Axes = fig.add_subplot(4, 1, 4)
    ax.plot(x_d, u_d, linestyle="", marker="x", label="approx d")
    ax.plot(x_d, u_abcd(x_d), label="solution")
    ax.legend()

    plt.show()
