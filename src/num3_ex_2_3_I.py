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

    x_grid_abce = np.array([0, .1, .3, .4, .45, .6, .8, .88, .9, 1])
    a_abce = x_grid_abce[0]
    b_abce = x_grid_abce[-1]
    el_type = 1
    in_type = 0
    fig.suptitle(f"exercise 2.2 using el_type: {el_type}, in_type: {in_type}")

    k = lambda x: 1
    r = lambda x: 1
    q = lambda x: 1
    f = lambda x: np.exp(x) - np.exp(-x) + 1
    left_boundary = (1, 0, u_abcd(a_abce))
    right_boundary = (1, 0, u_abcd(b_abce))
    u_a, x_a = rwp_fem_1d(x_grid=x_grid_abce,
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

    right_boundary = (3, 1, k(b_abce) * u_abcd_dx(b_abce) + u_abcd(b_abce))
    u_b, x_b = rwp_fem_1d(x_grid=x_grid_abce,
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
    right_boundary = (3, 1, k(b_abce) * u_abcd_dx(b_abce) + u_abcd(b_abce))
    u_c, x_c = rwp_fem_1d(x_grid=x_grid_abce,
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

    x_grid_d = x_grid_abce - 1 / 2
    a_d = x_grid_d[0]
    b_d = x_grid_d[-1]
    left_boundary = (3, 0, -k(a_d) * u_abcd_dx(a_d))
    right_boundary = (3, 1, k(b_d) * u_abcd_dx(b_d) + u_abcd(b_d))
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

    ####################################################################################################################
    # problems from 2.3
    alpha = 1 / 4
    u_e = lambda x: np.power(x, alpha)
    u_e_dx = lambda x: alpha * np.power(x, alpha - 1)
    u_e_dx2 = lambda x: alpha * (alpha - 1) * np.power(x, alpha - 2)

    fig: Figure = plt.figure("Ex 2_3")
    fig.suptitle(f"exercise 2.3 using el_type: {el_type}, in_type: {in_type}")
    k = lambda x: 1
    r = lambda x: 0
    q = lambda x: 1
    f = lambda x: u_e(x) - u_e_dx2(x)
    left_boundary = (1, 0, u_e(a_abce))
    right_boundary = (1, 0, u_e(b_abce))
    u_e_approx, x_e = rwp_fem_1d(x_grid=x_grid_abce,
                                 k=k,
                                 r=r,
                                 q=q,
                                 f=f,
                                 rba=left_boundary,
                                 rbb=right_boundary,
                                 el_typ=el_type,
                                 in_typ=in_type)

    ax: Axes = fig.add_subplot(1, 1, 1)
    ax.plot(x_e, u_e_approx, linestyle="", marker="x", label="approx a")
    ax.plot(x_e, u_e(x_e), label="solution")
    ax.legend()

    plt.show()
