

if __name__ == '__main__':
    from fdm_ivbc_solver import *
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

    ####################################################################################################################
    # problem
    ####################################################################################################################

    # Example (a)
    # u = lambda x, t: np.sin(x) * np.cos(t)
    # u_dt = lambda x, t: -np.sin(x) * np.sin(t)
    # u_dx = lambda x, t: np.cos(x) * np.cos(t)
    # u_dx2 = lambda x, t: -np.sin(x) * np.cos(t)

    # k = lambda x: 1
    # k_dx = lambda x: 0
    # q = lambda x: 1

    # Example (b)
    u = lambda x, t: np.exp(-2 * t) * np.cos(np.pi * x)
    u_dt = lambda x, t: -2 * np.exp(-2 * t) * np.cos(np.pi * x)
    u_dx = lambda x, t: -np.pi * np.exp(-2 * t) * np.sin(np.pi * x)
    u_dx2 = lambda x, t: -np.pi * np.pi * np.exp(-2 * t) * np.cos(np.pi * x)
    n = 2

    k = lambda x: 1 + 2 * np.power(x, n)
    k_dx = lambda x: 2 * n * np.power(x, n - 1)
    q = lambda x: (1 - x) * x

    # Playground
    # u = lambda x, t: np.sin(x) * np.cos(t)
    # u_dt = lambda x, t: -np.sin(x) * np.sin(t)
    # u_dx = lambda x, t: np.cos(x) * np.cos(t)
    # u_dx2 = lambda x, t: -np.sin(x) * np.cos(t)

    # n = 4
    # k = lambda x: 1 + np.power(x, n)
    # k_dx = lambda x: n * np.power(x, n - 1)
    # q = lambda x: np.sin(x)

    # do not change
    mu_a = lambda t: u(a, t)
    mu_b = lambda t: u(b, t)

    phi = lambda x: u(x, 0)

    f = lambda x, t: u_dt(x, t) - (k_dx(x) * u_dx(x, t) + k(x) * u_dx2(x, t)) + q(x) * u(x, t)

    ####################################################################################################################
    # settings
    ####################################################################################################################

    # n + 1, space discretization
    # n_p1 = 11
    n_p1 = 41

    a = 0
    b = 1
    hx = (b - a) / (n_p1 - 1)
    x = (a, b, n_p1)

    # time discretization
    t_start = 0
    t_end = 1
    # default for explicit euler
    kmax = k(np.linspace(a, b, 1000, endpoint=True)).max()
    tau_ex = np.power(hx, 2) / (2 * kmax)
    m_ex = int(np.ceil((t_end - t_start) / tau_ex) + 1)
    t_explicit = (t_start, t_end, m_ex)

    # default for implicit euler
    tau_im = np.power(hx, 2)
    m_im = int(np.ceil((t_end - t_start) / tau_im) + 1)
    t_implicit = (t_start, t_end, m_im)

    # default for Crank-N
    tau_cn = hx
    m_cn = int(np.ceil((t_end - t_start) / tau_cn) + 1)
    t_cn = (t_start, t_end, m_cn)

    draw_solutions = False
    draw_difference = True
    draw_dgl = False

    ####################################################################################################################
    # solve
    ####################################################################################################################
    explicit = fdm_ivbc_solver(space=x,
                               time=t_explicit,
                               k=k,
                               q=q,
                               f=f,
                               mu_a=mu_a,
                               mu_b=mu_b,
                               phi=phi,
                               sigma=0)

    crank_nicolson = fdm_ivbc_solver(space=x,
                                     time=t_cn,
                                     k=k,
                                     q=q,
                                     f=f,
                                     mu_a=mu_a,
                                     mu_b=mu_b,
                                     phi=phi,
                                     sigma=1 / 2)

    implicit = fdm_ivbc_solver(space=x,
                               time=t_implicit,
                               k=k,
                               q=q,
                               f=f,
                               mu_a=mu_a,
                               mu_b=mu_b,
                               phi=phi,
                               sigma=1)
    solutions = {"explicit": explicit, "crank nicolson": crank_nicolson, "implicit": implicit}
    ####################################################################################################################
    # plot
    ####################################################################################################################
    for i, (key, item) in enumerate(solutions.items()):
        X = item[1]
        T = item[2]
        xx, tt = np.meshgrid(X, T)
        approx = item[0]
        solution = u(xx, tt)
        print(f"{key}: h={X[1] - X[0]} tau={T[1] - T[0]} error={np.abs(solution - approx).max()}")
        if draw_solutions:
            fig: Figure = plt.figure()
            fig.suptitle(key)
            fig.canvas.manager.set_window_title(f"{key} - solutions")
            ax: Axes = fig.add_subplot(1, 2, 1, projection='3d')
            ax.set_title("solution")
            ax.set_xlabel("space")
            ax.set_ylabel("time")
            ax.plot_surface(xx, tt, solution, label="solution")
            ax: Axes = fig.add_subplot(1, 2, 2, projection='3d')
            ax.set_title("approx")
            ax.set_xlabel("space")
            ax.set_ylabel("time")
            ax.plot_surface(xx, tt, approx, label="Approx. using explicit euler")
        if draw_difference:
            fig: Figure = plt.figure()
            fig.suptitle(key)
            fig.canvas.manager.set_window_title(f"{key} - solution - approx")
            ax: Axes = fig.add_subplot(1, 1, 1, projection='3d')
            ax.set_title("solution - approx")
            ax.set_xlabel("space")
            ax.set_ylabel("time")
            ax.plot_surface(xx, tt, solution - approx, label="Approx. using explicit euler")

    if draw_dgl:
        X = implicit[1]
        T = implicit[2]
        xx, tt = np.meshgrid(X, T)
        dgl = f(xx, tt)
        fig: Figure = plt.figure()
        fig.suptitle("dgl")
        fig.canvas.manager.set_window_title(f"dgl")
        ax: Axes = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_title("dgl")
        ax.set_xlabel("space")
        ax.set_ylabel("time")
        ax.plot_surface(xx, tt, dgl, label="Approx. using explicit euler")

    plt.show()
