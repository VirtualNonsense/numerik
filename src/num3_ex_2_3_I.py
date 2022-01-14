import numpy as np

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes

    from fem_solver import rwp_fem_1d
    import numpy as np

    u_abcd = lambda x: np.exp(-x) + np.exp(x) + 1

    x_grid = np.array([i * 0.2 for i in range(5)])
    k = lambda x: 1
    r = lambda x: 1
    q = lambda x: 1
    f = lambda x: np.exp(x) - np.exp(-x) + 1
    left_boundary = (1, 0, u_abcd(x_grid[0]))
    right_boundary = (1, 0, u_abcd(x_grid[-1]))

    u, x = rwp_fem_1d(x_grid=x_grid,
                      k=k,
                      r=r,
                      q=q,
                      f=f,
                      rba=left_boundary,
                      rbb=right_boundary,
                      el_typ=1,
                      in_typ=1)
    fig: Figure = plt.figure("a")
    ax: Axes = fig.add_subplot()
    ax.plot(x, u, label="approx")
    ax.plot(x, u_abcd(x), label="solution")
    ax.legend()
    plt.show()
