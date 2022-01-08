if __name__ == '__main__':
    import numpy as np
    from pprint import pprint
    from fdm_ivbc_solver import *
    from typing import *
    from numpy.typing import *
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from scipy.optimize import curve_fit

    ####################################################################################################################
    # problem
    ####################################################################################################################
    # Example (b)
    u = lambda x, t: np.exp(-2 * t) * np.cos(np.pi * x)
    u_dt = lambda x, t: -2 * np.exp(-2 * t) * np.cos(np.pi * x)
    u_dx = lambda x, t: -np.pi * np.exp(-2 * t) * np.sin(np.pi * x)
    u_dx2 = lambda x, t: -np.pi * np.pi * np.exp(-2 * t) * np.cos(np.pi * x)
    n = 2

    k = lambda x: 1 + 2 * np.power(x, n)
    k_dx = lambda x: 2 * n * np.power(x, n - 1)
    q = lambda x: (1 - x) * x

    # do not change
    mu_a = lambda t: u(a, t)
    mu_b = lambda t: u(b, t)

    phi = lambda x: u(x, 0)

    f = lambda x, t: u_dt(x, t) - (k_dx(x) * u_dx(x, t) + k(x) * u_dx2(x, t)) + q(x) * u(x, t)

    ####################################################################################################################
    # settings
    ####################################################################################################################
    draw_fit = True

    # n + 1, space discretization
    # n_p1 = 11
    n_p1_start = 60
    n_p1_end = 5
    n_p1_steps = 20
    a = 0
    b = 1
    sigmas = [0, 1 / 2, 1]
    tau_1_label = "τ = h"
    tau_2_label = "τ = h^2"
    tau_3_label = "τ = h^2/6"

    sig_dict = {
        0: "explicit euler",
        1 / 2: "crank nicolson",
        1: "implicit euler"
    }
    ####################################################################################################################
    # calculate
    ####################################################################################################################

    # structure
    # { "τ = h": {0: [[error_0 .... error_n-1]]} }
    errors: Dict[int, Dict[str, list]] = {i: {} for i in sigmas}
    hx_list = []
    for i, n_p1 in enumerate(np.linspace(start=n_p1_start, stop=n_p1_end, num=n_p1_steps, endpoint=True)):
        ################################################################################################################
        # setup
        n_p1 = int(n_p1)
        hx = (b - a) / (n_p1 - 1)
        hx_list.append(hx)
        x = (a, b, n_p1)

        # time discretization
        t_start = 0
        t_end = 1
        # default for explicit euler
        kmax = k(np.linspace(a, b, 1000, endpoint=True)).max()

        # dict for each tau according to table
        tau_dict = {
            tau_1_label: hx,
            tau_2_label: np.square(hx),
            tau_3_label: np.square(hx) / 6
        }

        ################################################################################################################
        # iterate
        for ii, (key, tau) in enumerate(tau_dict.items()):
            # calculating m and t for specific tau
            m = int(np.ceil((t_end - t_start) / tau) + 1)
            t = (t_start, t_end, m)

            # iterating over each sigma specified in sigmas
            for sig in sigmas:
                # skipping "first to columns" in sig = 0
                # "τ = h" and "τ = h^2" should be skipped for explicit euler
                if sig == 0 and key != tau_3_label:
                    continue

                # solve
                approx_result = fdm_ivbc_solver(space=x,
                                                time=t,
                                                k=k,
                                                q=q,
                                                f=f,
                                                mu_a=mu_a,
                                                mu_b=mu_b,
                                                phi=phi,
                                                sigma=sig)

                # extract values
                approx = approx_result[0]
                X = approx_result[1]
                T = approx_result[2]

                # calc solution
                xx, tt = np.meshgrid(X, T)
                solution = u(xx, tt)

                # calculate absolute error
                if key not in errors[sig]:
                    errors[sig][key] = []
                errors[sig][key].append(np.abs(solution - approx).max())

    # converting hx_list to np array
    hx_list = np.array(hx_list)
    ####################################################################################################################
    # fitting data with expected curves
    ####################################################################################################################
    # structure
    # { "τ = h": {0: ([fit_0 .... fit_n-1], "ax^2 + bx + c") } }
    fits: Dict[int, Dict[str, Tuple[list, str]]] = {i: {} for i in sigmas}
    if draw_fit:
        def linear(x, m, t):
            return m * x + t


        def polynom_2(x, a, b, c):
            return a * np.square(x) + b * x + c


        for _, (sigma, dictionary) in enumerate(errors.items()):

            # Iterating over every the different rules for T
            for r, (tau_rule, error_values) in enumerate(dictionary.items()):
                if sigma == 1 and tau_rule == tau_1_label:
                    param, _ = curve_fit(linear, hx_list, error_values)
                    print(f"{param[0]}x + {param[1]}")
                    fits[sigma][tau_rule] = (linear(hx_list, *param), f"{param[0]:.3e}x + {param[1]:.3e}")
                else:
                    param, _ = curve_fit(polynom_2, hx_list, error_values)
                    print(f"{param[0]}x^2 + {param[1]}x + {param[2]}")
                    fits[sigma][tau_rule] = (
                    polynom_2(hx_list, *param), f"{param[0]:.3e}x^2 + {param[1]:.3e}x + {param[2]:.3e}")

    ####################################################################################################################
    # plot
    ####################################################################################################################
    pprint(errors)

    for _, (sigma, dictionary) in enumerate(errors.items()):
        # creating a new figure/window for every procedure and window and figure title
        fig: Figure = plt.figure(f"{sig_dict[sigma]} (sigma: {sigma})")
        fig.suptitle(f"{sig_dict[sigma]}")
        # creating sub figures
        axs: Union[Axes, List[Axes]] = fig.subplots(len(dictionary), 1, sharex='col')

        # Setting x_label for the flor plot
        if isinstance(axs, Axes):
            axs.set_xlabel("h")
        else:
            axs[-1].set_xlabel("h")

        # Iterating over every the different rules for T
        for r, (tau_rule, error_values) in enumerate(dictionary.items()):
            if isinstance(axs, Axes):
                ax = axs
            else:
                ax = axs[r]
            ax.set_xlabel("h")
            ax.set_ylabel("max(abs(error))")
            ax.plot(hx_list, error_values, label=tau_rule)
            if draw_fit:
                fit, label = fits[sigma][tau_rule]
                ax.plot(hx_list, fit, color='red', linestyle='dashed', label=f"fit: {label}")
            ax.legend()
    plt.show()
