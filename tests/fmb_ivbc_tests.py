import unittest
import numpy as np
from src.fdm_ivbc_solver import *


# noinspection DuplicatedCode
class FDMIVBCSolverTests(unittest.TestCase):
    m_p_1 = 20
    n_p_1 = 20
    t_start = 0
    t_end = 1

    a = 0
    b = 1

    t = (t_start, t_end, m_p_1)
    x = (a, b, n_p_1)

    sigma = 0

    def test_Ex_2_2_a(self):
        u = lambda x, t: np.sin(x) * np.cos(t)

        k = lambda x: 1
        q = lambda x: k(x)
        mu_a = lambda t: 0
        mu_b = lambda t: np.sin(1) * np.cos(t)

        phi = lambda x: np.sin(x)

        f = lambda x, t: np.sin(x) * (2 * np.cos(t) - np.sin(t))

        explicit = fdm_ivbc_solver(space=self.x,
                                   time=self.t,
                                   k=k,
                                   q=q,
                                   f=f,
                                   mu_a=mu_a,
                                   mu_b=mu_b,
                                   phi=phi,
                                   sigma=0)

        crank_nicolson = fdm_ivbc_solver(space=self.x,
                                         time=self.t,
                                         k=k,
                                         q=q,
                                         f=f,
                                         mu_a=mu_a,
                                         mu_b=mu_b,
                                         phi=phi,
                                         sigma=1 / 2)

        implicit = fdm_ivbc_solver(space=self.x,
                                   time=self.t,
                                   k=k,
                                   q=q,
                                   f=f,
                                   mu_a=mu_a,
                                   mu_b=mu_b,
                                   phi=phi,
                                   sigma=1)
        self.assertTrue(False)

    def test_Ex_2_2_b(self):
        u = lambda x, t: np.exp(-2 * t) * np.cos(np.pi * x)

        k = lambda x: 1 + 2 * np.power(x, 2)
        q = lambda x: x * (1 - x)

        mu_a = lambda t: np.exp(-2 * t)
        mu_b = lambda t: - np.exp(-2 * t)

        phi = lambda x: np.cos(np.pi * x)

        f = lambda x, t: (np.exp(-2 * t) *
                          (np.cos(np.pi * x) *
                           ((2 * np.power(np.pi, 2) - 1) *
                            np.power(x, 2) + x + np.power(np.pi, 2) - 2)
                           - np.sin(np.pi * x) * (4 * np.pi * x)))

        explicit = fdm_ivbc_solver(space=self.x,
                                   time=self.t,
                                   k=k,
                                   q=q,
                                   f=f,
                                   mu_a=mu_a,
                                   mu_b=mu_b,
                                   phi=phi,
                                   sigma=0)

        crank_nicolson = fdm_ivbc_solver(space=self.x,
                                         time=self.t,
                                         k=k,
                                         q=q,
                                         f=f,
                                         mu_a=mu_a,
                                         mu_b=mu_b,
                                         phi=phi,
                                         sigma=1 / 2)

        implicit = fdm_ivbc_solver(space=self.x,
                                   time=self.t,
                                   k=k,
                                   q=q,
                                   f=f,
                                   mu_a=mu_a,
                                   mu_b=mu_b,
                                   phi=phi,
                                   sigma=1)
        self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
