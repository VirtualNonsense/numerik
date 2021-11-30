import unittest
import numpy as np
from src.fdm_ivbc_solver import *


# noinspection DuplicatedCode
class FDMIVBCSolverTests(unittest.TestCase):
    m_explicit = 5
    n_explicit = 2
    m_cn = 80
    n_cn = 7
    m_implicit = 80
    n_implicit = 7
    t_start = 0
    t_end = 1

    a = 0
    b = 1

    t_explicit = (t_start, t_end, m_explicit)
    x_explicit = (a, b, n_explicit)
    t_cn = (t_start, t_end, m_cn)
    x_cn = (a, b, n_cn)
    t_implicit = (t_start, t_end, m_implicit)
    x_implicit = (a, b, n_implicit)

    sigma = 0

    epsilon_explicit = 5e-2
    epsilon_crank_nicolson = 5e-2
    epsilon_implicit = 5e-2

    def test_Ex_2_2_a(self):
        u = lambda x, t: np.sin(x) * np.cos(t)

        k = lambda x: 1
        q = lambda x: k(x)
        mu_a = lambda t: 0
        mu_b = lambda t: np.sin(1) * np.cos(t)

        phi = lambda x: np.sin(x)

        f = lambda x, t: np.sin(x) * (2 * np.cos(t) - np.sin(t))

        explicit = fdm_ivbc_solver(space=self.x_explicit,
                                   time=self.t_explicit,
                                   k=k,
                                   q=q,
                                   f=f,
                                   mu_a=mu_a,
                                   mu_b=mu_b,
                                   phi=phi,
                                   sigma=0)

        crank_nicolson = fdm_ivbc_solver(space=self.x_cn,
                                         time=self.t_cn,
                                         k=k,
                                         q=q,
                                         f=f,
                                         mu_a=mu_a,
                                         mu_b=mu_b,
                                         phi=phi,
                                         sigma=1 / 2)

        implicit = fdm_ivbc_solver(space=self.x_implicit,
                                   time=self.t_implicit,
                                   k=k,
                                   q=q,
                                   f=f,
                                   mu_a=mu_a,
                                   mu_b=mu_b,
                                   phi=phi,
                                   sigma=1)

        tmp = explicit
        xx, tt = np.meshgrid(tmp[1], tmp[2])
        diff_explicit = u(xx, tt) - explicit[0]
        tmp = crank_nicolson
        xx, tt = np.meshgrid(tmp[1], tmp[2])
        diff_crank_nicolson = u(xx, tt) - crank_nicolson[0]
        tmp = implicit
        xx, tt = np.meshgrid(tmp[1], tmp[2])
        diff_implicit = u(xx, tt) - implicit[0]
        self.assertTrue((np.abs(diff_explicit) < self.epsilon_explicit).all())
        self.assertTrue((np.abs(diff_crank_nicolson) < self.epsilon_crank_nicolson).all())
        self.assertTrue((np.abs(diff_implicit) < self.epsilon_implicit).all())

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


        explicit = fdm_ivbc_solver(space=self.x_explicit,
                                   time=self.t_explicit,
                                   k=k,
                                   q=q,
                                   f=f,
                                   mu_a=mu_a,
                                   mu_b=mu_b,
                                   phi=phi,
                                   sigma=0)

        crank_nicolson = fdm_ivbc_solver(space=self.x_cn,
                                         time=self.t_cn,
                                         k=k,
                                         q=q,
                                         f=f,
                                         mu_a=mu_a,
                                         mu_b=mu_b,
                                         phi=phi,
                                         sigma=1 / 2)

        implicit = fdm_ivbc_solver(space=self.x_implicit,
                                   time=self.t_implicit,
                                   k=k,
                                   q=q,
                                   f=f,
                                   mu_a=mu_a,
                                   mu_b=mu_b,
                                   phi=phi,
                                   sigma=1)

        tmp = explicit
        xx, tt = np.meshgrid(tmp[1], tmp[2])
        diff_explicit = u(xx, tt) - explicit[0]
        tmp = crank_nicolson
        xx, tt = np.meshgrid(tmp[1], tmp[2])
        diff_crank_nicolson = u(xx, tt) - crank_nicolson[0]
        tmp = implicit
        xx, tt = np.meshgrid(tmp[1], tmp[2])
        diff_implicit = u(xx, tt) - implicit[0]
        self.assertTrue((np.abs(diff_explicit) < self.epsilon_explicit).all())
        self.assertTrue((np.abs(diff_crank_nicolson) < self.epsilon_crank_nicolson).all())
        self.assertTrue((np.abs(diff_implicit) < self.epsilon_implicit).all())



if __name__ == '__main__':
    unittest.main()
