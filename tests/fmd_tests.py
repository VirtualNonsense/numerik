import unittest
import numpy as np
from src.fdm_solver import *
from src.util import *


class FDMSolverTests(unittest.TestCase):
    n = 50

    @staticmethod
    def u(x: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
        return np.exp(-x) + np.exp(x) + 1

    def test_Ex_2_1_a(self):
        def k(x: float):
            return 1

        def f(x: float):
            return np.exp(x) - np.exp(-x) + 1

        interval = [0, 1]
        h = calc_h(interval, self.n)
        x = np.arange(start=interval[0], stop=interval[1], step=h)
        boundary = (
            DirichletBoundaryCondition(location=interval[0], mu=3),
            DirichletBoundaryCondition(location=interval[1], mu=np.exp(1) + 1 / np.exp(1) + 1),
        )

        solution = self.u(x)
        u = fdm_solver(k=k,
                       r=k,
                       q=k,
                       f=f,
                       h=h,
                       border=boundary,
                       interval=interval)
        epsilon = .01
        self.assertTrue(((np.abs(solution - u) <= epsilon).all()))
        # self.assertTrue(True)

    def test_Ex_2_1_b(self):
        def k(x: float):
            return 1

        def f(x: float):
            return np.exp(x) - np.exp(-x) + 1

        interval = [0, 1]
        h = calc_h(interval, self.n)
        x = np.arange(start=interval[0], stop=interval[1], step=h)

        boundary = (
            DirichletBoundaryCondition(location=interval[0], mu=3),
            RobinBoundaryCondition(location=interval[1], mu=2 * np.exp(1) + 1, kappa=1),
        )
        solution = self.u(x)
        u = fdm_solver(k=k,
                       r=k,
                       q=k,
                       f=f,
                       h=h,
                       border=boundary,
                       interval=interval)
        epsilon = .01
        self.assertTrue(((np.abs(solution - u) <= epsilon).all()))


class AMatrixTests(unittest.TestCase):
    def test_script_example(self):
        n = 16

        def k(x: float):
            return 1

        def q(x: float):
            return 0

        def f(x: float):
            return np.exp(x) - np.exp(-x) + 1

        interval = [0, 1]
        h = calc_h(interval, n)
        x = np.arange(start=interval[0],
                      stop=interval[1],
                      step=h)

        # generate A
        r = q
        N = x.shape[0]
        a, b, c = gen_A_vectors(x, k, r, q, N, h)
        a_ref = np.zeros(shape=15)
        a_ref[:] = -256
        b_ref = np.ones(shape=16)
        b_ref[:] = 512
        c_ref = np.zeros(shape=15)
        c_ref[:] = -256
        self.assertTrue((a[1:] == a_ref).all())
        self.assertTrue((b == b_ref).all())
        self.assertTrue((c[:-1] == c_ref).all())


if __name__ == '__main__':
    unittest.main()
