import unittest
import numpy as np
from src.fdm_solver import *


def u(x: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
    return np.exp(-x) + np.exp(x) + 1


class AMatrixTests(unittest.TestCase):
    def script_example(self):
        n = 16

        def k(x: float):
            return 1

        def q(x: float):
            return 0

        def f(x: float):
            return np.exp(x) - np.exp(-x) + 1

        interval = [0, 1]
        h = (interval[1] - interval[0]) / n
        x = np.arange(start=interval[0],
                      stop=interval[1] + h,  # + h to include interval border
                      step=h)

        # generate A
        r = q
        N = x.shape[0] - 1
        A = gen_A_matrix(x, k, r, q, N, h)
        A_ref = [[512, -256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.],
                 [-256, 512, -256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.],
                 [0, -256, 512, -256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.],
                 [0, 0, -256, 512, -256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.],
                 [0, 0, 0, -256, 512, -256, 0, 0, 0, 0, 0, 0, 0, 0, 0.],
                 [0, 0, 0, 0, -256, 512, -256, 0, 0, 0, 0, 0, 0, 0, 0.],
                 [0, 0, 0, 0, 0, -256, 512, -256, 0, 0, 0, 0, 0, 0, 0.],
                 [0, 0, 0, 0, 0, 0, -256, 512, -256, 0, 0, 0, 0, 0, 0.],
                 [0, 0, 0, 0, 0, 0, 0, -256, 512, -256, 0, 0, 0, 0, 0.],
                 [0, 0, 0, 0, 0, 0, 0, 0, -256, 512, -256, 0, 0, 0, 0.],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, -256, 512, -256, 0, 0, 0.],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -256, 512, -256, 0, 0.],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -256, 512, -256, 0.],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -256, 512, -256.],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -256, 512.]]
        self.assertTrue((A_ref == A).all())


if __name__ == '__main__':
    unittest.main()
