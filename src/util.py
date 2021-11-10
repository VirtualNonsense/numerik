import numpy as np
from numpy.typing import *
from pprint import pprint


def thomas_algorithm(a: ArrayLike,
                     b: ArrayLike,
                     c: ArrayLike,
                     d: ArrayLike) -> ArrayLike:
    # Dimension des GLS und Speicher für Loesung
    n = d.shape[0]
    x = np.zeros(shape=n)

    # Neue modifizierte Koeffizienten/Vektoren = Vorwaetrs-Durchlauf
    c_new = np.zeros(shape=n)
    d_new = np.zeros(shape=n)

    # Gauss-Elimination Vorwaerts
    c_new[0] = c[0] / b[0]
    d_new[0] = d[0] / b[0]
    for i in range(1, n - 1):
        q = b[i] - c_new[i - 1] * a[i - 1]
        c_new[i] = c[i] / q
        d_new[i] = (d[i] - d_new[i - 1] * a[i - 1]) / q

    q = b[n - 1] - c_new[n - 2] * a[n - 2]
    d_new[n - 1] = (d[n - 1] - d_new[n - 2] * a[n - 2]) / q

    # Rueckwaerts-Durchlauf
    x[-1] = d_new[-1]

    for i in range(n - 2, -1, -1):
        x[i] = d_new[i] - c_new[i] * x[i + 1]
    return x


if __name__ == '__main__':
    a_ref = np.array([-18, -18, -18, -32])
    b_ref = np.array([1, 33, 33, 33, 40])
    c_ref = np.array([0, -14, -14, -14])
    d_ref = np.array([3, 1, 1.5052, 2.0422, 48.4063])

    x_ref = np.array([3., 3.0096198, 3.16553239, 3.48458661, 3.99782679])

    x_ = thomas_algorithm(a_ref,
                          b_ref,
                          c_ref,
                          d_ref)
    pprint(x_ref)
    pprint(x_)
    pprint(x_ref - x_)


def calc_h(interval, n):
    return (interval[1] - interval[0]) / (n - 1)