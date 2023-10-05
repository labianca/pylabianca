import numpy as np
import pylabianca as pln


def test_monotonic_unique_counts():
    values = np.array([2, 2, 2, 5, 5, 5, 5, 5, 8, 8,
                    9, 9, 9, 9, 9, 9, 9, 9, 10, 10,
                    10, 10])
    out = pln._numba._monotonic_unique_counts(values)

    assert (out[0] == np.array([ 2,  5,  8,  9, 10], dtype='int64')).all()
    assert (out[1] == np.array([3, 5, 2, 8, 4], dtype='int64')).all()
