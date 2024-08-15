import pytest
import numpy as np
from .utils import has_numba


@pytest.mark.skipif(not has_numba(), reason="requires numba")
def test_monotonic_unique_counts():
    from pylabianca._numba import _monotonic_unique_counts

    values = np.array([2, 2, 2, 5, 5, 5, 5, 5, 8, 8,
                       9, 9, 9, 9, 9, 9, 9, 9, 10, 10,
                       10, 10])
    out = _monotonic_unique_counts(values)

    assert (out[0] == np.array([ 2,  5,  8,  9, 10], dtype='int64')).all()
    assert (out[1] == np.array([3, 5, 2, 8, 4], dtype='int64')).all()
