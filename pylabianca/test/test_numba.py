import numpy as np


def test_monotonic_unique_counts():
    from pylabianca._numba import _monotonic_unique_counts

    values = np.array([2, 2, 2, 5, 5, 5, 5, 5, 8, 8,
                       9, 9, 9, 9, 9, 9, 9, 9, 10, 10,
                       10, 10])
    out = _monotonic_unique_counts(values)

    assert (out[0] == np.array([ 2,  5,  8,  9, 10], dtype='int64')).all()
    assert (out[1] == np.array([3, 5, 2, 8, 4], dtype='int64')).all()


def test_numba_select_spikes():
    from pylabianca._numba import _select_spikes_numba

    def select_spikes(spikes, trials, tri_sel):
        msk = np.in1d(trials, tri_sel)
        return spikes[msk]

    spikes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    trials = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4])
    tri_sel = np.array([1, 3, 4])

    out = _select_spikes_numba(spikes, trials, tri_sel)
    expected = select_spikes(spikes, trials, tri_sel)
    assert (out == expected).all()
