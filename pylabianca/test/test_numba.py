import pytest
import numpy as np

import pylabianca as pln
from pylabianca.utils import has_numba


@pytest.mark.skipif(not has_numba(), reason="requires numba")
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


def test_trial_boundaries():
    from pylabianca.utils import has_numba
    from pylabianca.utils import _get_trial_boundaries

    if has_numba:
        from pylabianca._numba import (
            _get_trial_boundaries as _get_trial_boundaries_numba_high_level)
        from pylabianca._numba import _get_trial_boundaries_numba

        # TEST 1
        # ------
        n_tri = 200
        per_tri = (0, 35)

        trials = list()
        times = list()
        for tri in range(n_tri):
            n_spk = np.random.randint(*per_tri)
            this_trials = np.full(n_spk, tri)
            this_times = np.sort(np.random.rand(n_spk) * 2. - 0.5)
            trials.append(this_trials)
            times.append(this_times)

        trials = np.concatenate(trials)
        times = np.concatenate(times)
        spk = pln.SpikeEpochs([times], [trials])

        # get boundaries
        boundaries, tri = _get_trial_boundaries(spk, 0)
        boundaries2, tri2 = _get_trial_boundaries_numba(
            spk.trial[0], spk.n_trials)

        assert (boundaries == boundaries2).all()
        assert (tri == tri2).all()

        # TEST 2
        # ------
        trials = np.sort(np.random.randint(0, 13, size=253))
        times = np.random.rand(len(trials)) * 2. - 0.5
        spk = pln.SpikeEpochs([times], [trials])

        # get boundaries
        tri_bnd, tri_num = _get_trial_boundaries(spk, 0)
        tri_bnd_numba, tri_num_numba = (
            _get_trial_boundaries_numba_high_level(spk, 0)
        )
        assert (tri_num == tri_num_numba).all()
        assert (tri_bnd == tri_bnd_numba).all()


def test_find_first():
    from pylabianca._numba import _monotonic_find_first

    values = np.array([2, 5, 8, 9, 10], dtype='int64')
    idx = _monotonic_find_first(values, 9)
    assert idx == 3

    has_nine = False
    while not has_nine:
        values = np.random.randint(0, 100, size=120)
        has_nine = 9 in values

    idx_nb = _monotonic_find_first(values, 9)
    assert np.all(values[idx_nb] == 9)

    idx_np = np.where(values == 9)[0][0]
    assert idx_nb == idx_np


def test_trial_boundaries_numba():
    from pylabianca.utils import has_numba

    if has_numba:
        from pylabianca._numba import _get_trial_boundaries_numba
        n_tri = 200
        per_tri = (0, 35)
        trials = list()
        times = list()
        for tri in range(n_tri):
            n_spk = np.random.randint(*per_tri)
            this_trials = np.full(n_spk, tri)
            this_times = np.sort(np.random.rand(n_spk) * 2. - 0.5)
            trials.append(this_trials)
            times.append(this_times)

        trials = np.concatenate(trials)
        times = np.concatenate(times)
        spk = pln.SpikeEpochs([times], [trials])

        # get boundaries
        boundaries, tri = _get_trial_boundaries(spk, 0)
        boundaries2, tri2 = _get_trial_boundaries_numba(
            spk.trial[0], spk.n_trials)

        assert (boundaries == boundaries2).all()
        assert (tri == tri2).all()
