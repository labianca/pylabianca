import os.path as op
import numpy as np
import pandas as pd
import pytest

import pylabianca as pln
from pylabianca.utils import (download_test_data, get_data_path,
                              has_elephant, create_random_spikes)
from pylabianca.testing import ft_data, spk_epochs


download_test_data()
data_dir = get_data_path()


def test_input_validation():
    tri = 'abc'
    times = [np.random.random(len(tri[0]) + 3),
             np.random.random(len(tri[1]) + 2)]
    times = [t.tolist() for t in times]

    msg = 'Both time and trial have to be lists or object '
    with pytest.raises(ValueError, match=msg):
        pln.SpikeEpochs(times, tri)

    tri = [[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]

    msg = 'All elements of time list must be numpy arrays.'
    with pytest.raises(ValueError, match=msg):
        pln.SpikeEpochs(times, tri)

    tri = [np.array(t) for t in tri]
    with pytest.raises(ValueError, match=msg):
        pln.SpikeEpochs(times, tri)

    times = [np.array(t) for t in times]
    times2 = times.copy()
    times2.append(np.array([0.1, 0.2, 0.3]))

    msg = 'Length of time and trial lists must be the same.'
    with pytest.raises(ValueError, match=msg):
        pln.SpikeEpochs(times2, tri)

    # adding cellinfo to Spikes
    spk = create_random_spikes(n_cells=4)

    cellinfo = pd.DataFrame({'name': list('abcd'),
                            'channel': np.random.randint(20, 120, size=4),
                            'cluster': np.random.randint(100, 1_000, size=4)})

    # make sure we get error if the cellinfo df is too short
    msg = 'Number of rows in cellinfo has to be equal'
    with pytest.raises(ValueError, match=msg):
        spk.cellinfo = cellinfo.iloc[:2, :]

    # if the indexing does not match cell_names indices - warning and
    # re-indexing
    cellinfo2 = cellinfo.iloc[[0, 2, 3, 1], :]
    msg = 'cellinfo index does not match cell indices'
    with pytest.warns(UserWarning, match=msg):
        spk.cellinfo = cellinfo2

    assert not (spk.cellinfo.index == cellinfo2.index).all()


def create_fake_spikes():
    times = [[-0.3, -0.1, 0.025, 0.11, 0.22, 0.25, 0.4,
              -0.08, 0.12, 0.14, 0.19, 0.23, 0.32],
             [-0.22, -0.13, -0.03, 0.08, 0.16, 0.33, -0.2,
              -0.08, 0.035, 0.148, 0.32]]
    times = [np.array(t) for t in times]

    trials = [[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]
    trials = [np.array(t) for t in trials]

    spk = pln.SpikeEpochs(times, trials)
    return spk


def test_crop():
    spk_orig = create_fake_spikes()
    spk = spk_orig.copy().crop(tmin=0.1)

    assert (spk.time[0] == np.array(
        [0.11, 0.22, 0.25, 0.4, 0.12, 0.14, 0.19, 0.23, 0.32])).all()
    assert (spk.time[1] == np.array([0.16, 0.33, 0.148, 0.32])).all()
    assert (spk.trial[0] == np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])).all()
    assert (spk.trial[1] == np.array([0, 0, 1, 1])).all()

    spk = spk_orig.copy().crop(tmax=0.1)

    assert (spk.time[0] == np.array(
        [-0.3, -0.1, 0.025, -0.08])).all()
    assert (spk.time[1] == np.array(
        [-0.22, -0.13, -0.03, 0.08, -0.2, -0.08, 0.035])).all()
    assert (spk.trial[0] == np.array([0, 0, 0, 1])).all()
    assert (spk.trial[1] == np.array([0, 0, 0, 0, 1, 1, 1])).all()

    with pytest.raises(TypeError, match="tmin and/or tmax"):
        spk_orig.copy().crop(tmin=None, tmax=None)


# add .n_trials (if present in SpikeEpochs)
def test_num():
    for n_cells in [3, 5, 10]:
        spk = create_random_spikes(n_cells=n_cells)
        assert spk.n_units() == n_cells

    for n_tri in [5, 8, 12]:
        spk = create_random_spikes(n_trials=n_tri)
        assert len(spk) == n_tri

    for n_spk_per_tri in [10, 15, 23]:
        spk = create_random_spikes(
            n_cells=1, n_trials=10, n_spikes=n_spk_per_tri)
        assert spk.n_spikes()[0] == n_spk_per_tri * 10
        assert (spk.n_spikes(per_epoch=True) == n_spk_per_tri).all()


def test_concatenate():
    args = dict(n_trials=False, sfreq=10_000)
    spk1 = create_random_spikes(n_cells=2, n_spikes=(5, 11), **args)
    spk2 = create_random_spikes(n_cells=2, n_spikes=(5, 11), **args)
    spk3 = pln.spikes.concatenate_spikes([spk1, spk2], sort=False)

    assert len(spk3) == len(spk1) + len(spk2)
    assert (spk3.timestamps[2] == spk2.timestamps[0]).all()
    assert (spk3.timestamps[3] == spk2.timestamps[1]).all()


def test_repr():
    # SpikeEpochs repr
    spk = create_fake_spikes()
    should_be = '<SpikeEpochs, 2 epochs, 2 cells, 12.0 spikes/cell on average>'
    assert str(spk) == should_be

    # another test for SpikeEpochs
    n_cells, n_trials, n_spikes = 5, 4, 23
    spk = create_random_spikes(
        n_cells=n_cells, n_trials=n_trials, n_spikes=n_spikes)
    expected_str = (f'<SpikeEpochs, {n_trials} epochs, {n_cells} cells, '
                    f'{n_spikes * n_trials}.0 spikes/cell on average>')
    assert str(spk) == expected_str

    # Spikes repr
    n_cells, n_trials, n_spikes = 23, 0, 100
    spk = create_random_spikes(
        n_cells=n_cells, n_trials=n_trials, n_spikes=n_spikes)
    expected_str = (f'<Spikes, {n_cells} cells, '
                    f'{n_spikes}.0 spikes/cell on average>')
    assert str(spk) == expected_str


def test_pick_cells():
    spk = create_fake_spikes()
    assert len(spk.time) == 2
    spk.pick_cells('cell001')
    assert len(spk.time) == 1
    assert spk.time[0][0] == -0.22

    spk = create_random_spikes(n_cells=5)
    spk.cell_names = np.array(list("ABCDE"))

    spk2 = spk.copy().pick_cells(['A', 'C', 'E'])
    spk3 = spk.copy().pick_cells(np.array([0, 2, 4]))
    spk4 = spk.copy().pick_cells(np.array([True, False, True, False, True]))

    for cell_idx in range(spk2.n_units()):
        assert spk2.cell_names[cell_idx] == spk3.cell_names[cell_idx]
        assert spk2.cell_names[cell_idx] == spk4.cell_names[cell_idx]

        assert (spk2.time[cell_idx] == spk3.time[cell_idx]).all()
        assert (spk2.time[cell_idx] == spk4.time[cell_idx]).all()

        assert (spk2.trial[cell_idx] == spk3.trial[cell_idx]).all()
        assert (spk2.trial[cell_idx] == spk4.trial[cell_idx]).all()

    # picks=None returns the same object
    spk5 = spk.pick_cells(picks=None)
    assert id(spk) == id(spk5)

    # .drop_cells() works as expected
    spk6 = spk.copy().drop_cells(['A', 'C', 'E'])
    assert spk6.n_units() == 2
    assert (spk6.cell_names == np.array(['B', 'D'])).all()

    spk7 = spk.copy().pick_cells(['B', 'D'])
    assert (spk6.time[0] == spk7.time[0]).all()
    assert (spk6.time[1] == spk7.time[1]).all()


def test_pick_trials():
    spk = create_random_spikes(n_cells=3, n_trials=10)

    tri = [1, 3, 8]
    for repr in ['list', 'array', 'bool']:
        if repr == 'list':
            spk_sel = spk[tri]
        elif repr == 'array':
            spk_sel = spk[np.array(tri)]
        else:
            msk = np.zeros(10, dtype=bool)
            msk[tri] = True
            spk_sel = spk[msk]

        # make sure we have 3 trials now
        assert spk_sel.n_trials == 3

        # make sure trial numbers have been renumbered
        assert all([(tri < 3).all() for tri in spk_sel.trial])

        # make sure spike times are the same
        for trial_idx, trial in enumerate(tri):
            for unit_idx in range(spk.n_units()):
                msk1 = spk.trial[unit_idx] == trial
                msk2 = spk_sel.trial[unit_idx] == trial_idx
                assert (spk.time[unit_idx][msk1]
                        == spk_sel.time[unit_idx][msk2]).all()


def test_pick_cells_cellinfo_query():
    from copy import deepcopy
    cellinfo = pd.DataFrame({'cell_idx': [10, 15, 20, 25],
                             'letter': list('abcd')})
    spk = create_random_spikes(cellinfo=cellinfo)

    spk2 = deepcopy(spk)
    spk2.pick_cells(query='cell_idx > 18')
    assert len(spk2.time) == 2
    assert spk2.cellinfo.shape[0] == 2
    assert (spk2.cellinfo.letter.values == np.array(['c', 'd'])).all()

    spk3 = deepcopy(spk)
    spk3.pick_cells(query="letter in ['a', 'c']")
    assert len(spk3.time) == 2
    assert len(spk3.time[1]) == len(spk.time[2])
    assert (spk3.cellinfo.cell_idx.values == [10, 20]).all()


# TODO: move to io tests?
def test_to_raw():
    times = [np.array([-0.3, -0.28, -0.26, 0.15, 0.18, 0.2]),
             np.array([-0.045, 0.023, -0.1, 0.13])]
    trials = [np.array([0, 0, 0, 0, 0, 0]), np.array([0, 0, 1, 1])]
    spk = pln.SpikeEpochs(times, trials)

    spk_tm, spk_raw = pln.spikes._spikes_to_raw(spk, sfreq=10)
    print(spk_raw)
    good_tms = np.arange(-0.3, 0.21, step=0.1)
    good_raw = np.array(
        [[[3, 0, 0, 0, 1, 2], [0, 0, 0, 2, 0, 0]],
         [[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0]]]
    )
    assert (spk_tm == good_tms).all()
    assert (spk_raw == good_raw).all()


def test_apply():
    spk = create_random_spikes(n_cells=4, n_trials=23)
    avg = spk.apply(np.mean)
    test_idx = [(0, 5), (1, 18), (3, 22)]

    for cell_idx, tri_idx in test_idx:
        msk = spk.trial[cell_idx] == tri_idx
        assert avg[cell_idx, tri_idx].item() == np.mean(spk.time[cell_idx][msk])


def test_epoching_vs_fieldtrip(spk_epochs):

    # read fieldtrip results
    fname = 'ft_spk_epoched.mat'
    spk_epo_test_ft = pln.io.read_fieldtrip(
        op.join(data_dir, fname), kind='epochs', data_name='spikeTrials_test')

    # check that the results are the same
    for ch_idx in range(2):
        assert (spk_epo_test_ft.timestamps[ch_idx]
                == spk_epochs.timestamps[ch_idx]).all()

        assert (spk_epo_test_ft.trial[ch_idx]
                == spk_epochs.trial[ch_idx]).all()

        np.testing.assert_almost_equal(
            spk_epochs.time[ch_idx], spk_epo_test_ft.time[ch_idx])


def test_metadata():
    spk = create_random_spikes()

    # good metadata and selection by condition
    df = pd.DataFrame({'cond': ['A'] * 15 + ['B'] * 10})
    spk.metadata = df

    spk2 = spk['cond == "B"']
    for cell_idx in range(4):
        first_idx = np.where(spk.trial[cell_idx] == 15)[0][0]
        assert (spk2.time[cell_idx] == spk.time[cell_idx][first_idx:]).all()
        assert (spk2.trial[cell_idx] + 15
                == spk.trial[cell_idx][first_idx:]).all()

def test_sort():
    spk = create_random_spikes(n_cells=6, n_trials=0, n_spikes=(50, 120))

    # no .cellinfo attribute
    with pytest.raises(ValueError, match=".cellinfo attribute has to contain"):
        spk.sort()

    # now with a dataframe
    cellinfo = pd.DataFrame({'name': list('abcdef'),
                            'channel': np.random.randint(20, 120, size=6),
                            'cluster': np.random.randint(100, 1_000, size=6)})
    spk.cellinfo = cellinfo.copy()

    spk_srt = spk.sort(inplace=False)

    # assert that .cell_names have different order
    assert not (spk.cell_names == spk_srt.cell_names).all()

    # see how 'name' changed and infer reordering
    sorting = np.argsort(spk_srt.cellinfo.name.values)
    for orig_idx in range(6):
        # see if .cell_names, timestamps agree with reordering
        srt_idx = sorting[orig_idx]
        assert spk_srt.cell_names[srt_idx] == spk.cell_names[orig_idx]
        assert (spk_srt.timestamps[srt_idx] == spk.timestamps[orig_idx]).all()

    # original object is changed with inplace=True
    spk_srt = spk.sort(inplace=True)
    assert id(spk) == id(spk_srt)
    assert (spk.cell_names == spk_srt.cell_names).all()


def test_merge():
    spk = create_random_spikes(n_cells=5, n_trials=0, n_spikes=(50, 120))
    spk_m1 = spk.copy().merge([0, 2])

    assert len(spk) > len(spk_m1)
    assert spk.cell_names[0] == spk_m1.cell_names[0]
    assert not (spk.cell_names[2] == spk_m1.cell_names[2])

    n_spk = spk.n_spikes()
    mrg = np.sort(np.concatenate([spk.timestamps[0], spk.timestamps[2]]))
    assert (n_spk[0] + n_spk[2]) == spk_m1.n_spikes()[0]
    assert (spk_m1.timestamps[0] == mrg).all()

    # make sure it works also with cellinfo dataframe and waveforms
    cellinfo = pd.DataFrame(
        {'name': list('abcd'), 'channel': np.random.randint(20, 120, size=4),
         'cluster': np.random.randint(100, 1_000, size=4)}
    )
    spk_m1.cellinfo = cellinfo.copy()

    n_smp = 32
    n_spk_m1 = spk_m1.n_spikes()
    spk_m1.waveform = [np.random.rand(n_sp, n_smp) for n_sp in n_spk_m1]

    spk_m2 = spk_m1.copy().merge([1, 3])
    assert (spk_m2.cellinfo == spk_m1.cellinfo[:3]).all().all()

    n_spk_m2 = spk_m2.n_spikes()
    assert (n_spk_m2[1] == (n_spk_m1[1] + n_spk_m1[3])
            == spk_m2.waveform[1].shape[0])


# TODO: MOVE to viz tests
def test_plot_waveform():
    spk = create_random_spikes(n_cells=2, n_trials=0, n_spikes=(50, 120))

    n_smp = 6
    n_spk = spk.n_spikes()
    wfrm = [np.random.randn(n, n_smp) * 0.2 for n in n_spk]
    shape = np.array([0, 1, 5, -3, -1, 0.5])
    wfrm[0] += shape[None, :]
    spk.waveform = wfrm

    # currently just a smoke test for plotting waveforms
    spk.plot_waveform(0)
    spk.plot_waveform(0, upsample=20)


# TODO: MOVE to io tests
def test_to_neo_and_to_spiketools():
    spk = create_random_spikes(n_cells=2, n_trials=3, n_spikes=(50, 120))

    # .to_neo, join='concat'
    spk_neo = spk.to_neo(0, join='concat', sep_time=0.1)
    spk_lst = spk.to_spiketools(picks=0)

    n_spk = spk.n_spikes(per_epoch=True)
    spk_idx = np.concatenate([
        np.zeros((2, 1)), np.cumsum(n_spk, axis=1)],
        axis=1
    ).astype(int)
    epoch_time = np.diff(spk.time_limits)

    for idx in range(spk_idx.shape[1] - 1):
        np.testing.assert_almost_equal(
            spk_neo[spk_idx[0, idx]:spk_idx[0, idx + 1]].magnitude,
            spk_lst[idx] + 0.1 * idx + epoch_time * idx
        )

    # .to_neo, join='pool'
    spk_neo = spk.to_neo(0, join='pool')

    assert n_spk.sum(axis=1)[0] == len(spk_neo)
    assert (spk_neo.magnitude == np.sort(np.concatenate(spk_lst))).all()
