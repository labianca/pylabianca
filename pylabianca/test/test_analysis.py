import os.path as op
import time
import numpy as np
import pandas as pd
import xarray as xr

import pytest
import pylabianca as pln
from pylabianca.testing import gen_random_xarr, ft_data
from pylabianca.utils import (
    create_random_spikes, get_fieldtrip_data, get_data_path)


get_fieldtrip_data()
data_dir = get_data_path()


def test_spike_centered_windows():
    # simple case - one spike per trial
    n_trials, n_channels, n_times = 10, 5, 150
    data = np.random.rand(n_trials, n_channels, n_times)
    time = np.linspace(-0.5, 1., num=n_times)

    spike_indices = np.random.randint(0, n_times, size=n_trials)
    spike_times = time[spike_indices]
    spike_trials = np.arange(n_trials)

    spk = pln.SpikeEpochs([spike_times], [spike_trials])

    # raises ValueError
    msg = 'has to be an array of time points'
    with pytest.raises(ValueError, match=msg):
        pln.analysis.spike_centered_windows(spk, data)

    # works with time array
    spk_cent = pln.analysis.spike_centered_windows(
        spk, data, time=time, winlen=0.01)

    correct = data[np.arange(n_trials), :, spike_indices]
    assert spk_cent.shape == (10, 5, 1)
    assert (spk_cent.data[:, :, 0] == correct).all()

    # should get the same with xarray
    xarr = xr.DataArray(
        data, dims=['trial', 'channel', 'time'],
        coords={'time': time}
    )

    spk_cent2 = pln.analysis.spike_centered_windows(
        spk, xarr, winlen=0.01)

    assert (spk_cent == spk_cent2).all().item()

    # make sure that using pln.utils version raises warning,
    # but gives the same result
    with pytest.warns(DeprecationWarning):
        spk_cent3 = pln.utils.spike_centered_windows(
            spk, xarr, winlen=0.01)

    assert (spk_cent2 == spk_cent3).all().item()

    # but not when we specify incorrect time dim
    msg = 'Coordinate named "emit" not found'
    with pytest.raises(ValueError, match=msg):
        pln.analysis.spike_centered_windows(
            spk, xarr, time='emit', winlen=0.01)

    # unless that is the correct name
    xarr = xarr.rename({'time': 'emit'})
    spk_cent4 = pln.analysis.spike_centered_windows(
        spk, xarr, time='emit', winlen=0.01)
    assert (spk_cent2 == spk_cent4).all().item()

    # but then we can't leave time arg to its default value
    msg = ('When ``time=None`` the ``arr`` xarray has to contain '
           'a coordinate named "time".')
    with pytest.raises(ValueError, match=msg):
        pln.analysis.spike_centered_windows(spk, xarr, winlen=0.01)

    # time argument has to be correct
    msg = ("When ``arr`` is an xarray ``time`` input argument has to "
           "be either ``None`` or a string, got <class 'list'>.")
    with pytest.raises(ValueError, match=msg):
        pln.analysis.spike_centered_windows(
            spk, xarr, time=list('emit'), winlen=0.01)


def test_spike_centered_windows_against_fieldtrip(ft_data):
    import mne

    # read raw signal and events
    fname = 'p029_sort_final_01.nex'
    raw = pln.io.read_signal_plexon_nex(op.join(data_dir, fname))
    events = pln.io.read_events_plexon_nex(
        op.join(data_dir, fname), format='mne')

    # select cells with waveforms and epoch
    has_waveforms = [wave is not None for wave in ft_data.waveform]
    ft_data.pick_cells(has_waveforms)
    spk_epochs = ft_data.epoch(events, event_id=10030, tmin=0., tmax=2.75)

    # epoch raw cnt data
    downsample = ft_data.sfreq / raw.info['sfreq']
    events_smp = events.copy()
    samples = np.round(events[:, 0] / downsample).astype(int)
    events_smp[:, 0] = samples - 1
    lfp_epochs = mne.Epochs(
        raw, events_smp, event_id=10030, tmin=0., tmax=2.75,
        baseline=None, preload=True
    )

    metadata = pd.read_csv(op.join(data_dir, 'monkey_stim.csv'))
    metadata = metadata.query('has_stimon == True')
    spk_epochs.metadata = metadata
    lfp_epochs.metadata = metadata

    # select only correct trials
    spk_epochs = spk_epochs['correct == True']
    lfp_epochs = lfp_epochs['correct == True']

    channels = ['AD01', 'AD02', 'AD03', 'AD04']
    lfp_poststim = (
        lfp_epochs.copy()
        .crop(tmin=0.3)
        .pick(channels)
    )
    spk_poststim = spk_epochs.copy().crop(tmin=0.3)

    lfp_trig = pln.analysis.spike_centered_windows(
        spk_poststim, lfp_poststim,
        pick='sig002a_wf', winlen=0.4)
    lfp_trig -= lfp_trig.mean(dim='time')

    psd, freq = mne.time_frequency.psd_array_multitaper(
        lfp_trig.sel(channel='AD03').mean(dim='spike'),
        sfreq=raw.info['sfreq'], bandwidth=7, fmin=10, fmax=100
    )

    # now on shuffled data
    new_spk = pln.analysis.shuffle_trials(spk_poststim)
    lfp_trig_shuffled = pln.analysis.spike_centered_windows(
        new_spk, lfp_poststim,
        pick='sig002a_wf', winlen=0.4)
    lfp_trig_shuffled -= lfp_trig_shuffled.mean(dim='time')

    psd_shuffled, freq = mne.time_frequency.psd_array_multitaper(
        lfp_trig_shuffled.sel(channel='AD03').mean(dim='spike'),
        raw.info['sfreq'], bandwidth=7, fmin=10, fmax=100
    )

    # make sure that real psd has higher power in 50 - 60 Hz range
    freq_mask = (freq >= 50) & (freq <= 60)
    assert psd[freq_mask].mean() > psd_shuffled[freq_mask].mean()


def test_xarr_dct_conversion():
    from string import ascii_lowercase
    import xarray as xr

    def compare_dicts(x_dct1, x_dct2):
        keys1 = list(x_dct1.keys())
        keys2 = list(x_dct2.keys())
        assert keys1 == keys2

        for key in keys1:
            assert (x_dct1[key].data == x_dct2[key].data).all()
            coord_list = list(x_dct1[key].coords)
            for coord in coord_list:
                assert (x_dct1[key].coords[coord].values
                        == x_dct2[key].coords[coord].values).all()

    n_cells1, n_cells2, n_trials, n_times = 10, 15, 20, 100
    xarr1 = gen_random_xarr(n_cells1, n_trials, n_times)
    xarr2 = gen_random_xarr(n_cells2, n_trials, n_times)

    # add load information
    load = np.concatenate([np.ones(10), np.ones(10) * 2])
    np.random.shuffle(load)
    xarr1 = xarr1.assign_coords({'load': ('trial', load)})
    load2 = load.copy()
    np.random.shuffle(load2)
    xarr2 = xarr2.assign_coords({'load': ('trial', load2)})

    x_dct1 = {'sub-A01': xarr1, 'sub-A02': xarr2}
    xarr = pln.dict_to_xarray(x_dct1)
    x_dct2 = pln.xarray_to_dict(xarr, ensure_correct_reduction=False)
    compare_dicts(x_dct1, x_dct2)

    # make sure we can do the same via pln.utils,
    # but with a deprecation warning
    with pytest.warns(DeprecationWarning):
        xarr3 = pln.utils.dict_to_xarray(x_dct1)
        x_dct3 = pln.utils.xarray_to_dict(xarr3)

    compare_dicts(x_dct2, x_dct3)

    # test with non-sorted keys - this previously failed
    # because xarray sorts during groupby operation used in xarray_to_dict
    x_dct1 = {'C03': xarr1, 'A02': xarr2, 'W05': xarr1.copy()}
    xarr = pln.dict_to_xarray(x_dct1)

    t_start = time.time()
    x_dct2 = pln.xarray_to_dict(xarr, ensure_correct_reduction=True)
    t_taken = time.time() - t_start
    assert t_taken < 0.1

    compare_dicts(x_dct1, x_dct2)

    xarr_2 = pln.dict_to_xarray(x_dct2)
    assert (xarr == xarr_2).all().item()

    # for some reason performing a query will not work without
    # assigning a name to the DataArray
    # we test this here to be warned when this behavior is changed in xarray
    with pytest.raises(ValueError, match='without providing an explicit name'):
        xarr.name = None
        xarr.query(cell='sub == "W05"')

    # selecting by condition
    xarr1 = xarr1.assign_coords(
        cnd1=('trial', np.random.choice(['A', 'B'], n_trials)))
    xarr2 = xarr2.assign_coords(
        cnd2=('trial', np.random.choice(['A', 'B'], n_trials)))

    x_dct1 = {'sub-A01': xarr1, 'sub-A02': xarr2}
    xarr = pln.dict_to_xarray(x_dct1, select='load == 1')
    n_tri = xarr.shape[0]

    n_per_condition = 10
    assert 'cnd1' not in xarr.coords
    assert 'cnd2' not in xarr.coords
    assert xarr.shape[0] == xarr1.shape[0] + xarr2.shape[0]
    assert (xarr.trial.data == np.arange(n_per_condition)).all()


def test_extract_data_and_aggregate():
    '''Test extract_data and some basic dict -> xarray operations.'''

    # create dict of SpikeEpochs
    keys = ['sub-a01', 'sub-a02', 'sub-a04']
    n_trials = 40
    n_cells = [10, 12, 8]
    anat_region = ['HIP', 'AMY', 'ACC']
    conditions = ['A', 'B', 'C']

    spk_dict = dict()
    for this_key, this_n_cells in zip(keys, n_cells):
        cnd = np.random.choice(conditions, size=n_trials)
        metadata = pd.DataFrame({'condition': cnd})

        anat = np.sort(
            np.random.choice(anat_region, size=this_n_cells)
        )
        cellinfo = pd.DataFrame({'anat': anat})

        spk = pln.utils.create_random_spikes(
            n_cells=this_n_cells, n_trials=n_trials, n_spikes=(25, 65),
            metadata=metadata, cellinfo=cellinfo
        )
        spk_dict[this_key] = spk

    # get firing rates
    frates = {sub: spk_dict[sub].spike_rate() for sub in spk_dict.keys()}

    # assert that all frates have the same time
    time = frates[keys[0]].time.values.copy()
    assert (time == frates[keys[1]].time.values).all()
    assert (time == frates[keys[2]].time.values).all()

    n_sub = len(keys)
    df_sel = pd.DataFrame({'sub': keys, 'anat': ['HIP'] * n_sub})
    frates_sel, row_indices = pln.analysis.extract_data(
        frates, df_sel, sub_col='sub', df2xarr={'anat': 'anat'}
    )

    # turn the dictionaries to xarrays
    frates_sel_x = pln.dict_to_xarray(frates_sel)
    frates_x = pln.dict_to_xarray(frates)

    # assert that selection via extract_data and xarray query is the same
    frates_sel2_x = frates_x.query(cell='anat == "HIP"')
    assert (frates_sel_x.data == frates_sel2_x.data).all()
    assert (frates_sel_x.condition.data == frates_sel2_x.condition.data).all()
    assert (frates_sel_x.sub.data == frates_sel2_x.sub.data).all()
    assert (frates_sel_x.anat.data == frates_sel2_x.anat.data).all()

    # in this particular case row_indices match subjects
    sub_should_have = np.array(keys)[row_indices]
    sub_has = frates_sel_x.sub.data
    assert (sub_has == sub_should_have).all()

    # check that extract_data on concat xarrays also gives the same result
    frates_sel2_x, row_indices2 = pln.analysis.extract_data(
        frates_x, df_sel, sub_col='sub', df2xarr={'anat': 'anat'}
    )
    assert (row_indices == row_indices2).all()
    assert (frates_sel_x.data == frates_sel2_x.data).all()
    assert (frates_sel_x.condition.data == frates_sel2_x.condition.data).all()
    assert (frates_sel_x.sub.data == frates_sel2_x.sub.data).all()
    assert (frates_sel_x.anat.data == frates_sel2_x.anat.data).all()

    # aggregation does not work on concatenated xarrays
    pattern = 'groupby coordinate cannot be cell x trial'
    with pytest.raises(ValueError, match=pattern):
        pln.aggregate(frates_sel_x, groupby='condition')

    # but should work on a single xarray
    frates_agg_one = pln.aggregate(
        frates_sel['sub-a01'], groupby='condition')
    frates_agg = pln.aggregate(
        frates_sel, groupby='condition')
    n_cells_one = frates_sel['sub-a01'].shape[0]
    assert frates_agg_one.shape[0] == n_cells_one
    assert (frates_agg_one.data == frates_agg[:n_cells_one].data).all()


def test_aggregate_per_cell():
    '''Test aggregation per cell.'''
    n_cells = 10
    n_trials = 50
    arr = gen_random_xarr(n_cells, n_trials, 120, per_cell_coord=True)
    arr_agg = pln.aggregate(arr, groupby='preferred', per_cell=True)

    # check that the aggregation per cell is correct:
    cell_indices = np.random.randint(0, n_cells, size=min(5, n_trials))
    cell_indices = np.unique(cell_indices)

    for cell_idx in cell_indices:
        frate_cell = arr_agg.isel(cell=cell_idx)

        groups = arr.preferred.data[cell_idx]
        for group_id in np.unique(groups):
            agg_gave = frate_cell.sel(preferred=group_id)
            mask = groups == group_id
            avg = arr.data[cell_idx, mask].mean(axis=0)
            assert np.allclose(avg, agg_gave.data)


def test_zscore_xarray():
    # create random xarray
    n_trials, n_cells, n_times = 50, 10, 100
    time = np.linspace(-0.5, 1.5, num=n_times)
    cell_names = ['cell_{}'.format(i) for i in range(n_cells)]

    xarr = xr.DataArray(np.random.rand(n_cells, n_trials, n_times),
                        dims=['cell', 'trial', 'time'],
                        coords={'cell': cell_names,
                                'trial': np.arange(n_trials),
                                'time': time})

    # zscore
    xarr_z = pln.analysis.zscore_xarray(xarr)

    # check that mean is zero and std is 1
    assert np.allclose(xarr_z.mean(dim=['trial', 'time']).data, 0, atol=1e-6)
    assert np.allclose(xarr_z.std(dim=['trial', 'time']).data, 1, atol=1e-6)

    # check that z-scored data is the same as z-scored data from numpy
    xarr_np = xarr.data
    xarr_np_z = (
        (xarr_np - xarr_np.mean(axis=(1, 2), keepdims=True))
         / xarr_np.std(axis=(1, 2), keepdims=True)
    )
    assert np.allclose(xarr_z.data, xarr_np_z, atol=1e-6)

    # test baseline argument
    xarr_z2 = pln.analysis.zscore_xarray(xarr, baseline=(-0.5, 0.))
    time_idx = (time >= -0.5) & (time < 0.)
    baseline_part = xarr_z2[:, :, time_idx]
    assert np.allclose(baseline_part.mean(dim=['trial', 'time']).data,
                       0., atol=1e-6)
    assert np.allclose(baseline_part.std(dim=['trial', 'time']).data,
                       1., atol=1e-6)

    xarr_np_z2 = (
        (xarr_np - xarr_np[:, :, time_idx].mean(axis=(1, 2), keepdims=True))
         / xarr_np[:, :, time_idx].std(axis=(1, 2), keepdims=True)
    )
    assert np.allclose(xarr_z2.data, xarr_np_z2, atol=1e-6)
