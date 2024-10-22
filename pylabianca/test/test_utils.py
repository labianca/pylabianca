import time

import numpy as np
import pandas as pd
import xarray as xr
import pytest

import pylabianca as pln
from pylabianca.testing import gen_random_xarr
from pylabianca.utils import (_get_trial_boundaries, find_cells,
                              create_random_spikes, _inherit_metadata)


def test_trial_boundaries():
    """Test detection of trial boundaries in SpikeEpochs object."""
    trials = np.array([0] * 6 + [1] * 10 + [2] * 3 + [4] * 8)
    times = np.random.rand(len(trials)) * 2. - 0.5
    spk = pln.SpikeEpochs([times], [trials])

    # get boundaries
    tri_bnd, tri_num = _get_trial_boundaries(spk, 0)
    assert (tri_num == np.array([0, 1, 2, 4])).all()
    assert (tri_bnd == np.array([0, 6, 16, 19, 27])).all()


def test_find_index():
    vec = np.array([0.1, 0.5, 0.54, 0.8, 0.95, 1.])

    idx = pln.utils.find_index(vec, 0.5)
    assert idx[0] == 1

    idx = pln.utils.find_index(vec, 0.53)
    assert idx[0] == 2

    idx = pln.utils.find_index(vec, 0.81)
    assert idx[0] == 3

    idx = pln.utils.find_index(vec, 0.99)
    assert idx[0] == 5

    idx = pln.utils.find_index(vec, 0.1)
    assert idx[0] == 0

    idx = pln.utils.find_index(vec, [0.5, 0.55, 0.91])
    assert (idx == np.array([1, 2, 4])).all()


def test_inherit_metadata():
    """Test inheritance of metadata."""
    import pandas as pd

    # just _inherit_metadata function
    metadata = pd.DataFrame({'a': [1, 2, 3], 'b': list('ABC')})
    coords = {'time': np.linspace(-0.2, 0.5, num=25),
            'trial': np.arange(1, 4)}

    _inherit_metadata(coords, metadata, 'trial')

    assert isinstance(coords['a'], tuple)
    assert coords['a'][0] == 'trial'
    assert (coords['a'][1] == metadata['a']).all()
    assert isinstance(coords['a'], tuple)
    assert coords['b'][0] == 'trial'
    assert (coords['b'][1] == metadata['b']).all()


def test_cellinfo_from_xarray():
    spk = create_random_spikes(n_trials=10, n_cells=10)
    cellinfo = pd.DataFrame({'a': np.arange(10), 'b': list('ABCDEFGHIJ'),
                             'd': np.random.rand(10) > 0.5})
    spk.cellinfo = cellinfo

    frate = spk.spike_rate(winlen=0.5, step=0.1)
    cellinfo_reconstructed = pln.utils.cellinfo_from_xarray(frate)
    assert cellinfo_reconstructed.equals(cellinfo)


def test_find_cells():
    spk = create_random_spikes(n_trials=10, n_cells=10)
    channel = (np.tile(np.arange(5)[:, None], [1, 2]) + 1).ravel()
    cluster_id = np.random.randint(50, 1000, size=10)
    spk.cellinfo = pd.DataFrame({'channel': channel, 'cluster': cluster_id})

    cell_idx = 3
    cluster = cluster_id[cell_idx]
    idx = find_cells(spk, cluster=cluster)
    len(idx) == 1
    assert idx[0] == cell_idx
    assert (spk.cellinfo.loc[idx, 'cluster'] == cluster).all()

    # multiple clusters matching
    spk.cellinfo.loc[cell_idx + 1, 'cluster'] = cluster
    with pytest.raises(ValueError, match='Found more than one match'):
        find_cells(spk, cluster=cluster)

    chan = channel[cell_idx + 1]
    idx = find_cells(spk, cluster=cluster, channel=chan)
    len(idx) == 1
    assert idx[0] == cell_idx + 1

    # no such cluster
    cluster = cluster_id.max() + 1
    with pytest.raises(ValueError, match='Could not find any match'):
        find_cells(spk, cluster=cluster)


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
        pln.utils.spike_centered_windows(spk, data)

    # works with time array
    spk_cent = pln.utils.spike_centered_windows(
        spk, data, time=time, winlen=0.01)

    correct = data[np.arange(n_trials), :, spike_indices]
    assert spk_cent.shape == (10, 5, 1)
    assert (spk_cent.data[:, :, 0] == correct).all()

    # should get the same with xarray
    xarr = xr.DataArray(
        data, dims=['trial', 'channel', 'time'],
        coords={'time': time}
    )

    spk_cent2 = pln.utils.spike_centered_windows(
        spk, xarr, winlen=0.01)

    assert (spk_cent == spk_cent2).all().item()

    # but not when we specify incorrect time dim
    msg = 'Coordinate named "emit" not found'
    with pytest.raises(ValueError, match=msg):
        pln.utils.spike_centered_windows(
            spk, xarr, time='emit', winlen=0.01)


def test_sub_ses_parsing():
    sub, ses = pln.utils.parse_sub_ses('U12')
    assert sub == 'U12'
    assert ses is None

    sub, ses = pln.utils.parse_sub_ses('sub-A111')
    assert sub == 'A111'
    assert ses is None

    sub, ses = pln.utils.parse_sub_ses('gc012_main')
    assert sub == 'gc012'
    assert ses == 'main'

    sub, ses = pln.utils.parse_sub_ses('sub-switch001_ses-stim1')
    assert sub == 'switch001'
    assert ses == 'stim1'

    sub, ses = pln.utils.parse_sub_ses(
        'sub-switch001_ses-stim1',
        remove_sub_prefix=False,
        remove_ses_prefix=False
    )
    assert sub == 'sub-switch001'
    assert ses == 'ses-stim1'


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
    xarr = pln.utils.dict_to_xarray(x_dct1)
    x_dct2 = pln.utils.xarray_to_dict(xarr, ensure_correct_reduction=False)
    compare_dicts(x_dct1, x_dct2)

    # test with non-sorted keys - this previously failed
    # because xarray sorts during groupby operation used in xarray_to_dict
    x_dct1 = {'C03': xarr1, 'A02': xarr2, 'W05': xarr1.copy()}
    xarr = pln.utils.dict_to_xarray(x_dct1)

    t_start = time.time()
    x_dct2 = pln.utils.xarray_to_dict(xarr, ensure_correct_reduction=True)
    t_taken = time.time() - t_start
    assert t_taken < 0.1

    compare_dicts(x_dct1, x_dct2)

    xarr_2 = pln.utils.dict_to_xarray(x_dct2)
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
    xarr = pln.utils.dict_to_xarray(x_dct1, select='load == 1')
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
    frates_sel, row_indices = pln.utils.extract_data(
        frates, df_sel, sub_col='sub', df2xarr={'anat': 'anat'}
    )

    # turn the dictionaries to xarrays
    frates_sel_x = pln.utils.dict_to_xarray(frates_sel)
    frates_x = pln.utils.dict_to_xarray(frates)

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
    frates_sel2_x, row_indices2 = pln.utils.extract_data(
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
        pln.utils.aggregate(frates_sel_x, groupby='condition')

    # but should work on a single xarray
    frates_agg_one = pln.utils.aggregate(
        frates_sel['sub-a01'], groupby='condition')
    frates_agg = pln.utils.aggregate(
        frates_sel, groupby='condition')
    n_cells_one = frates_sel['sub-a01'].shape[0]
    assert frates_agg_one.shape[0] == n_cells_one
    assert (frates_agg_one.data == frates_agg[:n_cells_one].data).all()


def test_aggregate_per_cell():
    '''Test aggregation per cell.'''
    n_cells = 10
    n_trials = 50
    arr = gen_random_xarr(n_cells, n_trials, 120, per_cell_coord=True)
    arr_agg = pln.utils.aggregate(arr, groupby='preferred', per_cell=True)

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
    xarr_z = pln.utils.zscore_xarray(xarr)

    # check that mean is zero and std is 1
    assert np.allclose(xarr_z.mean(dim=['trial', 'time']).data, 0, atol=1e-6)
    assert np.allclose(xarr_z.std(dim=['trial', 'time']).data, 1, atol=1e-6)

    # check that zscored data is the same as zscored data from numpy
    xarr_np = xarr.data
    xarr_np_z = (
        (xarr_np - xarr_np.mean(axis=(1, 2), keepdims=True))
         / xarr_np.std(axis=(1, 2), keepdims=True)
    )
    assert np.allclose(xarr_z.data, xarr_np_z, atol=1e-6)

    # test baseline argument
    xarr_z2 = pln.utils.zscore_xarray(xarr, baseline=(-0.5, 0.))
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
