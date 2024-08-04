import numpy as np
import pandas as pd
import xarray as xr
import pytest

import pylabianca as pln
from pylabianca.utils import (_get_trial_boundaries, find_cells_by_cluster_id,
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


def test_find_cells_by_cluster_id():
    spk = create_random_spikes(n_trials=10, n_cells=10)
    channel = (np.tile(np.arange(5)[:, None], [1, 2]) + 1).ravel()
    cluster_id = np.random.randint(50, 1000, size=10)
    spk.cellinfo = pd.DataFrame({'channel': channel, 'cluster': cluster_id})

    cell_idx = 3
    cluster = cluster_id[cell_idx]
    idx = find_cells_by_cluster_id(spk, [cluster])
    len(idx) == 1
    assert idx[0] == cell_idx
    assert (spk.cellinfo.loc[idx, 'cluster'] == cluster).all()

    # multiple clusters matching
    spk.cellinfo.loc[cell_idx + 1, 'cluster'] = cluster
    with pytest.raises(ValueError, match='Found 0 or > 1 cluster IDs.'):
        find_cells_by_cluster_id(spk, [cluster])

    chan = channel[cell_idx + 1]
    idx = find_cells_by_cluster_id(spk, [cluster], channel=chan)
    len(idx) == 1
    assert idx[0] == cell_idx + 1

    # no such cluster
    cluster = cluster_id.max() + 1
    with pytest.raises(ValueError, match='Found 0 or > 1 cluster IDs.'):
        find_cells_by_cluster_id(spk, [cluster])


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

    sub, ses = pln.utils.parse_sub_ses('sub-switch001_ses-stim1',
                                    remove_sub_prefix=False,
                                    remove_ses_prefix=False)
    assert sub == 'sub-switch001'
    assert ses == 'ses-stim1'


def test_xarr_dct_conversion():
    from string import ascii_lowercase
    import xarray as xr

    def compare_dct_xarr(darr1, darr2):
        for key in darr1.keys():
            assert (darr1[key].data == darr2[key].data).all()
            coord_list = list(darr1[key].coords)
            for coord in coord_list:
                assert (darr1[key].coords[coord].values
                        == darr2[key].coords[coord].values).all()

    letters = list(ascii_lowercase)
    n_cells1, n_cells2, n_trials, n_times = 10, 15, 20, 100
    time = np.linspace(-0.5, 1.5, num=n_times)
    cell_names = ['cell_{}'.format(''.join(np.random.choice(letters, 10)))
                  for _ in range(max(n_cells1, n_cells2))]

    dim_names = ['cell', 'trial', 'time']
    xarr1 = xr.DataArray(np.random.rand(n_cells1, n_trials, n_times),
                         dims=dim_names,
                         coords={'cell': cell_names[:n_cells1],
                                 'trial': np.arange(n_trials),
                                 'time': time})
    xarr2 = xr.DataArray(np.random.rand(n_cells2, n_trials, n_times),
                         dims=dim_names,
                         coords={'cell': cell_names[:n_cells2],
                                 'trial': np.arange(n_trials),
                                 'time': time})

    load = np.concatenate([np.ones(10), np.ones(10) * 2])
    np.random.shuffle(load)
    xarr1 = xarr1.assign_coords({'load': ('trial', load)})
    load2 = load.copy()
    np.random.shuffle(load2)
    xarr2 = xarr2.assign_coords({'load': ('trial', load2)})
    x_dct1 = {'sub-A01': xarr1, 'sub-A02': xarr2}

    xarr = pln.utils.dict_to_xarray(x_dct1)
    x_dct2 = pln.utils.xarray_to_dict(xarr)
    compare_dct_xarr(x_dct1, x_dct2)

    x_dct3 = pln.utils.xarray_to_dict(xarr, ensure_correct_reduction=False)
    compare_dct_xarr(x_dct1, x_dct3)


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
    # TODO
    pass


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
