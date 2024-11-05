import time

import numpy as np
import pandas as pd
import pytest

import pylabianca as pln
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

    # when no cellinfo it should return None:
    spk.cellinfo = None
    frate = spk.spike_rate(winlen=0.5, step=0.1)
    cellinfo_reconstructed = pln.utils.cellinfo_from_xarray(frate)
    assert cellinfo_reconstructed is None


def test_find_cells():
    spk = create_random_spikes(n_trials=10, n_cells=10)
    channel = (np.tile(np.arange(5)[:, None], [1, 2]) + 1).ravel()

    # generate unique cluster ids
    is_unique = False
    while not is_unique:
        cluster_id = np.random.randint(50, 1000, size=10)
        is_unique = len(np.unique(cluster_id)) == 10

    # create and assign cellinfo
    spk.cellinfo = pd.DataFrame({'channel': channel, 'cluster': cluster_id})

    # test _get_cellinfo
    info = pln.utils._get_cellinfo(spk)
    assert (info == spk.cellinfo).all().all()

    fr = spk.spike_rate()
    info = pln.utils._get_cellinfo(fr)
    assert (info == spk.cellinfo).all().all()

    info = pln.utils._get_cellinfo(spk.cellinfo)
    assert (info == spk.cellinfo).all().all()

    msg = 'has to be a Spikes, SpikeEpochs, xarray'
    with pytest.raises(ValueError, match=msg):
        info = pln.utils._get_cellinfo(list('abcd'))

    spk2 = spk.copy()
    spk2.cellinfo = None
    msg = 'No cellinfo found in the provided object.'
    with pytest.raises(ValueError, match=msg):
        info = pln.utils._get_cellinfo(spk2)

    # test find_cells
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

    with pytest.warns(UserWarning):
        find_cells(spk, cluster=cluster, more_found='warn')

    idx = find_cells(spk, cluster=cluster, more_found='ignore')
    assert len(idx) == 2

    chan = channel[cell_idx + 1]
    idx = find_cells(spk, cluster=cluster, channel=chan)
    len(idx) == 1
    assert idx[0] == cell_idx + 1

    # no such cluster
    cluster = cluster_id.max() + 1
    with pytest.raises(ValueError, match='Could not find any match'):
        find_cells(spk, cluster=cluster)

    # no such feature
    msg = 'Feature "numpy" is not present in the cellinfo'
    with pytest.raises(ValueError, match=msg):
        find_cells(spk, numpy=[1, 2, 3])

    # wrong more_found or not_found argument:
    msg = '"{}" has to be one of:'
    with pytest.raises(ValueError, match=msg.format('more_found')):
        find_cells(spk, cluster=cluster, more_found='wrong_arg')

    with pytest.raises(ValueError, match=msg.format('not_found')):
        find_cells(spk, cluster=cluster, not_found='wrong_arg')

    # tiling length one search features:
    spk.cellinfo.loc[0:1, 'cluster'] = [17, 23]
    spk.cellinfo.loc[2:3, 'cluster'] = [17, 23]

    row_idx = find_cells(spk, channel=2, cluster=[17, 23])
    assert (row_idx == [2, 3]).all()

    # when search features of different lengths are provided
    # (except the length one case) we should get a ValueError
    msg = ('Number of elements per search feature has to be '
           'the same across all search features')
    with pytest.raises(ValueError, match=msg):
        idx = pln.utils.find_cells(
            info, channel=[0, 2], cluster=[10, 25, 30])

    # test dropping with drop_cells_by_channel_and_cluster_id
    should_drop = [3, 7]
    drop_cells = [spk.cell_names[idx] for idx in should_drop]
    to_drop = [(x['cluster'], x['channel'])
               for _, x in spk.cellinfo.loc[should_drop, :].iterrows()]
    pln.utils.drop_cells_by_channel_and_cluster_id(spk, to_drop)

    for cell in drop_cells:
        assert cell not in spk.cell_names


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


def test_turn_spike_rate_to_xarray():
    # test 1: 2d array, cell_names None
    # then it is trials x times
    times = np.linspace(0.1, 0.8, num=20)
    spk = create_random_spikes(n_trials=10, n_cells=2)
    spk.metadata = pd.DataFrame({'a': np.arange(10), 'b': list('ABCDEFGHIJ')})
    arr = np.random.rand(10, 20)
    xr = pln.utils._turn_spike_rate_to_xarray(times, arr, spk)

    assert xr.dims == ('trial', 'time')
    assert (xr.time.values == times).all()
    assert (xr.trial.values == np.arange(10)).all()

    # make sure that metadata is inherited
    trial_dims = pln.utils.xr_find_nested_dims(xr, 'trial')
    assert len(trial_dims) == 2
    assert 'a' in trial_dims
    assert 'b' in trial_dims

    # test 2: ndim = 2, cell_names not None, and times str
    # then it is cells x times
    times = '0.1 - 0.8 s'
    arr = np.random.rand(2, 10)
    xr = pln.utils._turn_spike_rate_to_xarray(
        times, arr, spk, cell_names=spk.cell_names)

    assert xr.dims == ('cell', 'trial')
    assert (xr.cell.values == spk.cell_names).all()
    assert (xr.trial.values == np.arange(10)).all()

    # test 3: ndim = 2, cell_names not None, and times array
    # then it is cells x times
    times = np.linspace(0.1, 0.8, num=10)
    xr = pln.utils._turn_spike_rate_to_xarray(
        times, arr, spk, cell_names=spk.cell_names)

    assert xr.dims == ('cell', 'time')
    assert (xr.cell.values == spk.cell_names).all()
    assert (xr.time.values == times).all()
