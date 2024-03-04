import numpy as np
import pandas as pd
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
