import numpy as np
import pytest

import pylabianca as pln
from pylabianca.utils import _get_trial_boundaries


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

    pln.utils._inherit_metadata(coords, metadata, 'trial')

    assert isinstance(coords['a'], tuple)
    assert coords['a'][0] == 'trial'
    assert (coords['a'][1] == metadata['a']).all()
    assert isinstance(coords['a'], tuple)
    assert coords['b'][0] == 'trial'
    assert (coords['b'][1] == metadata['b']).all()
