import numpy as np
import pytest

import pylabianca as pln
from pylabianca.utils import get_fieldtrip_data


@pytest.fixture(scope="session")
def ft_data():
    ft_data = get_fieldtrip_data()
    spk = pln.io.read_plexon_nex(ft_data)
    return spk


@pytest.fixture(scope="session")
def spk_epochs(ft_data):
    # read and epoch data
    events_test = np.array([[22928800, 0, 1],
                            [171087520, 0, 1],
                            [300742480, 0, 1]])

    spk_epo_test = (ft_data.copy().pick_cells(['sig002a_wf', 'sig003a_wf'])
                    .epoch(events_test, tmin=-2.75, tmax=3.,
                           keep_timestamps=True)
    )
    return spk_epo_test
