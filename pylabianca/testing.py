from string import ascii_lowercase

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


def gen_random_xarr(n_cells, n_trials, n_times, per_cell_coord=False):
    import xarray as xr

    letters = np.array(list(ascii_lowercase))
    dim_names = ['cell', 'trial', 'time']
    time = np.linspace(-0.5, 1.5, num=n_times)
    data = np.random.rand(n_cells, n_trials, n_times)
    cell_names = [''.join(np.random.choice(letters, 5))
                  for _ in range(n_cells)]

    xarr = xr.DataArray(
        data, dims=dim_names,
        coords={'cell': cell_names,
                'trial': np.arange(n_trials),
                'time': time}
    )

    if per_cell_coord:
        prefs = np.zeros((n_cells, n_trials), dtype=int)
        for cell_idx in range(n_cells):
            this_prefs = np.random.choice([0, 1, 2], size=n_trials)
            prefs[cell_idx, :] = this_prefs

        xarr = xarr.assign_coords(preferred=(('cell', 'trial'), prefs))

    return xarr