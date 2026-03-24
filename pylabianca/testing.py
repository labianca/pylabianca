from string import ascii_lowercase

import numpy as np
import pytest

import pylabianca as pln
from pylabianca.utils import get_fieldtrip_data, create_random_spikes


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


def gen_random_xarr(n_cells, n_trials, n_times, per_cell_coord=False,
                    trial_condition_levels=None):
    import xarray as xr

    letters = np.array(list(ascii_lowercase))
    dim_names = ['cell', 'trial', 'time']
    time = np.linspace(-0.5, 1.5, num=n_times)
    data = np.random.rand(n_cells, n_trials, n_times)
    cell_names = [''.join(np.random.choice(letters, 5))
                  for _ in range(n_cells)]

    coords={'cell': cell_names, 'trial': np.arange(n_trials), 'time': time}
    if trial_condition_levels is not None:
        levels = np.random.choice(trial_condition_levels, size=n_trials)
        coords['cond'] = ('trial', levels)

    xarr = xr.DataArray(data, dims=dim_names, coords=coords)

    if per_cell_coord:
        prefs = np.zeros((n_cells, n_trials), dtype=int)
        for cell_idx in range(n_cells):
            this_prefs = np.random.choice([0, 1, 2], size=n_trials)
            prefs[cell_idx, :] = this_prefs

        xarr = xarr.assign_coords(preferred=(('cell', 'trial'), prefs))

    return xarr


def create_multisession_data(n_sessions, cells_per_session=(5, 25), out='fr'): 
    import pandas as pd

    assert out in ['fr', 'spk']

    output = dict()
    sub_idx, ses_idx = 1, 1
    n_cell_diff = cells_per_session[1] - cells_per_session[0]
    for ses_idx in range(n_sessions):
        if np.random.rand() < 0.7:
            sub_idx += 1
            ses_idx = 1
        else:
            ses_idx += 1
        subses_key = f'sub-{sub_idx:02d}_ses-{ses_idx:02d}'
        n_cells = int(np.round(
            np.random.rand() * n_cell_diff + cells_per_session[0]
        ))

        spk_epochs = create_random_spikes(
            n_cells=n_cells, n_trials=60, n_spikes=(10, 50))
        emo = np.random.choice(['sad', 'happy', 'neutral'], size=60)
        block = np.tile([1, 2], (30, 1)).ravel()
        spk_epochs.metadata = pd.DataFrame({'emo': emo, 'block': block})

        if out == 'spk':
            output[subses_key] = spk_epochs
        elif out == 'fr':
            output[subses_key] = spk_epochs.spike_rate(
                tmin=0.1, tmax=1.1, step=False)

    return output
