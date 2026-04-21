from string import ascii_lowercase

import numpy as np
import pytest

import pylabianca as pln
from pylabianca.utils import get_fieldtrip_data


@pytest.fixture(scope="session")
def ft_data():
    """Session fixture with example FieldTrip spikes data."""
    ft_data = get_fieldtrip_data()
    spk = pln.io.read_plexon_nex(ft_data)
    return spk


@pytest.fixture(scope="session")
def spk_epochs(ft_data):
    """Session fixture with epoched FieldTrip spikes data."""
    # read and epoch data
    events_test = np.array([[22928800, 0, 1],
                            [171087520, 0, 1],
                            [300742480, 0, 1]])

    spk_epo_test = (ft_data.copy().pick_cells(['sig002a_wf', 'sig003a_wf'])
                    .epoch(events_test, tmin=-2.75, tmax=3.,
                           keep_timestamps=True)
    )
    return spk_epo_test


def random_xarray(n_cells, n_trials, n_times, per_cell_coord=False,
                  trial_condition_levels=None):
    """Create a random `(cell, trial, time)` firing-rate-like DataArray.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    n_trials : int
        Number of trials.
    n_times : int
        Number of time points.
    per_cell_coord : bool
        Whether to attach a 2D per-cell trial coordinate named
        ``preferred``.
    trial_condition_levels : array-like | None
        Optional condition levels used to generate a trial-wise coordinate
        named ``cond``.

    Returns
    -------
    xarray.DataArray
        Random DataArray with dimensions ``('cell', 'trial', 'time')``.
    """
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


def random_multisession_xarray(n_sessions, cells_per_session=(5, 25),
                               out='fr'):
    """Create dictionary of random per-session spike or firing-rate data.

    Parameters
    ----------
    n_sessions : int
        Number of sessions to create.
    cells_per_session : tuple of int
        Inclusive low/high range used when sampling the number of cells.
    out : {'fr', 'spk'}
        Output object for each session:
        * ``'fr'`` returns an xarray firing-rate DataArray.
        * ``'spk'`` returns a :class:`pylabianca.SpikeEpochs`.

    Returns
    -------
    dict
        Dictionary keyed by ``sub-XX_ses-YY`` containing either
        :class:`xarray.DataArray` (``out='fr'``) or
        :class:`pylabianca.SpikeEpochs` (``out='spk'``).
    """
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

        spk_epochs = random_spikes(
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


def random_multisession_spikes(
    keys, n_cells, n_trials=40, n_spikes=(25, 65), conditions=('A', 'B', 'C'),
    anat_region=('HIP', 'AMY', 'ACC')
):
    """Create a dictionary of random :class:`pylabianca.SpikeEpochs`.

    Parameters
    ----------
    keys : sequence of str
        Session identifiers used as dictionary keys.
    n_cells : sequence of int
        Number of cells for each session in ``keys``.
    n_trials : int
        Number of trials per session.
    n_spikes : int | tuple of int
        Spike count configuration passed to :func:`random_spikes`.
    conditions : sequence
        Condition levels used to create trial metadata column ``condition``.
    anat_region : sequence
        Region labels used to create cellinfo column ``anat``.

    Returns
    -------
    dict
        Dictionary where each value is a :class:`pylabianca.SpikeEpochs`
        containing per-session ``metadata`` and ``cellinfo``.
    """
    import pandas as pd

    spk_dict = {}
    for this_key, this_n_cells in zip(keys, n_cells):
        metadata = pd.DataFrame({
            'condition': np.random.choice(conditions, size=n_trials)
        })
        cellinfo = pd.DataFrame({
            'anat': np.sort(np.random.choice(anat_region, size=this_n_cells))
        })
        spk_dict[this_key] = random_spikes(
            n_cells=this_n_cells, n_trials=n_trials, n_spikes=n_spikes,
            metadata=metadata, cellinfo=cellinfo
        )
    return spk_dict


def random_spikes(n_cells=4, n_trials=25, n_spikes=(10, 21), **args):
    """Create random :class:`pylabianca.Spikes` or `SpikeEpochs` test data.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    n_trials : int | None
        Number of trials. If ``None`` or non-positive then
        :class:`pylabianca.Spikes` is returned.
    n_spikes : int | tuple of int
        Number of spikes per trial (or per cell for raw spikes).
        If an ``int``, the same number is used everywhere.
        If a tuple, spikes are sampled uniformly in
        ``[n_spikes[0], n_spikes[1])``.
    **args : dict
        Extra keyword arguments passed to :class:`pylabianca.Spikes` or
        :class:`pylabianca.SpikeEpochs`.

    Returns
    -------
    spikes : pylabianca.Spikes | pylabianca.SpikeEpochs
        Randomly generated spike object.
    """
    from pylabianca.spikes import SpikeEpochs, Spikes

    tmin, tmax = -0.5, 1.5
    tlen = tmax - tmin
    constant_n_spikes = isinstance(n_spikes, int)
    if constant_n_spikes:
        n_spk = n_spikes

    return_epochs = isinstance(n_trials, int) and n_trials > 0
    if not return_epochs:
        n_trials = 1
        tmin = 0
        tmax = 1e6

    times = list()
    trials = list()
    for _ in range(n_cells):
        this_tri = list()
        this_tim = list()
        for tri_idx in range(n_trials):
            if not constant_n_spikes:
                n_spk = np.random.randint(*n_spikes)

            if return_epochs:
                tms = np.random.rand(n_spk) * tlen + tmin
                this_tri.append(np.ones(n_spk, dtype=int) * tri_idx)
            else:
                tms = np.random.randint(tmin, tmax, size=n_spk)
            this_tim.append(np.sort(tms))

        times.append(np.concatenate(this_tim))

        if return_epochs:
            trials.append(np.concatenate(this_tri))

    if return_epochs:
        return SpikeEpochs(times, trials, time_limits=(tmin, tmax), **args)

    if 'sfreq' not in args:
        args['sfreq'] = 10_000

    return Spikes(times, **args)
