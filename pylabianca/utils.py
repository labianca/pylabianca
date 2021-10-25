import numpy as np


def _deal_with_picks(spk, picks):
    has_str = False
    if picks is None:
        # pick all cells by default
        picks = np.arange(len(spk.cell_names))
    if isinstance(picks, (list, np.ndarray)):
        if isinstance(picks[0], str):
            is_str = [isinstance(x, str) for x in picks[1:]]
            has_str = all(is_str) or len(picks) == 1
    if not isinstance(picks, (list, np.ndarray)):
        if isinstance(picks, str):
            has_str = True
        picks = [picks]
    if has_str:
        if isinstance(spk.cell_names, list):
            picks = [spk.cell_names.index(name) for name in picks]
        else:
            picks = [np.where(spk.cell_names == name)[0][0] for name in picks]
    return picks


# TODO: consider changing the array dim order to: trials, cells, times
#       (mne-python-like)
def _turn_spike_rate_to_xarray(times, frate, spike_epochs, cell_names=None,
                               tri=None):
    '''Turn spike rate data to xarray.

    Parameters
    ----------
    times : numpy array | str
        Vector of time points for which spike rate was calculated (middle
        timepoints for the time window used). Can also be a string
        describing the time window if static window was used.
    frate : numpy array
        Numpy array with firing rate, with the following dimensions:
        * 3d ``n_cells x n_trials x n_times`` (``cell_names`` has to be not
          None)
        * 2d ``n_cells x n_trials`` (``cell_names`` not None and ``times``
          as string)
        * 2d ``n_trials x n_times`` (``cell_names`` is None and ``times``
          is an array)
    spike_epochs : SpikeEpochs object
        SpikeEpochs object.
    cell_names : list-like of str | None
        Names of the picked cells. If not ``None`` then indicates that the
        first dimension of ``frate`` contains cells.
    tri : np.ndarray | None
        Array of trial indices. Use when the repetitions dimesnion is not
        equivalent to trials, but spikes within trials (spike-centered
        windows). Passing this argument allows to copy the trial metadata
        correctly.

    Returns
    -------
    firing : xarray
        Firing rate xarray.
    '''
    import xarray as xr

    # later: consider having firing rate from many neurons...
    n_trials = frate.shape[0] if cell_names is None else frate.shape[1]
    dimname = 'trial' if tri is None else 'spike'
    coords = {dimname: np.arange(n_trials)}
    dims = [dimname]
    attrs = None
    if isinstance(times, np.ndarray):
        dims.append('time')
        coords['time'] = times
    else:
        attrs = {'timewindow': times}

    if cell_names is not None:
        assert frate.shape[0] == len(cell_names)
        dims = ['cell'] + dims
        coords['cell'] = cell_names

    if tri is not None:
        coords['trial'] = (dimname, tri)

    for col in spike_epochs.metadata.columns:
        if tri is None:
            coords[col] = (dimname, spike_epochs.metadata[col])
        else:
            coords[col] = (dimname, spike_epochs.metadata[col].iloc[tri])

    firing = xr.DataArray(frate, dims=dims, coords=coords,
                          name='firing rate', attrs=attrs)
    return firing
