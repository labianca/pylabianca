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
        Array of trial indices. Use when the repetitions dimension is not
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


def _symmetric_window_samples(winlen, sfreq):
    '''Returns a symmetric window of given length.'''
    half_len_smp = int(np.round(winlen / 2 * sfreq))
    win_smp = np.arange(-half_len_smp, half_len_smp + 1)
    return win_smp, half_len_smp


def _gauss_kernel_samples(window, gauss_sd):
    '''Returns a gaussian kernel given window and sd in samples.'''
    from scipy.stats.distributions import norm
    kernel = norm(loc=0, scale=gauss_sd)
    kernel = kernel.pdf(window)
    return kernel


# TODO: add offsets to spike-centered neurons
def spike_centered_windows(spk, cell_idx, arr, time, sfreq, winlen=0.1):
    '''Cut out windows from signal centered on spike times.

    Parameters
    ----------
    spk : FIXME
        FIXME
    cell_idx : FIXME
        FIXME
    arr : xarray.DataArray
        FIXME
    time : FIXME
        FIXME
    sfreq : FIXME
        FIXME
    winlen : FIXME
        FIXME
    '''
    from borsar.utils import find_index

    spike_centered = list()
    _, half_win = _symmetric_window_samples(winlen, sfreq)
    winlims = np.array([-half_win, half_win + 1])[None, :]
    lims = [0, len(time)]
    tri_is_ok = np.zeros(len(spk.trial[cell_idx]), dtype='bool')

    n_tri = max(spk.trial[cell_idx])
    for tri_idx in range(n_tri):
        sel = spk.trial[cell_idx] == tri_idx
        if sel.any():
            tms = spk.time[cell_idx][sel]
            if len(tms) < 1:
                continue

            closest_smp = find_index(time, tms)
            twins = closest_smp[:, None] + winlims
            good = ((twins >= lims[0]) & (twins <= lims[1])).all(axis=1)
            twins = twins[good]
            tri_is_ok[sel] = good

            for twin in twins:
                sig_part = arr[:, tri_idx, twin[0]:twin[1]]
                spike_centered.append(sig_part)

    spike_centered = np.stack(spike_centered, axis=1)
    tri = spk.trial[cell_idx][tri_is_ok]
    return spike_centered, tri
