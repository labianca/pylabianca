import numpy as np
import pandas as pd


def _deal_with_picks(spk, picks):
    '''Deal with various formats in which ``picks`` can be passed.'''
    has_str = False
    if picks is None:
        # pick all cells by default
        picks = np.arange(len(spk.cell_names))
        return picks
    if isinstance(picks, (list, np.ndarray, pd.Series)):
        if isinstance(picks[0], str):
            # list / array of names
            is_str = [isinstance(x, str) for x in picks[1:]]
            has_str = all(is_str) or len(picks) == 1
        elif all([isinstance(picks[ix], (bool, np.bool_))
                  for ix in range(len(picks))]):
            # list / array of booleans
            picks = np.where(picks)[0]
        elif isinstance(picks, pd.Series):
            picks = picks.values
    if not isinstance(picks, (list, np.ndarray, pd.Series)):
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
                               tri=None, copy_cellinfo=True):
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
        Array of trial indices. Use when the repetitions dimension of ``frate``
        array is not equivalent to trials, but at least some repetitions come
        from the same trial (for example - spikes within trials when using
        spike-centered windows). Passing ``tri`` allows to copy the trial
        metadata correctly.
    copy_cellinfo : bool
        Whether to copy ``spike_epochs.cellinfo`` to xarray.

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

    if spike_epochs.metadata is not None:
        for col in spike_epochs.metadata.columns:
            if tri is None:
                coords[col] = (dimname, spike_epochs.metadata[col])
            else:
                coords[col] = (dimname, spike_epochs.metadata[col].iloc[tri])

    if copy_cellinfo:
        if cell_names is not None and spike_epochs.cellinfo is not None:
            ch_idx = _deal_with_picks(spike_epochs, cell_names)
            for col in spike_epochs.cellinfo.columns:
                coords[col] = (
                    'cell', spike_epochs.cellinfo[col].iloc[ch_idx])

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
    spk : pylabianca.SpikeEpochs
        Spike epochs object.
    cell_idx : int
        Index of the cell providing spikes.
    arr : np.ndarray | xarray.DataArray
        Array with the signal to be cut out. The dimesions should be
        ``n_trials x n_channels x n_times``.
    time : np.ndarray
        Coordinates of the time axis.
    sfreq : float
        Sampling frequency.
    winlen : float
        Window length in seconds.

    Returns
    -------
    spike_centered : xarray.DataArray
        Spike-centered windows. ``n_spikes x n_channels x n_times``.
    '''
    import xarray as xr
    from borsar.utils import find_index

    spike_centered = list()
    window_samples, half_win = _symmetric_window_samples(winlen, sfreq)
    window_time = window_samples / sfreq
    winlims = np.array([-half_win, half_win + 1])[None, :]
    lims = [0, len(time)]
    tri_is_ok = np.zeros(len(spk.trial[cell_idx]), dtype='bool')

    n_tri = max(spk.trial[cell_idx]) + 1
    for tri_idx in range(n_tri):
        sel = spk.trial[cell_idx] == tri_idx
        if sel.any():
            tms = spk.time[cell_idx][sel]

            closest_smp = find_index(time, tms)
            twins = closest_smp[:, None] + winlims
            good = ((twins >= lims[0]) & (twins <= lims[1])).all(axis=1)
            twins = twins[good]
            tri_is_ok[sel] = good

            for twin in twins:
                sig_part = arr[tri_idx, :, twin[0]:twin[1]]
                spike_centered.append(sig_part)

    # stack windows
    spike_centered = np.stack(spike_centered, axis=0)

    # prepare coordinates
    spike_idx = np.where(tri_is_ok)[0]
    tri = spk.trial[cell_idx][tri_is_ok]
    n_channels = arr.shape[1]
    channel_idx = np.arange(n_channels)

    # construct xarray and assign coords
    spike_centered = xr.DataArray(
        spike_centered, dims=['spike', 'channel', 'time'],
        coords={'spike': spike_idx, 'channel': channel_idx,
                'time': window_time})
    spike_centered = spike_centered.assign_coords(trial=('spike', tri))

    return spike_centered


# TODO - if other sorters are used, alignment point (sample_idx) for the
#        spike waveforms should be saved somewhere in spk
def infer_waveform_polarity(spk, cell_idx, threshold=1.75, baseline_range=50):
    """Decide whether waveform polarity is positive, negative or unknown.

    The decision is based on comparing baselined min and max average waveform
    peak values. The value for the peak away from alignment point is calculated
    from single spike waveforms to simulate alignment and reduce bias (30
    samples around that peak are taken and min/max values for this time window).
    The alignment point is expected where it is for osort - around sample 92.

    Parameters
    ----------
    spk : pylabianca.Spikes
        Spikes object to use.
    cell_idx : int
        Index of the cell whose waveform should be checked.
    threshold : float
        Threshold ratio for the minimum and maximum waveform peak values to
        decide about polarity. Default is ``1.75``, which means that one of
        the peaks (min or max) must be at least 1.75 times higher than the
        other to decide on polarity. If given waveform does not pass this
        test it is labelled as ``'unknown'``.
    baseline_range : int
        Number of first samples to use as baseline. Default is ``50``.

    Returns
    -------
    unit_type : str
        Polarity label for the waveform. Either ``'positive'``, ``'negative'``
        or ``'unknown'``.
    """

    inv_threshold = 1 / threshold

    # decide whether the waveform is pos or neg
    avg_waveform = spk.waveform[cell_idx].mean(axis=0)
    min_val_idx, max_val_idx = avg_waveform.argmin(), avg_waveform.argmax()
    min_val, max_val = avg_waveform.min(), avg_waveform.max()

    # the value not aligned to will be underestimated, correct for that ...
    further_away = np.abs(np.array([min_val_idx, max_val_idx]) - 92).argmax()
    operation = [np.min, np.max][further_away]
    away_idx = [min_val_idx, max_val_idx][further_away]

    # ... by estimating this value in a wider window
    rng = slice(away_idx - 15, away_idx + 15)
    slc = spk.waveform[cell_idx][:, rng]
    this_val = operation(slc, axis=1).mean()

    if further_away == 0:
        min_val = this_val
    else:
        max_val = this_val

    # the min and max values are baselined to further reduce bias
    baseline = avg_waveform[:baseline_range].mean()
    min_val -= baseline
    max_val -= baseline

    # based on min / max ratio a decision is made
    prop = min_val / max_val

    if np.abs(prop) > threshold:
        unit_type = 'neg'
    elif np.abs(prop) < inv_threshold:
        unit_type = 'pos'
    else:
        unit_type = 'unknown'

    return unit_type
