import os
import os.path as op
from warnings import warn

import numpy as np
import pandas as pd


def _deal_with_picks(spk, picks):
    '''Deal with various formats in which ``picks`` can be passed.

    Parameters
    ----------
    spk : pylabianca.Spikes | pylabianca.SpikeEpochs
        Spikes or SpikeEpochs object.
    picks : int | str | list-like of int | list-like of str | None
        The units to pick.

    Returns
    -------
    picks : list
        List of indices of the picked cells.
    '''
    has_str = False
    if picks is None:
        # pick all cells by default
        picks = np.arange(len(spk.cell_names))
        return picks
    if isinstance(picks, (list, np.ndarray, pd.Series)):
        if len(picks) == 0:
            raise ValueError('No cells selected.')
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


# CONSIDER: changing the array dim order to: trials, cells, times
#           (mne-python-like)
# CHANGE name to something more general - it is now used for xarray and decoding
#        results (and more in the future)
# spike_epochs is used only when n_trials > 0 to inherit metadata
#              and to copy cellinfo
def _turn_spike_rate_to_xarray(times, frate, spike_epochs, cell_names=None,
                               tri=None, copy_cellinfo=True,
                               x_dim_name='time'):
    '''Turn spike rate data to xarray.

    Parameters
    ----------
    times : numpy array | str
        Vector of time points for which spike rate was calculated (middle
        time points for the time window used). Can also be a string
        describing the time window if static window was used.
    frate : numpy array
        Numpy array with firing rate, with the following dimensions:

        * 3d ``n_cells x n_trials x n_times`` (``cell_names`` has to be not
          None)
        * 2d ``n_cells x n_trials`` (``cell_names`` not None and ``times``
          as string)
        * 2d ``n_trials x n_times`` (``cell_names`` is None and ``times``
          is an array)
        * 2d ``n_cells x n_times`` (``cell_names`` is not None and ``times``
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
    x_dim_name : str
        Name of the last dimension. Defaults to ``'time'``.

    Returns
    -------
    firing : xarray
        Firing rate xarray.
    '''
    import xarray as xr

    # later: consider having firing rate from many neurons...
    times_array = isinstance(times, np.ndarray)
    if frate.ndim == 3:
        n_trials = frate.shape[0] if cell_names is None else frate.shape[1]
    elif frate.ndim == 2:
        if cell_names is None:
            n_trials = frate.shape[0]
        else:
            if times_array:
                n_trials = 0
            else:
                n_trials = frate.shape[1]

    if n_trials > 0:
        dimname = 'trial' if tri is None else 'spike'
        coords = {dimname: np.arange(n_trials)}
        dims = [dimname]
    else:
        coords = dict()
        dims = list()

    attrs = None
    if isinstance(times, np.ndarray):
        dims.append(x_dim_name)
        coords[x_dim_name] = times
    else:
        attrs = {'timewindow': times}

    if cell_names is not None:
        assert frate.shape[0] == len(cell_names)
        dims = ['cell'] + dims
        coords['cell'] = cell_names

    if tri is not None:
        coords['trial'] = (dimname, tri)

    if n_trials > 0:
        coords = _inherit_metadata(
            coords, spike_epochs.metadata, dimname, tri=tri)

    if copy_cellinfo:
        if cell_names is not None and spike_epochs.cellinfo is not None:
            ch_idx = _deal_with_picks(spike_epochs, cell_names)
            for col in spike_epochs.cellinfo.columns:
                coords[col] = (
                    'cell', spike_epochs.cellinfo[col].iloc[ch_idx])

    firing = xr.DataArray(frate, dims=dims, coords=coords,
                          attrs=attrs)
    return firing


def _inherit_metadata(coords, metadata, dimname, tri=None):
    if metadata is not None:
        for col in metadata.columns:
            if tri is None:
                coords[col] = (dimname, metadata[col])
            else:
                coords[col] = (dimname, metadata[col].iloc[tri])
    return coords


def _inherit_metadata_from_xarray(xarr_from, xarr_to, dimname,
                                  copy_coords=None):
    if copy_coords is None:
        copy_coords = xr_find_nested_dims(xarr_from, dimname)
    if len(copy_coords) > 0:
        coords = {coord: (dimname, xarr_from.coords[coord].values)
                  for coord in copy_coords}
        xarr_to = xarr_to.assign_coords(coords)
    return xarr_to


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


# ENH: change `sel` var creation to use trial boundaries
# ENH: test speed and consider numba
# ENH: allow for asymmetric windows (like in fieldtrip)
def spike_centered_windows(spk, arr, pick=None, time=None, sfreq=None,
                           winlen=0.1):
    '''Cut out windows from signal centered on spike times.

    Parameters
    ----------
    spk : pylabianca.SpikeEpochs
        Spike epochs object.
    arr : np.ndarray | xarray.DataArray | mne.Epochs
        Array with the signal to be cut out. The dimensions should be
        ``n_trials x n_channels x n_times``.
    pick : int | str
        Cell providing spikes centering windows.
    time : None | str | np.ndarray
        Time information for ``arr`` input:

        * if ``arr`` is an ``np.ndarray`` then ``time`` should be an array
          of time points for each sample of the time dimension.
        * if ``arr`` is an ``xarray.DataArray`` then ``time`` can be either
          a name of the time dimension or ``None`` - in the latter case the
          time coordinates are taken from ``'time'`` coordinate of ``arr``.
        * if ``arr`` is an ``mne.Epochs`` object then ``time`` is ignored.
    sfreq : float | None
        Sampling frequency. Inferred from analog signal time dimension if not
        provided (or taken from ``.info['sfreq']`` if ``arr`` is ``mne.Epochs``
        ).
    winlen : float
        Window length in seconds. This is the full window length: a window
        length of ``0.1`` means that the window will start 0.05 seconds before
        and end 0.05 seconds after the spike time. Defaults to ``0.1``.

    Returns
    -------
    spike_centered : xarray.DataArray
        Spike-centered windows. ``n_spikes x n_channels x n_times``.
    '''
    import xarray as xr

    # check inputs
    picks = _deal_with_picks(spk, pick)
    cell_idx = picks[0]
    ch_names = None
    unit = None
    metadata = spk.metadata

    if isinstance(arr, xr.DataArray):
        if time is None:
            if 'time' not in arr.coords:
                raise ValueError('When ``time=None`` the ``arr`` xarray has '
                                 'to contain a coordinate named "time". '
                                 'Alternatively, pass the name of the time '
                                 'coordinate in ``time`` input argument.')
            time = arr.coords['time'].values
        else:
            if isinstance(time, str):
                if time not in arr.coords:
                    raise ValueError(f'Coordinate named "{time}" not found in '
                                     '``arr`` xarray.')
                time = arr.coords[time].values
            else:
                raise ValueError('When ``arr`` is an xarray ``time`` input '
                                 'argument has to be either ``None`` or a '
                                 f'string, got {type(time)}.')

    elif isinstance(arr, np.ndarray):
        if time is None:
            raise ValueError('When ``arr`` is an ndarray, ``time`` input '
                             'argument has to be an array of time points '
                             'for each sample of the time dimension.')
    else:
        import mne
        if isinstance(arr, mne.Epochs):
            time = arr.times
            ch_names = arr.ch_names
            sfreq = arr.info['sfreq']
            unit = 'V'
            if metadata is None:
                metadata = arr.metadata

            arr = arr.get_data()
        else:
            raise ValueError('``arr`` has to be either an xarray, numpy array '
                             f'or mne.Epochs, got {type(arr)}.')

    if sfreq is None:
        sfreq = 1 / np.diff(time).mean()
    if ch_names is None:
        n_channels = arr.shape[1]
        ch_names = np.arange(n_channels)

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

    coords = {'spike': spike_idx, 'channel': ch_names,
              'time': window_time, 'trial': ('spike', tri)}
    coords = _inherit_metadata(coords, metadata, 'spike', tri=tri)

    # construct xarray and assign coords
    spike_centered = xr.DataArray(
        spike_centered, dims=['spike', 'channel', 'time'],
        coords=coords, name='amplitude')
    if unit is not None:
        spike_centered.attrs['unit'] = unit

    return spike_centered


# TODO: differentiate between shuffling spike-trials vs just metadata
#       -> an argument for that?
#       -> or just name this shuffle_spikes
def shuffle_trials(spk, drop_timestamps=True, drop_waveforms=True):
    '''Create a copy of the SpikeEpochs object with shuffled trials.

    Here the spike-trial relationship is shuffled, not trial metadata.
    Shuffling spikes is more costly than simply shuffling metadata, but it is
    necessary when the spike-trial relationship is important (for example in
    spike-triggered averaging).

    Parameters
    ----------
    spk : SpikeEpochs
        SpikeEpochs object.
    drop_timestamps : bool
        If True, timestamps are not copied to the new object.
    drop_waveforms : bool
        If True, waveforms are not copied to the new object.

    Returns
    -------
    new_spk : SpikeEpochs
        SpikeEpochs object with shuffled trials.
    '''
    new_spk = spk.copy()

    n_tri = spk.n_trials
    n_cells = spk.n_units()
    tri_idx = np.arange(n_tri)
    np.random.shuffle(tri_idx)

    has_timestamps = spk.timestamps is not None
    has_waveforms = spk.waveform is not None

    if drop_timestamps:
        new_spk.timestamps = None
        has_timestamps = False
    if drop_waveforms:
        new_spk.waveform = None
        has_waveforms = False

    for cell_idx in range(n_cells):
        tri_limits, tri_id = _get_trial_boundaries(spk, cell_idx)
        n_spikes = np.diff(tri_limits)

        start_idx = 0
        for ix, tri in enumerate(tri_idx):
            pos = np.where(tri_id == tri)[0]
            if len(pos) > 0:
                pos = pos[0]
                limits = tri_limits[pos:pos + 2]
                n_spk = n_spikes[pos]

                slc = slice(start_idx, start_idx + n_spk)
                new_spk.trial[cell_idx][slc] = ix
                new_spk.time[cell_idx][slc] = (
                    spk.time[cell_idx][limits[0]:limits[1]]
                )

                if has_timestamps:
                    new_spk.timestamps[cell_idx][slc] = (
                        spk.timestamps[cell_idx][limits[0]:limits[1]]
                    )

                if has_waveforms:
                    new_spk.waveform[cell_idx][slc] = (
                        spk.waveform[cell_idx][limits[0]:limits[1]]
                    )

                start_idx += n_spk
    return new_spk


# TODO - if other sorters are used, alignment point (sample_idx) for the
#        spike waveforms should be saved somewhere in spk and used here.
def infer_waveform_polarity(spk, cell_idx, threshold=1.75, baseline_range=50,
                            rich_output=False):
    """Decide whether waveform polarity is positive, negative or unknown.

    This may be useful to detect units/clusters with bad alignment.
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
    rich_output : bool
        If True, return a dictionary with the following fields:
        * 'type' : 'positive' or 'negative' or 'unknown'
        * 'min_peak' : minimum peak value
        * 'max_peak' : maximum peak value
        * 'min_idx' : index of the minimum peak
        * 'max_idx' : index of the maximum peak
        * 'align_idx' : index of the alignment point
        * 'align_sign' : polarity of the alignment point (-1 or 1)

    Returns
    -------
    unit_type : str | dict
        Polarity label for the waveform. Either ``'positive'``, ``'negative'``
        or ``'unknown'``. If ``rich_output`` is True, a dictionary with
        multiple fields is returned (see description of ``rich_output``
        argument).
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

    if not rich_output:
        return unit_type
    else:
        align_which = 1 - further_away
        align_idx = [min_val_idx, max_val_idx][align_which]
        align_sign = [-1, 1][align_which]
        output = {'type': unit_type, 'min_peak': min_val, 'max_peak': max_val,
                  'min_idx': min_val_idx, 'max_idx': max_val_idx,
                  'align_idx': align_idx, 'align_sign': align_sign}
        return output


def _realign_waveforms(waveforms, pad_nans=False, reject=True):
    '''Realign waveforms. Used in ``realign_waveforms()`` function.'''
    mean_wv = np.nanmean(waveforms, axis=0)
    min_idx, max_idx = np.argmin(mean_wv), np.argmax(mean_wv)

    if min_idx < max_idx:
        waveforms *= -1
        mean_wv *= -1
        min_idx, max_idx = max_idx, min_idx

    # checking slope
    # --------------
    if reject:
        slope = np.nansum(np.diff(waveforms[:, :max_idx], axis=1), axis=1)
        bad_slope = slope < 0

    # realigning
    # ----------
    spike_max = np.argmax(waveforms, axis=1)
    new_waveforms = np.empty(waveforms.shape)
    new_waveforms.fill(np.nan)

    unique_mx = np.unique(spike_max)

    if reject:
        n_samples = waveforms.shape[1]
        max_dist = int(n_samples / 5)
        dist_to_peak = np.abs(spike_max - max_idx)
        bad_peak_dist = dist_to_peak > max_dist

        unique_mx = unique_mx[np.abs(unique_mx - max_idx) <= max_dist]

    for uni_ix in unique_mx:
        diff_idx = max_idx - uni_ix
        spk_msk = spike_max == uni_ix

        if diff_idx == 0:
            new_waveforms[spk_msk, :] = waveforms[spk_msk, :]
        elif diff_idx > 0:
            # individual peak too early
            new_waveforms[spk_msk, diff_idx:] = waveforms[spk_msk, :-diff_idx]

            if not pad_nans:
                new_waveforms[spk_msk, :diff_idx] = (
                    waveforms[spk_msk, [0]][:, None])
        else:
            # individual peak too late
            new_waveforms[spk_msk, :diff_idx] = waveforms[spk_msk, -diff_idx:]

            if not pad_nans:
                new_waveforms[spk_msk, diff_idx:] = (
                    waveforms[spk_msk, [diff_idx - 1]][:, None])

    waveforms_to_reject = (np.where(bad_slope | bad_peak_dist)[0]
                           if reject else None)

    return new_waveforms, waveforms_to_reject


def realign_waveforms(spk, picks=None, min_spikes=10, reject=True):
    '''Realign single waveforms compared to average waveform. Works in place.

    Parameters
    ----------
    spk :  pylabianca.Spikes | pylabianca.SpikeEpochs
        Spikes or SpikeEpochs object.
    picks : int | str | list-like of int | list-like of str
        The units to realign waveforms for.
    min_spikes : int
        Minimum number of spikes to try realigning the waveform.
    reject : bool
        Also remove waveforms and
    '''
    picks = _deal_with_picks(spk, picks)
    for cell_idx in picks:
        waveforms = spk.waveform[cell_idx]
        if waveforms is not None and len(waveforms) > min_spikes:
            waveforms, reject_idx = _realign_waveforms(waveforms)
            spk.waveform[cell_idx] = waveforms

            # reject spikes
            # TODO: could be made a separate function one day
            n_reject = len(reject_idx)
            if n_reject > 0:
                msg = (f'Removing {n_reject} bad waveforms for cell '
                       f'{spk.cell_names[cell_idx]}.')
                print(msg)

                spk.waveform[cell_idx] = np.delete(
                    spk.waveform[cell_idx], reject_idx, axis=0)
                spk.timestamps[cell_idx] = np.delete(
                    spk.timestamps[cell_idx], reject_idx)


def _get_trial_boundaries(spk, cell_idx):
    n_spikes = len(spk.trial[cell_idx])
    if n_spikes > 0:
        trial_boundaries = np.where(np.diff(spk.trial[cell_idx]))[0] + 1
        trial_boundaries = np.concatenate(
            [[0], trial_boundaries, [n_spikes]])
        tri_num = spk.trial[cell_idx][trial_boundaries[:-1]]
    else:
        trial_boundaries, tri_num = np.array([]), np.array([])

    return trial_boundaries, tri_num


def _get_cellinfo(inst):
    '''Obtain the cellinfo dataframe from multiple input types.'''
    from .spikes import Spikes, SpikeEpochs
    spike_objects = (Spikes, SpikeEpochs)

    if isinstance(inst, spike_objects):
        cellinfo = inst.cellinfo
    elif isinstance(inst, pd.DataFrame):
        cellinfo = inst
    else:
        msg = ('``inst`` has to be a Spikes, SpikeEpochs, xarray, '
               'DataArray or a pandas DataFrame object.')
        try:
            import xarray as xr
            if isinstance(inst, xr.DataArray):
                cellinfo = cellinfo_from_xarray(inst)
            else:
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg)

    if cellinfo is None:
        raise ValueError('No cellinfo found in the provided object.')

    return cellinfo


def find_cells(inst, not_found='error', more_found='error', **features):
    '''Find cell indices that fullfil search criteria.

    Parameters
    ----------
    inst: pylabianca.Spikes | pylabianca.SpikeEpochs | xarray.DataArray | pandas.DataFrame
        Object containing cellinfo dataframe.
    not_found: str
        Whether to error (``'error'``, default), warn (``'warn'``) or ignore
        (``'ignore'``) when some search items were not found.
    more_found: str
        Whether to error (``'error'``, default), warn (``'warn'``) or ignore
        (``'ignore'``) when some search items were found multiple times.
    **features:
        Keyword argument with search criteria. Keys refer to column names in
        the cellinfo dataframe and values are the values to search for.

    Returns
    -------
    cell_idx: np.ndarray
        Array of cell indices that match the search criteria.
    '''
    from numbers import Number
    _check_str_options(not_found, 'not_found')
    _check_str_options(more_found, 'more_found')

    cellinfo = _get_cellinfo(inst)
    feature_names = list(features.keys())
    n_features = len(feature_names)

    # make sure is feature is present in cellinfo
    cellinfo_columns = cellinfo.columns.tolist()
    for name in feature_names:
        if name not in cellinfo_columns:
            raise ValueError(f'Feature "{name}" is not present in the '
                             'cellinfo DataFrame')

        if isinstance(features[name], (Number, str)):
            features[name] = np.array([features[name]])
        elif isinstance(features[name], (list, tuple)):
            features[name] = np.array(features[name])

    cell_idx = list()
    n_comparisons = np.array([len(val) for val in features.values()])
    max_comp = n_comparisons.max()
    if n_features > 1:
        # ignore length-1 features when comparing number of search elements
        length_one_mask = n_comparisons == 1
        comp_match = (n_comparisons[~length_one_mask] == max_comp).all()

        if not comp_match:
            raise ValueError('Number of elements per search feature has to be '
                             'the same across all search features (with the '
                             'exception of length one features, which can be '
                             'easily tiled to match the rest).')

        # if some search elements are length-1, tile them to the correct length
        one_len_features = np.array(feature_names)[length_one_mask]
        for key in one_len_features:
            features[key] = np.tile(features[key], max_comp)

    masks = list()
    for key, val in features.items():
        msk = cellinfo[key].values[:, None] == val[None, :]
        masks.append(msk)
    masks = np.stack(masks, axis=2)
    match_all = masks.all(axis=2)
    row_idx, col_idx = np.where(match_all)

    if len(col_idx) > max_comp:
        msg = 'Found more than one match for some search elements.'
        _raise_error_warn_or_ignore(msg, more_found)
    elif len(col_idx) < n_comparisons[0]:
        msg = 'Could not find any match for some search elements.'
        _raise_error_warn_or_ignore(msg, not_found)

    return row_idx


def _check_str_options(arg_val, arg_name,
                       good_values=('error', 'warn', 'ignore')):
    if not isinstance(arg_val, str) or arg_val not in good_values:
        raise ValueError(f'"{arg_name}" has to be one of: {good_values}. '
                         f'Got: {arg_val}.')


def _raise_error_warn_or_ignore(msg, action):
    if action == 'ignore':
        pass
    elif action == 'error':
        raise ValueError(msg)
    elif action == 'warn':
        warn(msg)


def read_drop_info(path):
    '''Reads (channels, cluster id) pairs to drop from a text file.

    The text file should follow a structure:
    channel_name1: [cluster_id1, cluster_id2, ...]
    channel_name2: [cluster_id1, cluster_id2, ...]

    Parameters
    ----------
    path : str
        Path to the text file.

    Returns
    -------
    to_drop : list
        List of (channel, cluster_id) tuples representing all such pairs
        read from the text file.
    '''
    # read merge info
    with open(path) as file:
        text = file.readlines()

    # drop info is organized into channels / cluster ids
    to_drop = list()
    for line in text:
        channel = line.split(', ')[0]
        idx1, idx2 = line.find('['), line.find(']') + 1
        clusters = eval(line[idx1:idx2])
        for cluster in clusters:
            to_drop.append((channel, cluster))

    return to_drop


def drop_cells_by_channel_and_cluster_id(spk, to_drop):
    '''Works in place!'''
    # find cell idx by channel + cluster ID
    cell_idx = list()
    clusters, channels = zip(*to_drop)
    cell_idx = find_cells(spk, cluster=clusters, channel=channels)
    spk.drop_cells(cell_idx)


def get_data_path():
    home_dir = os.path.expanduser('~')
    data_dir = 'pylabianca_data'
    full_data_dir = op.join(home_dir, data_dir)
    has_data_dir = op.exists(full_data_dir)

    if not has_data_dir:
        os.mkdir(full_data_dir)

    return full_data_dir


def get_fieldtrip_data():
    import pooch

    data_path = get_data_path()
    ft_url = ('https://download.fieldtriptoolbox.org/tutorial/spike/p029_'
              'sort_final_01.nex')
    known_hash = ('4ae4ed2a9613cde884b62d8c5713c418cff5f4a57c8968a3886'
                  'db1e9991a81c9')
    fname = pooch.retrieve(
        url=ft_url, known_hash=known_hash,
        fname='p029_sort_final_01.nex', path=data_path
    )
    return fname


def get_test_data_link():
    dropbox_lnk = ('https://www.dropbox.com/scl/fo/757tf3ujqga3sa2qocm4l/h?'
                   'rlkey=mlz44bcqtg4ds3gsc29b2k62x&dl=1')
    return dropbox_lnk


def download_test_data():
    # check if test data exist
    data_dir = get_data_path()
    check_files = [
        'ft_spk_epoched.mat', 'monkey_stim.csv',
        'p029_sort_final_01_events.mat',
        op.join('test_osort_data', 'sub-U04_switchorder',
                'CSCA130_mm_format.mat'),
        op.join('test_neuralynx', 'sub-U06_ses-screening_set-U6d_run-01_ieeg',
                'CSC129.ncs')
    ]

    if all([op.isfile(op.join(data_dir, f)) for f in check_files]):
        return

    import pooch
    import zipfile

    # set up paths
    fname = 'temp_file.zip'
    download_link = get_test_data_link()

    # download the file
    hash = None
    pooch.retrieve(url=download_link, known_hash=hash,
                   path=data_dir, fname=fname)

    # unzip and extract
    # TODO - optionally extract only the missing files
    destination = op.join(data_dir, fname)
    zip_ref = zipfile.ZipFile(destination, 'r')
    zip_ref.extractall(data_dir)
    zip_ref.close()

    # remove the zipfile
    os.remove(destination)


def has_numba():
    """Check if numba is available."""
    try:
        from numba import jit
        return True
    except ImportError:
        return False


def has_elephant():
    '''Test if elephant is available.'''
    try:
        import elephant
        return True
    except ImportError:
        return False


def has_datashader():
    '''Test if datashader is available.'''
    try:
        import datashader
        return True
    except ImportError:
        return False


def create_random_spikes(n_cells=4, n_trials=25, n_spikes=(10, 21),
                         **args):
    '''Create random spike data. Mostly useful for testing.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    n_trials : int
        Number of trials. If ``None`` or 0 then Spikes object is returned.
    n_spikes : int | tuple
        Number of spikes. If tuple then the first element is the minimum
        number of spikes and the second element is the maximum number of
        spikes.
    args : dict
        Additional arguments are passed to the Spikes / SpikeEpochs object.

    Returns
    -------
    spikes : Spikes | SpikeEpochs
        Spike data object.
    '''
    from .spikes import SpikeEpochs, Spikes

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
            tms = np.sort(tms)
            this_tim.append(tms)

        this_tim = np.concatenate(this_tim)
        times.append(this_tim)

        if return_epochs:
            this_tri = np.concatenate(this_tri)
            trials.append(this_tri)

    if return_epochs:
        return SpikeEpochs(times, trials, time_limits=(tmin, tmax), **args)
    else:
        if 'sfreq' not in args:
            args['sfreq'] = 10_000

        return Spikes(times, **args)


def is_list_or_array(obj, dtype=None):
    if isinstance(obj, list):
        return True
    return is_array(obj, dtype=dtype)


def is_array(obj, dtype=None):
    if isinstance(obj, np.ndarray):
        if dtype is None:
            return True
        dtype = (dtype,) if not isinstance(dtype, tuple) else dtype
        return any([np.issubdtype(obj.dtype, dtp) for dtp in dtype])
    return False


def is_list_of_non_negative_integer_arrays(this_list, error_str):
    for cell_values in this_list:
        if len(cell_values) > 0:
            if not (np.issubdtype(cell_values.dtype, np.integer)
                    and cell_values.min() >= 0):
                raise ValueError(error_str)


def is_iterable_of_strings(this_list):
    return all([isinstance(x, str) for x in this_list])


def _validate_spike_epochs_input(time, trial):
    '''Validate input for SpikeEpochs object.'''

    # both time and trial have to be lists ...
    if not (is_list_or_array(time, dtype=np.object_)
            and is_list_or_array(trial, dtype=np.object_)):
        raise ValueError('Both time and trial have to be lists or object '
                         'arrays.')

    # ... of the same length ...
    if len(time) != len(trial):
        raise ValueError('Length of time and trial lists must be the same.')

    # ... and all elements have to be numpy arrays
    if not all([isinstance(cell_time, np.ndarray) for cell_time in time]):
        raise ValueError('All elements of time list must be numpy arrays.')
    if not all([isinstance(cell_trial, np.ndarray) for cell_trial in trial]):
        raise ValueError('All elements of trial list must be numpy arrays.')

    # all corresponding time and trial arrays have to have the same length
    if not all([len(time[ix]) == len(trial[ix]) for ix in range(len(time))]):
        raise ValueError('All time and trial arrays must have the same length.')

    # trial arrays have to contain non-negative integers
    error_str = 'Trial list of arrays must contain non-negative integers.'
    is_list_of_non_negative_integer_arrays(trial, error_str)


def _validate_spikes_input(times):
    '''Validate input for SpikeEpochs object.'''

    # timestamps have to be lists ...
    if not is_list_or_array(times, dtype=np.object_):
        raise ValueError('Timestamps have to be list or object array.')

    # ... and all elements have to be numpy arrays
    if not all([isinstance(cell_times, np.ndarray) for cell_times in times]):
        raise ValueError('All elements of timestamp list must be numpy '
                         'arrays.')

    # timestamp arrays have to contain non-negative integers
    error_str = 'Timestamp lists of arrays must contain integers or floats.'
    if not all(is_array(obj, dtype=(np.integer, np.floating))
               for obj in times):
        raise ValueError(error_str)

def _handle_cell_names(cell_names, time):
    if cell_names is None:
        n_cells = len(time)
        cell_names = np.array(['cell{:03d}'.format(idx)
                               for idx in range(n_cells)])
    else:
        try:
            str_type = np.unicode_
        except AttributeError:
            str_type = np.str_

        if not is_list_or_array(cell_names, dtype=(str_type, np.object_)):
            raise ValueError('cell_names has to be list or object array.')
        if not is_iterable_of_strings(cell_names):
            raise ValueError('All elements of cell_names have to be strings.')
        cell_names = np.asarray(cell_names)
        equal_len = len(cell_names) == len(time)
        if not equal_len:
            raise ValueError('Length of cell_names has to be equal to the '
                             'length of list of time arrays.')
    return cell_names


# - [ ] does not have to be SpikeEpochs object, can be Spikes
def _validate_cellinfo(spk, cellinfo):
    '''Validate cellinfo input for SpikeEpochs object.'''
    if cellinfo is not None:
        if not isinstance(cellinfo, pd.DataFrame):
            raise ValueError('cellinfo has to be a pandas DataFrame.')
        if cellinfo.shape[0] != spk.n_units():
            raise ValueError('Number of rows in cellinfo has to be equal to '
                             'the number of cells in the SpikeEpochs object.')
        if not (cellinfo.index == np.arange(spk.n_units())).all():
            warn('cellinfo index does not match cell indices in the '
                 'SpikeEpochs object. Resetting the index.')
            cellinfo = cellinfo.reset_index(drop=True)

    return cellinfo


def xr_find_nested_dims(arr, dim_name):
    names = list()
    coords = list(arr.coords)

    if isinstance(dim_name, tuple):
        for dim in dim_name:
            coords.remove(dim)
        sub_dim = dim_name
    else:
        coords.remove(dim_name)
        sub_dim = (dim_name,)

    for coord in coords:
        if arr.coords[coord].dims == sub_dim:
            names.append(coord)

    return names


# CONSIDER: ses_name -> ses_coord ?
def assign_session_coord(arr, ses, dim_name='cell', ses_name='session'):
    n_cells = len(arr.coords[dim_name])
    sub_dim = [ses] * n_cells
    arr = arr.assign_coords({ses_name: (dim_name, sub_dim)})
    return arr


# CONSIDER: ses_name -> ses_coord ?
def dict_to_xarray(data, dim_name='cell', select=None, ses_name='sub'):
    '''Convert dictionary to xarray.DataArray.

    Parameters
    ----------
    data : dict
        Dictionary with xarray data to concatenate. Keys are subject / session
        names and values are xarrays.
    dim_name : str
        Name of the dimension to concatenate along. Defaults to ``'cell'``.
        This dimension is also enriched with subject / session information from
        the dictionary keys.
    select : dict | None
        Trial selection query. If not None, select is passed to .query() method
        of the xarray. This can be useful to select only specific data from the
        xarray, which can be difficult to do after concatenation (after
        concatenation some coordinates may become multi-dimensional and
        querying would raise an error "Unlabeled multi-dimensional array cannot
        be used for indexing").
    ses_name : str
        Name of the subject / session coordinate that will be automatically
        added to the concatenated dimension from the dictionary keys. Defaults
        to ``'sub'``.

    Returns
    -------
    arr : xarray.DataArray
        DataArray with data from the dictionary.
    '''
    import xarray as xr

    assert isinstance(data, dict)
    keys = list(data.keys())
    all_xarr = [isinstance(data[sb], (xr.DataArray, xr.Dataset))
                for sb in keys]
    assert all(all_xarr)

    if (select is not None) and (not isinstance(select, dict)):
        select = {'trial': select}

    use_coords = None
    arr_list = list()
    different_coords = False
    for key, arr in data.items():
        if select is not None:
            if select is not None and arr.name is None:
                arr.name = 'data'

            arr = arr.query(select)

            # if trial was in select dict, then we should reset trial indices
            if 'trial' in select:
                arr = arr.reset_index('trial', drop=True)

        # add subject / session information to the concatenated dimension
        arr = assign_session_coord(
            arr, key, dim_name=dim_name, ses_name=ses_name)

        # check if coordinates are shared
        if use_coords is None:
            use_coords = set(list(arr.coords))
        else:
            coords = set(list(arr.coords))
            use_coords = use_coords & coords
            if not different_coords and len(coords) != len(use_coords):
                different_coords = True

        arr_list.append(arr)

    # drop coordinates that are not shared
    if different_coords:
        for idx, arr in enumerate(arr_list):
            drop_coords = set(list(arr.coords)) - use_coords
            arr_list[idx] = arr.drop_vars(drop_coords)

    arr = xr.concat(arr_list, dim=dim_name)
    return arr


# CONSIDER: ses_name -> ses_coord ?
def xarray_to_dict(xarr, ses_name='sub', reduce_coords=True,
                   ensure_correct_reduction=True):
    '''Convert multi-session xarray to dictionary of session -> xarray pairs.

    Note, that it is assumed that each session is a contiguous block in the
    xarray along the cell dimension.

    Parameters
    ----------
    xarr : xarray.DataArray
        Multi-session DataArray.
    ses_name : str
        Name of the session coordinate. Defaults to ``'sub'``.
    reduce_coords : bool
        If True, reduce coordinates were turned to cell x trial coordinates
        when concatenating different session xarrays. This happens when
        the order of the trial metadata (additional trial coordinates like
        condition or response correctness) is different across concatenated
        sessions. To keep these original trial metadata they have to be turned
        to cell x trial format. When splitting back to dictionary of session
        xarrays, this reduction can be undone. Defaults to True.
    ensure_correct_reduction : bool
        If True, ensure that the coord reduction is correct: check that,
        indeed, all within-session trial metadata that is cell x trial is
        the same across cells. See ``reduce_coords`` argument for more
        information. Defaults to True, but can be slow, so set to False if
        you are sure that all within-session trial metadata can be reduced
        from ``cell x trial`` to just trial.

    Returns
    -------
    xarr_dct : dict
        Dictionary with session names as keys and xarrays as values.
    '''
    xarr_dct = dict()

    sessions, ses_idx = np.unique(xarr.coords[ses_name].values, return_index=True)

    sort_idx = np.argsort(ses_idx)
    sessions = sessions[sort_idx]
    ses_idx = ses_idx[sort_idx]
    ses_idx = np.append(ses_idx, xarr.cell.shape[0])

    for idx, ses in enumerate(sessions):
        arr = xarr.isel(cell=slice(ses_idx[idx], ses_idx[idx + 1]))
        if reduce_coords:
            new_coords = dict()
            drop_coords = list()
            if 'cell' in arr.coords and 'trial' in arr.coords:
                nested_coords = xr_find_nested_dims(arr, ('cell', 'trial'))
            else:
                nested_coords = list()

            for coord in nested_coords:
                these_coords = arr.coords[coord].values
                one_cell = these_coords[[0]]
                if ensure_correct_reduction:
                    cmp = one_cell == these_coords
                    if cmp.all():
                        drop_coords.append(coord)
                        new_coords[coord] = ('trial', one_cell[0])
                else:
                    drop_coords.append(coord)
                    new_coords[coord] = ('trial', one_cell[0])

            if len(drop_coords) > 0:
                arr = arr.drop_vars(drop_coords)
                arr = arr.assign_coords(new_coords)

        xarr_dct[ses] = arr

    return xarr_dct


def find_index(vec, vals):
    if not isinstance(vals, (list, tuple, np.ndarray)):
        vals = [vals]

    vals = np.asarray(vals)
    ngb = np.array([-1, 0])
    idxs = np.searchsorted(vec, vals)

    test_idx = idxs[None, :] + ngb[:, None]
    closest_idx = np.abs(vec[test_idx] - vals[None, :]).argmin(axis=0)
    idxs += ngb[closest_idx]

    return idxs


def cellinfo_from_xarray(xarr):
    '''
    Extract cell information (cellinfo) dataframe from xarray.

    Parameters
    ----------
    xarr : xarray.DataArray
        DataArray to use. Must contain cell dimension.

    Returns
    -------
    cellinfo : pd.DataFrame | None
        DataFrame with cell information. If there are multiple cell coordinates
        in the xarray, the DataFrame will have multiple columns. If there are
        no cell coordinates, None is returned.
    '''
    cell_dims = xr_find_nested_dims(xarr, 'cell')

    if len(cell_dims) > 1:
        cellinfo = dict()
        for dim in cell_dims:
            cellinfo[dim] = xarr.coords[dim].values
        cellinfo = pd.DataFrame(cellinfo)
    else:
        cellinfo = None

    return cellinfo


def parse_sub_ses(sub_ses, remove_sub_prefix=True, remove_ses_prefix=True):
    """Parse subject and session from a BIDS-like string."""
    if remove_sub_prefix:
        sub_ses = sub_ses.replace('sub-', '')
    if remove_ses_prefix:
        sub_ses = sub_ses.replace('ses-', '')

    if '_' in sub_ses:
        sub, ses = sub_ses.split('_')
    else:
        sub, ses = sub_ses, None

    return sub, ses


# TODO: change name to something more descriptive like "select_data"?
# CONSIDER: ses_name -> ses_coord ?
# CONSIDER: change the loop to use .groupby() xarr method instead of _get_arr
#           (might be faster)
def extract_data(xarr_dict, df, sub_col='sub', ses_col=None, ses_name='sub',
                 df2xarr=None):
    '''Extract data from xarray dictionary using a dataframe.

    Parameters
    ----------
    xarr_dict : dict | xarray.DataArray
        Dictionary with xarrays or one xarray.DataArray.
    df : pandas.DataFrame
        DataFrame with selection properties.
    sub_col : str
        Name of the column in the DataFrame that contains subject / session
        information.
    ses_col : str | None
        Name of the column in the DataFrame that contains session information.
    df2xarr : dict | None
        Dictionary that maps DataFrame columns to xarray coordinates. If None,
        the default is ``{'label': 'region'}``.

    Returns
    -------
    xarr_out : dict of xarray.DataArray | xarray.DataArray
        Dictionary with selected xarray cells.
    row_indices : np.ndarray
        Array with indices of rows.
    '''
    import xarray as xr
    assert isinstance(xarr_dict, (dict, xr.DataArray))
    has_dict = isinstance(xarr_dict, dict)

    if df2xarr is None:
        df2xarr = {'label': 'region'}

    if has_dict:
        keys = list(xarr_dict.keys())
        xarr_out = dict()
    else:
        keys = pd.unique(xarr_dict.coords[ses_name].values)
        xarr_out = list()

    # TODO - check for sub / ses consistency and raise / warn
    #        instead of doing too much magic
    remove_sub_prefix = 'sub-' in df[sub_col].values[0]
    if ses_col is not None:
        remove_ses_prefix = 'ses-' in df[ses_col].values[0]
    else:
        remove_ses_prefix = False

    # TODO - check for sub / ses consistency and raise / warn
    #        instead of doing too much magic
    if remove_sub_prefix or remove_ses_prefix:
        df = df.copy()
        if remove_sub_prefix:
            df[sub_col] = df[sub_col].str.replace('sub-', '')
        if remove_ses_prefix:
            df[ses_col] = df[ses_col].str.replace('ses-', '')

    row_indices = list()
    for key in keys:
        sub, ses = parse_sub_ses(key, remove_sub_prefix=remove_sub_prefix,
                                 remove_ses_prefix=remove_ses_prefix)
        df_sel = df.query(f'{sub_col} == "{sub}"')
        if ses is not None and ses_col is not None:
            df_sel = df_sel.query(f'{ses_col} == "{ses}"')

        xarr = _get_arr(xarr_dict, key, ses_name=ses_name)

        n_cells = len(xarr.coords['cell'])
        mask_all = np.zeros(n_cells, dtype=bool)
        row_per_unit = np.zeros(n_cells, dtype=int)

        for row_idx in df_sel.index:
            mask_this = np.ones(n_cells, dtype=bool)
            for df_col, xarr_col in df2xarr.items():
                mask = (xarr.coords[xarr_col].values
                        == df_sel.loc[row_idx, df_col])
                mask_this &= mask

            row_per_unit[mask_this] = row_idx
            mask_all |= mask_this

        xarr_sel = xarr.sel(cell=mask_all)
        row_per_unit = row_per_unit[mask_all]

        row_indices.append(row_per_unit)
        if has_dict:
            xarr_out[key] = xarr_sel
        else:
            xarr_out.append(xarr_sel)

    row_indices = np.concatenate(row_indices)
    if not has_dict:
        xarr_out = xr.concat(xarr_out, dim='cell')

    return xarr_out, row_indices


def _get_arr(arr, sub_ses, ses_name='sub'):
    import xarray as xr
    if isinstance(arr, dict):
        arr = arr[sub_ses]
    elif isinstance(arr, xr.DataArray):
        arr = arr.query({'cell': f'{ses_name} == "{sub_ses}"'})
    return arr


# TODO: stimulus selectivity should be added to the xarray -
#       it can be done as cell x trial coordinate, this could be a function
#       in .selectivity module
# - [ ] better argument names:
#     -> is per_cell_query (per_cell_select etc.) even needed is we
#        have per_cell=True and pass to specific subfunction?
# ? option to pass the baseline calculated from a different period
def aggregate(frate, groupby=None, select=None, per_cell_query=None,
              zscore=False, baseline=False, per_cell=False):
    """
    Prepare spikes object for firing rate analysis.

    Parameters
    ----------
    frate : xarray.DataArray | dict
        Firing rate data. Output of ``.spike_rate()`` or ``.spike_density()``
        methods.
    groupby : str | False
        Condition by which trials are grouped and averaged.
    select : str | None
        A query to perform on the SpikeEpochs object to select trials
        fulfilling the query. For example ``'ifcorrect == True'`` will select
        those trials where ifcorrect column (whether response was correct) is
        True.
    per_cell_query : dict | None
        An xarray-compatible query to perform on the SpikeEpochs object
        separately for each cell. These are often properties related to cell
        selectivity, like whether the preferred stimulus is in memory etc.
    zscore : bool | tuple | xarray.DataArray
        Whether (and how) to zscore firing rate of each cell.  Defaults to
        ``False``, which does not zscore cell firing rate timecourses.
        If True, the whole array is used to calculate mean and standard
        deviation for zscoring. If tuple, it is interpreted as time range to
        use to calculate mean and standard deviation. If xarray.DataArray,
        this xarray is used to calculate mean and standard deviation.
    baseline : tuple | False
        If not ``False`` (default) - ``(tmin, tmax)`` tuple specifying time
        limits for baseline calculation. Baseline correction is performed
        after zscoring.
    per_cell : bool
        Whether to perform selection and groupby operations on each cell
        separately. This is much slower, but necessary when the selection is
        cell-specific, e.g. when selecting only cells that are
        stimulus-selective (then the preferred stimulus is cell-specific).

    Returns
    -------
    frates : xarray.DataArray
        Aggregated firing rate data.
    """
    import xarray as xr

    if isinstance(frate, dict):
        return _aggregate_dict(
            frate, groupby=groupby, select=select,
            per_cell_query=per_cell_query, zscore=zscore, baseline=baseline,
            per_cell=per_cell
        )
    else:
        msg = ('frate has to be an xarray.DataArray or dictionary of '
               'xarray.DataArrays.')
        assert isinstance(frate, xr.DataArray), msg

        _validate_xarray_for_aggregation(frate, groupby, per_cell)

    if per_cell_query is not None:
        per_cell = True

    if not per_cell:
        # integrate with per_cell approach
        frates = _aggregate_xarray(
                frate, groupby, zscore, select, baseline
            )
        return frates
    else:
        frates = list()
        n_cells = len(frate.cell)
        for cell_idx in range(n_cells):
            frate_cell = frate[cell_idx]

            if per_cell_query is not None:
                frate_cell = frate_cell.query(per_cell_query)

            frate_cell = _aggregate_xarray(
                frate_cell, groupby, zscore, select, baseline
            )
            frates.append(frate_cell)

        if len(frates) > 0:
            frates = xr.concat(frates, dim='cell')
            return frates
        else:
            return None


def _validate_xarray_for_aggregation(arr, groupby, per_cell):
    if groupby is not None:
        nested = xr_find_nested_dims(arr, ('cell', 'trial'))
        if groupby in nested and per_cell is False:
            raise ValueError(
                'When using `per_cell=False`, the groupby coordinate cannot be'
                ' cell x trial, it has to be a simple trial dimension '
                'coordinate. Complex cell x trial coordinates often arise when'
                ' the data is transformed from a dictionary of xarrays to one'
                ' concatenated xarray (using `pylabianca.utils.dict_to_xarray`'
                'or `xarray.concat()` directly) and the trial metadata is not '
                'the same across sessions (e.g. the order of conditions is '
                'different, which is common for non-pseudo-random experiments)'
                '. Use pylabianca.utils.xarray_to_dict to convert the data to '
                'a dictionary of xarrays and then perform the aggregation on '
                'the dictionary. Alternatively, if the cell x trial '
                'coordinates are inherent to each session (for example it'
                'represents cell-specific properties that vary by trial, like'
                ' whether the preferred stimulus was shown), you can perform '
                'the aggregation using `per_cell=True`.')


def _aggregate_xarray(frate, groupby, zscore, select, baseline):
    """Aggregate xarray.DataArray with firing rate data.

    The aggregation is performed within cells, i.e. the firing rate of each
    cell is averaged across trials. Optionally, the data can be zscored
    (separately for each cell) and baseline corrected.

    Parameters
    ----------
    frate : xarray.DataArray
        Firing rate data.
    groupby : str | list | None
        Dimension to groupby and average along.
    zscore : bool | tuple | xarray.DataArray
        Whether (and how) to zscore firing rate of each cell.  Defaults to
        ``False``, which does not zscore cell firing rate time courses.
        If True, the whole array is used to calculate mean and standard
        deviation for zscoring. If tuple, it is interpreted as time range to
        use to calculate mean and standard deviation. If xarray.DataArray,
        this xarray is used to calculate mean and standard deviation.
    select : str | None
        A query to perform on the DataArray to select trials fulfilling the
        query. For example ``'ifcorrect == True'`` will select those trials
        where ifcorrect trial coord (whether response was correct) is True.
    baseline : tuple | False
        If not ``False`` - ``(tmin, tmax)`` tuple specifying time limits of
        baseline correction. The baseline correction is performed after
        zscoring.

    Returns
    -------
    frate : xarray.DataArray
        Aggregated firing rate data.
    """

    if zscore:
        bsln = None if isinstance(zscore, bool) else zscore
        frate = zscore_xarray(frate, baseline=bsln)

    if select is not None:
        frate = frate.query(trial=select)

    if groupby:
        if isinstance(groupby, list):
            frate = nested_groupby_apply(frate, groupby)
        else:
            frate = frate.groupby(groupby).mean()
    else:
        # average all trials
        frate = frate.mean(dim='trial')

    if baseline:
        time_range = slice(baseline[0], baseline[1])
        bsln = frate.sel(time=time_range).mean(dim='time')
        frate -= bsln

    return frate


# CONSIDER: use flox instead?
def nested_groupby_apply(array, groupby, apply_fn=None):
    """Apply function to nested groupby.

    A hack from xarray github, posted by user https://github.com/hottwaj:
    https://github.com/pydata/xarray/issues/324#issuecomment-265462343

    Parameters
    ----------
    array : xarray.DataArray
        DataArray to groupby and apply function to.
    groupby : str | list | None
        Dimensions/variables to apply groupby operation to. Can be:
        * str: single variable is used to group by.
        * list: multiple variables are used to group by.
        * None: no groupby operation is performed.
    apply_fn : function
        Function to apply to grouped DataArray. If ``None``, the mean along
        'trial' dimension is used.

    Returns
    -------
    array : xarray.DataArray
        DataArray after groupby operations.
    """

    if apply_fn is None:
        # average over trial by default
        apply_fn = lambda arr: arr.mean(dim='trial')

    if groupby is None:
        return apply_fn(array)
    elif isinstance(groupby, str):
        groupby = [groupby]

    if len(groupby) == 1:
        return array.groupby(groupby[0]).apply(apply_fn)
    else:
        return array.groupby(groupby[0]).apply(
            nested_groupby_apply, groupby=groupby[1:], apply_fn=apply_fn)


# TODO: this could be changed and used with apply_dict / dict_apply
#       the dict apply function could have output='xarray' option
def _aggregate_dict(frates, groupby=None, select=None,
                    per_cell_query=None, zscore=False, baseline=False,
                    per_cell=False):
    import xarray as xr

    aggregated = list()
    keys = list(frates.keys())

    for key in keys:
        frate = frates[key]
        frate_agg = aggregate(
            frate, groupby=groupby, select=select,
            per_cell_query=per_cell_query, zscore=zscore, baseline=baseline,
            per_cell=per_cell
        )
        if frate_agg is not None:
            aggregated.append(frate_agg)
            # TODO: assign subject / session information coordinate
            #       - only if not already present ?

    aggregated = xr.concat(aggregated, dim='cell')
    return aggregated


def zscore_xarray(arr, groupby='cell', baseline=None):
    '''
    Z-score an xarray.DataArray.

    If the array contains `groupby` dimension (by default 'cell'), z-score
    each element of this dimension (each cell) separately.

    Parameters
    ----------
    arr : xarray.DataArray
        Data to z-score.
    groupby : str
        Dimension name to z-score separately.
    baseline : None | tuple | xarray.DataArray
        If None, the whole array is used to calculate mean and standard
        deviation. If tuple, the time range is used to calculate mean and
        standard deviation. If xarray.DataArray, this xarray is used to
        calculate mean and standard deviation.

    Returns
    -------
    arr : xarray.DataArray
        Z-scored data.
    '''
    import xarray as xr

    if baseline is None:
        baseline_arr = arr
    elif isinstance(baseline, tuple):
        assert len(baseline) == 2
        time_range = slice(*baseline)
        baseline_arr = arr.sel(time=time_range)
    elif isinstance(baseline, xr.DataArray):
        baseline_arr = baseline
    else:
        raise ValueError('baseline has to be None, tuple or xarray.DataArray.')

    has_cell_dim = (groupby in baseline_arr.dims
                    and len(baseline_arr.coords[groupby]) > 1)

    if not has_cell_dim:
        avg, std = baseline_arr.mean(), baseline_arr.std()
    else:
        dims = tuple(dim for dim in baseline_arr.dims if dim != groupby)
        avg, std = baseline_arr.mean(dim=dims), baseline_arr.std(dim=dims)

    arr = (arr - avg) / std
    return arr


def reset_trial_id(xarr_dict):
    """Reset trial IDs in xarray dictionary."""
    keys = list(xarr_dict.keys())
    for key in keys:
        this_arr = xarr_dict[key]
        n_tri = len(this_arr.coords['trial'].values)
        new_tri = np.arange(n_tri)
        this_arr.coords['trial'].values[:] = new_tri[:]
