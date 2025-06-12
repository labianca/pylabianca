import numpy as np

from .utils import (
    _get_trial_boundaries, _deal_with_picks, find_index, parse_sub_ses)
from .utils.xarr import (
    find_nested_dims, _inherit_metadata, assign_session_coord)
from .utils.validate import _validate_xarray_for_aggregation


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


# ENH: speed up with numba sometime
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

            # VERSION: copy argument introduced in mne 1.6
            try:
                arr = arr.get_data(copy=False)
            except TypeError:
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
    all_windows = list()

    boundaries, trials = _get_trial_boundaries(spk, picks[0])
    for idx, tri_idx in enumerate(trials):
        sel = slice(boundaries[idx], boundaries[idx + 1])
        tms = spk.time[cell_idx][sel]

        closest_smp = find_index(time, tms)
        twins = closest_smp[:, None] + winlims
        good = ((twins >= lims[0]) & (twins <= lims[1])).all(axis=1)
        all_windows.append(twins[good])
        tri_is_ok[sel] = good

    window_idx = 0
    n_epochs = tri_is_ok.sum()
    spike_centered = np.zeros((n_epochs, len(ch_names), len(window_time)))
    for tri_idx, twins in zip(trials, all_windows):
        for twin in twins:
            sig_part = arr[tri_idx, :, twin[0]:twin[1]]
            spike_centered[window_idx] = sig_part
            window_idx += 1

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
        import pandas as pd
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
        ``False``, which does not zscore cell firing rate time courses.
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


def _aggregate_xarray(frate, groupby, zscore, select, baseline):
    """Aggregate xarray.DataArray with firing rate data.

    The aggregation is performed within cells, i.e. the firing rate of each
    cell is averaged across trials. Optionally, the data can be z-scored
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

    arr_list = list()
    all_coord_dims = {}  # coord_name -> dims
    coord_dtypes = {}    # coord_name -> preferred dtype

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

        for coord_name, coord in arr.coords.items():
            if coord_name not in all_coord_dims:
                all_coord_dims[coord_name] = coord.dims
                coord_dtypes[coord_name] = coord.dtype

        arr_list.append(arr)

    # Fill in missing coordinates with appropriate dummies
    for arr in arr_list:
        for coord_name, dims in all_coord_dims.items():
            if coord_name not in arr.coords:
                shape = tuple(arr.sizes[d] for d in dims)
                dtype = coord_dtypes[coord_name]
                missing = _get_missing_value(dtype)
                filler = np.full(shape, missing, dtype=dtype)
                arr.coords[coord_name] = (dims, filler)

    arr = xr.concat(arr_list, dim=dim_name, combine_attrs='override')
    return arr


def _get_missing_value(dtype):
    """Return a reasonable missing value for the given dtype."""
    if np.issubdtype(dtype, np.floating):
        return np.nan
    elif np.issubdtype(dtype, np.integer):
        return -1
    elif np.issubdtype(dtype, np.str_):
        return ''  # "missing" seemed to be a good choice, but it is not.
                   # Depending on the size of the string array, it can be too
                   # long and end up being truncated to "m", for example
    else:
        return None


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
                nested_coords = find_nested_dims(arr, ('cell', 'trial'))
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
