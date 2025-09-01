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


def _get_trial_boundaries(spk, cell_idx):
    '''Get trial boundaries for a given cell.'''
    return _get_trial_boundaries_array(spk.trial[cell_idx])


def _get_trial_boundaries_array(trials):
    n_spikes = len(trials)
    if n_spikes > 0:
        trial_boundaries = np.where(np.diff(trials))[0] + 1
        trial_boundaries = np.concatenate(
            [[0], trial_boundaries, [n_spikes]])
        tri_num = trials[trial_boundaries[:-1]]
    else:
        trial_boundaries, tri_num = np.array([]), np.array([])

    return trial_boundaries, tri_num


def _get_cellinfo(inst):
    '''Obtain the cellinfo dataframe from multiple input types.'''
    from ..spikes import Spikes, SpikeEpochs
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
            from .xarr import cellinfo_from_xarray

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
    from .validate import _check_str_options
    _check_str_options(not_found, 'not_found')
    _check_str_options(more_found, 'more_found')

    cellinfo = _get_cellinfo(inst)
    feature_names = list(features.keys())
    n_features = len(feature_names)

    # make sure every feature is present in cellinfo
    cellinfo_columns = cellinfo.columns.tolist()
    for name in feature_names:
        if name not in cellinfo_columns:
            raise ValueError(f'Feature "{name}" is not present in the '
                             'cellinfo DataFrame')

        if isinstance(features[name], (Number, str)):
            features[name] = np.array([features[name]])
        elif isinstance(features[name], (list, tuple)):
            features[name] = np.array(features[name])
        elif isinstance(features[name], pd.core.series.Series):
            features[name] = features[name].values

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


def _raise_error_warn_or_ignore(msg, action):
    if action == 'ignore':
        pass
    elif action == 'error':
        raise ValueError(msg)
    elif action == 'warn':
        warn(msg)


def drop_cells_by_channel_and_cluster_id(spk, to_drop):
    '''Works in place!'''
    # find cell idx by channel + cluster ID
    cell_idx = list()
    clusters, channels = zip(*to_drop)
    cell_idx = find_cells(spk, cluster=clusters, channel=channels)
    spk.drop_cells(cell_idx)


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


def reset_trial_id(xarr_dict):
    """Reset trial IDs in xarray dictionary."""
    keys = list(xarr_dict.keys())
    for key in keys:
        this_arr = xarr_dict[key]
        n_tri = len(this_arr.coords['trial'].values)
        new_tri = np.arange(n_tri)
        this_arr.coords['trial'].values[:] = new_tri[:]
