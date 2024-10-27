import numpy as np
from .base import _deal_with_picks


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
    import pandas as pd
    cell_dims = xr_find_nested_dims(xarr, 'cell')

    if len(cell_dims) > 1:
        cellinfo = dict()
        for dim in cell_dims:
            cellinfo[dim] = xarr.coords[dim].values
        cellinfo = pd.DataFrame(cellinfo)
    else:
        cellinfo = None

    return cellinfo


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
