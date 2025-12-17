import warnings
import numpy as np
from .base import _deal_with_picks


# CONSIDER: spike_epochs and times could be optional arguments
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
          is a string indicating time window or None)
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
    has_trials = False
    if frate.ndim == 3:
        has_trials = True
        # CHECK: if data is 3D can cell_names be really None?
        n_trials = frate.shape[0] if cell_names is None else frate.shape[1]
    elif frate.ndim == 2:
        if cell_names is None:
            has_trials = True
            n_trials = frate.shape[0]
        else:
            if times_array:
                has_trials = False
                n_trials = 0
            else:
                has_trials = True
                n_trials = frate.shape[1]

    if has_trials:
        dimname = 'trial' if tri is None else 'spike'
        coords = {dimname: np.arange(n_trials)}
        dims = [dimname]
    else:
        coords = dict()
        dims = list()

    attrs = None
    if times_array:
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


def df_from_xarray_coords(xarr, dim):
    '''
    Extract xarray coordinate information as a dataframe.

    Parameters
    ----------
    xarr : xarray.DataArray
        DataArray to use.
    dim : str
        Dimension coordinates to extract.

    Returns
    -------
    df : pd.DataFrame | None
        DataFrame with coordinate information. If there are multiple
        coordinates for dimension ``dim`` in the xarray, the DataFrame will
        contain multiple columns. If there are no dimension coordinates,
        None is returned.
    '''
    import pandas as pd
    use_dims = find_nested_dims(xarr, dim)

    if len(use_dims) > 1:
        df = {dim: xarr.coords[dim].values for dim in use_dims}
        df = pd.DataFrame(df)
    else:
        df = None

    return df


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
    cellinfo = df_from_xarray_coords(xarr, 'cell')
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
        copy_coords = find_nested_dims(xarr_from, dimname)
    if len(copy_coords) > 0:
        coords = {coord: (dimname, xarr_from.coords[coord].values)
                  for coord in copy_coords}
        xarr_to = xarr_to.assign_coords(coords)
    return xarr_to


def find_nested_dims(arr, dim_name):
    names = list()
    coords = list(arr.coords)

    if isinstance(dim_name, tuple):
        for dim in dim_name:
            if dim in coords:
                coords.remove(dim)
        sub_dim = dim_name
    else:
        if dim_name in coords:
            coords.remove(dim_name)
        sub_dim = (dim_name,)

    for coord in coords:
        if arr.coords[coord].dims == sub_dim:
            names.append(coord)

    return names


def assign_session_coord(arr, ses, dim_name='cell', ses_coord='session',
                         ses_name=None):
    '''Assign a coordinate with session info to all cells.

    Parameters
    ----------
    arr : xarray.DataArray
        Input xarray.
    ses : str
        Session name to assign.
    dim_name : str
        Name of the dimension corresponding to cells.
    ses_name : str
        Name of the session coordinate to create.

    Returns
    -------
    arr : xarray.DataArray
        Xarray with session coordinate assigned.
    '''
    # deprecate ses_name in favor of ses_coord
    if ses_name is not None:
        ses_coord = ses_name
        warnings.warn('`ses_name` is deprecated and will be removed in a '
                      'future release. Use `ses_coord` instead.',
                      FutureWarning, stacklevel=2)

    # check dim_name
    if dim_name in arr.dims:
        n_cells = len(arr.coords[dim_name])
    elif dim_name in arr.coords:
        n_cells = 1
    else:
        raise ValueError(f'Could not find dim_name "{dim_name}" in arr.dims'
                         'or arr.coords.')

    sub_dim = [ses] * n_cells
    arr = arr.assign_coords({ses_coord: (dim_name, sub_dim)})
    return arr
