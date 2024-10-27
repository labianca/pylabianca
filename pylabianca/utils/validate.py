from warnings import warn
import numpy as np


def _check_str_options(arg_val, arg_name,
                       good_values=('error', 'warn', 'ignore')):
    if not isinstance(arg_val, str) or arg_val not in good_values:
        raise ValueError(f'"{arg_name}" has to be one of: {good_values}. '
                         f'Got: {arg_val}.')


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
    import pandas as pd

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


def _validate_xarray_for_aggregation(arr, groupby, per_cell):
    if groupby is not None:
        from .xarr import xr_find_nested_dims
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
