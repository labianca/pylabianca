import warnings


def dict_to_xarray(*args, **kwargs):
    '''This function has moved. Use ``pylabianca.analysis.dict_to_xarray``
    instead.'''

    # raise deprecation warning
    warnings.warn('`pylabianca.utils.dict_to_xarray` has moved to '
                  '`pylabianca.analysis.dict_to_xarray`. The old import path '
                  'will be removed in a future version.', FutureWarning, stacklevel=3)

    from ..analysis import dict_to_xarray as _dict_to_xarray
    return _dict_to_xarray(*args, **kwargs)


def xarray_to_dict(*args, **kwargs):
    '''This function has moved. Use ``pylabianca.analysis.xarray_to_dict``
    instead.'''

    # raise deprecation warning
    warnings.warn('`pylabianca.analysis.xarray_to_dict` has moved to '
                  '`pylabianca.analysis.xarray_to_dict`. instead. The old '
                  'import path will be removed in a future version.',
                  FutureWarning, stacklevel=3)

    from ..analysis import xarray_to_dict as _xarray_to_dict
    return _xarray_to_dict(*args, **kwargs)


def spike_centered_windows(*args, **kwargs):
    '''This function has moved. Use
    ``pylabianca.analysis.spike_centered_windows`` instead.'''

    # raise deprecation warning
    warnings.warn('`pylabianca.analysis.spike_centered_windows` has moved to '
                  '`pylabianca.analysis.spike_centered_windows`. instead. The old '
                  'import path will be removed in a future version.',
                  FutureWarning, stacklevel=3)

    from ..analysis import spike_centered_windows as _spike_centered_windows
    return _spike_centered_windows(*args, **kwargs)


def shuffle_trials(*args, **kwargs):
    '''This function has moved. Use ``pylabianca.analysis.shuffle_trials``
    instead.'''

    # raise deprecation warning
    warnings.warn('`pylabianca.analysis.shuffle_trials` has moved to '
                  '`pylabianca.analysis.shuffle_trials`. instead. The old '
                  'import path will be removed in a future version.',
                  FutureWarning, stacklevel=3)

    from ..analysis import shuffle_trials as _shuffle_trials
    return _shuffle_trials(*args, **kwargs)


def read_drop_info(*args, **kwargs):
    '''This function has moved. Use ``pylabianca.io.read_drop_info`` instead.'''

    # raise deprecation warning
    warnings.warn('`pylabianca.analysis.read_drop_info` has moved to '
                  '`pylabianca.analysis.read_drop_info`. instead. The old '
                  'import path will be removed in a future version.',
                  FutureWarning, stacklevel=3)


    from ..io import read_drop_info as _read_drop_info
    return _read_drop_info(*args, **kwargs)
