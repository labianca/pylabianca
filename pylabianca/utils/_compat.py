import warnings


def dict_to_xarray(*args, **kwargs):
    '''This function has moved. Use ``pylabianca.analysis.dict_to_xarray``
    instead.'''

    # raise deprecation warning
    warnings.warn('This function has moved. Use `pylabianca.analysis.dict_'
                  'to_xarray` instead.', FutureWarning, stacklevel=2)

    from ..analysis import dict_to_xarray as _dict_to_xarray
    return _dict_to_xarray(*args, **kwargs)


def xarray_to_dict(*args, **kwargs):
    '''This function has moved. Use ``pylabianca.analysis.xarray_to_dict``
    instead.'''

    # raise deprecation warning
    warnings.warn('This function has moved. Use `pylabianca.analysis.xarray_'
                  'to_dict` instead.', FutureWarning, stacklevel=2)

    from ..analysis import xarray_to_dict as _xarray_to_dict
    return _xarray_to_dict(*args, **kwargs)


def spike_centered_windows(*args, **kwargs):
    '''This function has moved. Use
    ``pylabianca.analysis.spike_centered_windows`` instead.'''

    # raise deprecation warning
    warnings.warn('This function has moved. Use `pylabianca.analysis.spike_'
                  'centered_windows` instead.', FutureWarning, stacklevel=2)

    from ..analysis import spike_centered_windows as _spike_centered_windows
    return _spike_centered_windows(*args, **kwargs)


def shuffle_trials(*args, **kwargs):
    '''This function has moved. Use ``pylabianca.analysis.shuffle_trials``
    instead.'''

    # raise deprecation warning
    warnings.warn('This function has moved. Use `pylabianca.analysis.shuffle_'
                  'trials` instead.', FutureWarning, stacklevel=2)

    from ..analysis import shuffle_trials as _shuffle_trials
    return _shuffle_trials(*args, **kwargs)


def read_drop_info(*args, **kwargs):
    '''This function has moved. Use ``pylabianca.io.read_drop_info`` instead.'''

    # raise deprecation warning
    warnings.warn('This function has moved. Use `pylabianca.io.read_drop_'
                  'info` instead.', FutureWarning, stacklevel=2)

    from ..io import read_drop_info as _read_drop_info
    return _read_drop_info(*args, **kwargs)
