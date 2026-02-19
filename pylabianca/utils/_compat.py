import warnings


def _warn_different_import_path(old_path, new_path):
    warnings.warn(f'`pylabianca.{old_path}` has moved to '
                  f'`pylabianca.{new_path}`. The old import path '
                  'will be removed in a future version.', FutureWarning,
                  stacklevel=4)


def dict_to_xarray(*args, **kwargs):
    '''This function has moved. Use ``pylabianca.analysis.dict_to_xarray``
    instead.'''

    # raise deprecation warning
    _warn_different_import_path('utils.dict_to_xarray',
                                'analysis.dict_to_xarray')
    from ..analysis import dict_to_xarray as _dict_to_xarray
    return _dict_to_xarray(*args, **kwargs)


def xarray_to_dict(*args, **kwargs):
    '''This function has moved. Use ``pylabianca.analysis.xarray_to_dict``
    instead.'''

    # raise deprecation warning
    _warn_different_import_path('utils.xarray_to_dict',
                                'analysis.xarray_to_dict')

    from ..analysis import xarray_to_dict as _xarray_to_dict
    return _xarray_to_dict(*args, **kwargs)


def spike_centered_windows(*args, **kwargs):
    '''This function has moved. Use
    ``pylabianca.analysis.spike_centered_windows`` instead.'''

    # raise deprecation warning
    _warn_different_import_path('utils.spike_centered_windows',
                                'analysis.spike_centered_windows')

    from ..analysis import spike_centered_windows as _spike_centered_windows
    return _spike_centered_windows(*args, **kwargs)


def shuffle_trials(*args, **kwargs):
    '''This function has moved. Use ``pylabianca.analysis.shuffle_trials``
    instead.'''

    # raise deprecation warning
    _warn_different_import_path('utils.shuffle_trials',
                                'analysis.shuffle_trials')

    from ..analysis import shuffle_trials as _shuffle_trials
    return _shuffle_trials(*args, **kwargs)


def read_drop_info(*args, **kwargs):
    '''This function has moved. Use ``pylabianca.io.read_drop_info`` instead.'''

    # raise deprecation warning
    _warn_different_import_path('utils.read_drop_info',
                                'analysis.read_drop_info')

    from ..io import read_drop_info as _read_drop_info
    return _read_drop_info(*args, **kwargs)
