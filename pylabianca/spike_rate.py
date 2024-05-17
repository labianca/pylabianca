import numpy as np
from .utils import (_deal_with_picks, _turn_spike_rate_to_xarray,
                    _gauss_kernel_samples, _symmetric_window_samples)



# TODO: add n_jobs?
# CONSIDER wintype 'rectangular' vs 'gaussian'
# TODO: refactor (DRY: merge both loops into one?)
# TODO: better handling of numpy vs numba implementation
# TODO: consider adding `return_type` with `Epochs` option (mne object)
def compute_spike_rate(spk, picks=None, winlen=0.25, step=0.01, tmin=None,
                       tmax=None, backend='numpy'):
    '''Calculate spike rate with a running or static window.

    Parameters
    ----------
    spk : pylabianca.SpikeEpochs
        Spikes object to compute firing rate for.
    picks : int | listlike of int | None
        The neuron indices / names to use in the calculations. The default
        (``None``) uses all cells.
    winlen : float
        Length of the running window in seconds.
    step : float | bool
        The step size of the running window. If step is ``False`` then
        spike rate is not calculated using a running window but with
        a static one with limits defined by ``tmin`` and ``tmax``.
    tmin : float | None
        Time start in seconds. Default to trial start if ``tmin`` is ``None``.
    tmax : float | None
        Time end in seconds. Default to trial end if ``tmax`` is ``None``.
    backend : str
        Execution backend. Can be ``'numpy'`` or ``'numba'``.

    Returns
    -------
    frate : xarray.DataArray
            Xarray with following labeled dimensions: cell, trial, time.
    '''
    picks = _deal_with_picks(spk, picks)
    tmin = spk.time_limits[0] if tmin is None else tmin
    tmax = spk.time_limits[1] if tmax is None else tmax

    if isinstance(step, bool) and not step:

        times = f'{tmin} - {tmax} s'

        frate = list()
        cell_names = list()

        for pick in picks:
            frt = _compute_spike_rate_fixed(
                spk.time[pick], spk.trial[pick], [tmin, tmax],
                spk.n_trials)
            frate.append(frt)
            cell_names.append(spk.cell_names[pick])

    else:
        frate = list()
        cell_names = list()

        if backend == 'numpy':
            func = _compute_spike_rate_numpy
        elif backend == 'numba':
            from ._numba import _compute_spike_rate_numba
            func = _compute_spike_rate_numba
        else:
            raise ValueError('Backend can be only "numpy" or "numba".')

        for pick in picks:
            times, frt = func(
                spk.time[pick], spk.trial[pick], np.array([tmin, tmax]),
                spk.n_trials, winlen=winlen, step=step)
            frate.append(frt)
            cell_names.append(spk.cell_names[pick])

    frate = np.stack(frate, axis=0)
    frate = _turn_spike_rate_to_xarray(times, frate, spk,
                                        cell_names=cell_names)
    frate = _add_frate_info(frate)

    return frate


def _add_frate_info(arr, dep='rate'):
    arr.name = f'firing {dep}'
    arr.attrs['unit'] = "Hz"
    arr.attrs['coord_units'] = {'time': 's'}
    return arr


# ENH: speed up by using previous mask in the next step to pre-select spikes
# ENH: time limits per window could be calculated only once - for all units
def _compute_spike_rate_numpy(spike_times, spike_trials, time_limits,
                              n_trials, winlen=0.25, step=0.05):
    half_win = winlen / 2
    window_limits = np.array([-half_win, half_win])
    used_range = time_limits[1] - time_limits[0]
    n_steps = (used_range - winlen) / step + 1
    n_steps_int = int(n_steps)

    if n_steps - n_steps_int > 0.9:
        n_steps = n_steps_int + 1
    else:
        n_steps = n_steps_int

    times = np.arange(n_steps) * step + time_limits[0] + half_win
    frate = np.zeros((n_trials, n_steps))

    for step_idx in range(n_steps):
        win_lims = times[step_idx] + window_limits
        msk = (spike_times >= win_lims[0]) & (spike_times < win_lims[1])
        tri = spike_trials[msk]
        in_tri, count = np.unique(tri, return_counts=True)
        frate[in_tri, step_idx] = count / winlen

    return times, frate


# @numba.jit(nopython=True)
# currently raises warnings, and jit is likely not necessary here
# Encountered the use of a type that is scheduled for deprecation: type
# 'reflected list' found for argument 'time_limits' of function
# '_compute_spike_rate_fixed'
def _compute_spike_rate_fixed(spike_times, spike_trials, time_limits,
                              n_trials):

    winlen = time_limits[1] - time_limits[0]
    frate = np.zeros(n_trials)

    if len(spike_times) > 0:
        msk = (spike_times >= time_limits[0]) & (spike_times < time_limits[1])
        tri = spike_trials[msk]
        in_tri, count = np.unique(tri, return_counts=True)
        frate[in_tri] = count / winlen

    return frate


# TODO: add n_jobs
#       (use mne.parallel.parallel_func)
# TODO: check if time is symmetric wrt 0 (in most cases it should be as epochs
#       are constructed wrt specific event)
# TODO: consider an exact mode where the spikes are not transformed to raw /
#       binned representation (binned raster) and the gaussian kernel is placed
#       exactly where the spike is (`loc=spike_time`?) and evaluated
#       (maybe this is what is done by elephant?)
def _spike_density(spk, picks=None, winlen=0.3, gauss_sd=None, fwhm=None,
                   kernel=None, sfreq=500.):
    '''Calculates normal (constant) spike density.

    The density is computed by convolving the binary spike representation
    with a gaussian kernel.

    Parameters
    ----------
    spk : SpikeEpochs
        SpikeEpochs object.
    picks : array-like | None
        Indices or names of cells to use. If ``None`` all cells are used.
    winlen : float
        Length of the gaussian kernel in seconds. Default is ``0.3``.
        If ``gauss_sd`` is ``None`` the standard deviation of the gaussian
        kernel is set to ``winlen / 6``.
    gauss_sd : float | None
        Standard deviation of the gaussian kernel in seconds. If ``None``
        the standard deviation is set to ``winlen / 6``.
    fwhm : float | None
        Full width at half maximum of the gaussian kernel in seconds.
    kernel : array-like | None
        Kernel to use for convolution. If ``None`` the gaussian kernel is
        constructed from ``winlen`` and ``gauss_sd``.
    sfreq : float
        Sampling frequency (in Hz) of the spike density representation. Default
        is ``500``.
    '''
    from scipy.signal import oaconvolve

    kernel, trim = _setup_kernel(
        winlen=winlen, gauss_sd=gauss_sd, fwhm=fwhm, kernel=kernel,
        sfreq=sfreq
    )
    picks = _deal_with_picks(spk, picks)
    times, bin_rep = spk.to_raw(picks=picks, sfreq=sfreq)
    cnt_times = times[trim:-trim]

    cnt = oaconvolve(bin_rep, kernel[None, None, :], mode='valid')

    # FIX: for some reason in scipy.correlate we get a lot of close-to-zero
    #      numerical errors, that do not seem to be present if we do one
    #      cell-trial at a time, this needs to be investigated a bit more
    #      but now we just set them to zero
    noise_level = 1e-14

    mask = np.abs(cnt) < noise_level
    cnt[mask] = 0.

    return cnt_times, cnt


def _setup_kernel(winlen=0.3, gauss_sd=None, fwhm=None, kernel=None,
                  sfreq=500.):
    """
    Set up the gaussian kernel for the spike density calculation.

    Parameters
    ----------
    winlen : float
        Length of the gaussian kernel in seconds. Default is ``0.3``.
        If ``gauss_sd`` is ``None`` the standard deviation of the gaussian
        kernel is set to ``winlen / 6``.
    gauss_sd : float | None
        Standard deviation of the gaussian kernel in seconds. If ``None``
        the standard deviation is set to ``winlen / 6``.
    fwhm : float | None
        Full width at half maximum of the gaussian kernel in seconds.
    kernel : array-like | None
        Kernel to use for convolution. If ``None`` the gaussian kernel is
        constructed from ``winlen`` and ``gauss_sd``.
    sfreq : float
        Sampling frequency (in Hz) of the spike density representation. Default
        is ``500``.

    Returns
    -------
    kernel : array-like
        Gaussian kernel.
    trim : int
        Number of samples to trim from the beginning and end of the signal to
        get the same length as the convolved signal.
    """
    if kernel is None:
        if fwhm is not None:
            gauss_sd = _gauss_sd_from_FWHM(fwhm)
            winlen = gauss_sd * 6.
        else:
            gauss_sd = winlen / 6. if gauss_sd is None else gauss_sd

        gauss_sd = gauss_sd * sfreq
        win_smp, trim = _symmetric_window_samples(winlen, sfreq)
        kernel = _gauss_kernel_samples(win_smp, gauss_sd) * sfreq
    else:
        assert (len(kernel) % 2) == 1
        trim = int((len(kernel) - 1) / 2)

    return kernel, trim


def _gauss_sd_from_FWHM(FWHM):
    """Calculate the standard deviation of a gaussian kernel from the FWHM.

    Parameters
    ----------
    FWHM : float
        Full width at half maximum of the gaussian kernel in seconds.

    Returns
    -------
    gauss_sd : float
        Standard deviation of the gaussian kernel in seconds.
    """
    gauss_sd = FWHM / (2 * np.sqrt(2 * np.log(2)))
    return gauss_sd


def _FWHM_from_window(winlen=None, gauss_sd=None):
    """Calculate the full width at half maximum of a gaussian kernel.

    Parameters
    ----------
    winlen : float | None
        Length of the gaussian kernel in seconds. The length is assumed to be
        6 times the standard deviation of the gaussian kernel. Exactly one of
        ``winlen`` and ``gauss_sd`` must be specified. Default is ``None``,
        in which case ''winlen`` is ignored and ``gauss_sd`` is used.
    gauss_sd : float | None
        Standard deviation of the gaussian kernel in seconds. Exactly one of
        ``winlen`` and ``gauss_sd`` must be specified. Default is ``None``,
        in which case ''gauss_sd`` is ignored and ``winlen`` is used.

    Returns
    -------
    FWHM : float
        Full width at half maximum of the gaussian kernel in seconds.
    """
    # exactly one of the two must be specified
    assert winlen is not None or gauss_sd is not None
    assert winlen is None or gauss_sd is None

    gauss_sd = winlen / 6 if gauss_sd is None else gauss_sd
    FWHM = 2 * np.sqrt(2 * np.log(2)) * gauss_sd
    return FWHM
