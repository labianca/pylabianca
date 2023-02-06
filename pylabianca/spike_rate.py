import numpy as np
from .utils import (_deal_with_picks, _turn_spike_rate_to_xarray,
                    _gauss_kernel_samples, _symmetric_window_samples)



# CONSIDER wintype 'rectangular' vs 'gaussian'
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
                spk.time[pick], spk.trial[pick], [tmin, tmax],
                spk.n_trials, winlen=winlen, step=step)
            frate.append(frt)
            cell_names.append(spk.cell_names[pick])

    frate = np.stack(frate, axis=0)
    frate = _turn_spike_rate_to_xarray(times, frate, spk,
                                        cell_names=cell_names)
    return frate


# ENH: speed up by using previous mask in the next step to pre-select spikes
def _compute_spike_rate_numpy(spike_times, spike_trials, time_limits,
                              n_trials, winlen=0.25, step=0.05):
    half_win = winlen / 2
    used_range = time_limits[1] - time_limits[0]
    n_steps = int(np.floor((used_range - winlen) / step + 1))

    fr_t_start = time_limits[0] + half_win
    fr_t_end = time_limits[1] - half_win + step * 0.001
    times = np.arange(fr_t_start, fr_t_end, step=step)
    frate = np.zeros((n_trials, n_steps))

    for step_idx in range(n_steps):
        win_lims = times[step_idx] + np.array([-half_win, half_win])
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
    msk = (spike_times >= time_limits[0]) & (spike_times < time_limits[1])
    tri = spike_trials[msk]
    in_tri, count = np.unique(tri, return_counts=True)
    frate[in_tri] = count / winlen

    return frate


# TODO: consider an exact mode where the spikes are not transformed to raw
#       but placed exactly where the spike is (`loc=spike_time`) and evaluated
#       (maybe this is what is done by elephant?)
# TODO: check if time is symmetric wrt 0 (in most cases it should be as epochs
#       are constructed wrt specific event)
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
    '''
    from scipy.signal import correlate

    if kernel is None:
        if fwhm is not None:
            gauss_sd = _gauss_sd_from_FWHM(fwhm)
            winlen = gauss_sd * 6
        else:
            gauss_sd = winlen / 6 if gauss_sd is None else gauss_sd
            gauss_sd = gauss_sd * sfreq

        win_smp, trim = _symmetric_window_samples(winlen, sfreq)
        kernel = _gauss_kernel_samples(win_smp, gauss_sd) * sfreq
    else:
        assert (len(kernel) % 2) == 1
        trim = int((len(kernel) - 1) / 2)

    picks = _deal_with_picks(spk, picks)
    times, binrep = spk.to_raw(picks=picks, sfreq=sfreq)
    cnt_times = times[trim:-trim]

    cnt = correlate(binrep, kernel[None, None, :], mode='valid')
    return cnt_times, cnt


# TODO: move to borsar
def permutation_test(*arrays, paired=False, n_perm=1000, progress=False,
                     return_pvalue=True, return_distribution=True, n_jobs=1):
    import sarna

    n_groups = len(arrays)
    tail = 'both' if n_groups == 2 else 'pos'
    stat_fun = sarna.cluster._find_stat_fun(n_groups=n_groups, paired=paired,
                                            tail=tail)

    thresh, dist = sarna.cluster._compute_threshold_via_permutations(
        arrays, paired=paired, tail=tail, stat_fun=stat_fun,
        return_distribution=True, n_permutations=n_perm, progress=progress,
        n_jobs=n_jobs)

    stat = stat_fun(*arrays)

    # this does not make sense for > 1d,
    # maybe it didn't make sense for 1d too?
    # if isinstance(stat, np.ndarray):
    #     try:
    #         stat = stat[0]
    #     except IndexError:
    #         stat = stat.item()

    if return_pvalue:
        multiply_p = 2 if tail == 'both' else 1

        if not isinstance(stat, np.ndarray):
            if tail == 'pos' or tail == 'both' and stat >= 0:
                pval = (dist >= stat).mean() * multiply_p
            elif tail == 'neg' or tail == 'both' and stat < 0:
                pval = (dist <= stat).mean() * multiply_p

            if pval > 1.:
                pval = 1.
        else:
            if tail == 'pos':
                pval = (dist >= stat[None, :]).mean(axis=0)
            elif tail == 'neg':
                pval = (dist <= stat[None, :]).mean(axis=0)
            elif tail == 'both':
                is_pos = stat >= 0
                pval = np.zeros(stat.shape)
                pval[is_pos] = (dist[:, is_pos] >= stat[None, is_pos]
                                .mean(axis=0))
                pval[~is_pos] = (dist[:, ~is_pos] <= stat[None, ~is_pos]
                                .mean(axis=0))
                pval *= multiply_p

                above_one = pval > 1.
                if above_one.any():
                    pval[above_one] = 1.

    if return_distribution:
        out = dict()
        out['stat'] = stat
        out['thresh'] = thresh
        out['dist'] = dist

        if return_pvalue:
            out['pval'] = pval

        return out
    else:
        if return_pvalue:
            return stat, pval
        else:
            return stat


# TODO: auto-infer paired from xarray
def cluster_based_test(frate, compare='image', cluster_entry_pval=0.05,
                       paired=False, stat_fun=None, n_permutations=1_000,
                       n_stat_permutations=0, tail=None, progress=True):
    '''Perform cluster-based tests on firing rate data.

    Performs cluster-based test (ANOVA or t test, depending on the data) on
    firing rate to test, for example, category-selectivity of the neurons.

    Parameters
    ----------
    frate : xarray.DataArray
        Xarray with spike rate  or spike density containing
        observations as the first dimension (for example trials for
        between-trials analysis or cells for between-cells analysis).
        If you have both cells and trials then the cell should already be
        selected, via ``frate.isel(cell=0)`` for example or the trials
        dimension should be averaged (for example ``frate.mean(dim='trial')``).
    compare : str
        Dimension labels specified for ``'trial'`` dimension that constitutes
        categories to test selectivity for.
    cluster_entry_pval : float
        p value used as a cluster-entry threshold. The default is ``0.05``.
    paired : bool
        Whether a paired (repeated measures) or unpaired test should be used.

    Returns
    -------
    stats : numpy.ndarray
        Anova F statistics for every timepoint.
    clusters : list of numpy.ndarray
        List of cluster memberships.
    pval : numpy.ndarray
        List of p values from anova.
    '''
    from sarna.cluster import permutation_cluster_test_array

    # TODO: check if theres is a condition dimension (if so -> paired)
    arrays = [arr.values for _, arr in frate.groupby(compare)]

    if tail is None:
        n_groups = len(arrays)
        tail = 'both' if n_groups == 2 else 'pos'

    stat, clusters, pval = permutation_cluster_test_array(
        arrays, adjacency=None, stat_fun=stat_fun, threshold=None,
        p_threshold=cluster_entry_pval, paired=paired, tail=tail,
        n_permutations=n_permutations, n_stat_permutations=n_stat_permutations,
        progress=progress)

    return stat, clusters, pval


def _gauss_sd_from_FWHM(FWHM):
    gauss_sd = FWHM / (2 * np.sqrt(2 * np.log(2)))
    return gauss_sd


def _FWHM_from_window(winlen=None, gauss_sd=None):
    # exactly one of the two must be specified
    assert winlen is not None or gauss_sd is not None
    assert winlen is None or gauss_sd is None

    gauss_sd = winlen / 6 if gauss_sd is None else gauss_sd
    FWHM = 2 * np.sqrt(2 * np.log(2)) * gauss_sd
    return FWHM
