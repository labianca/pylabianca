import numpy as np
from .utils import (_deal_with_picks, _turn_spike_rate_to_xarray,
                    _gauss_kernel_samples, _symmetric_window_samples)



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

    if isinstance(step, bool) and not step:
        assert tmin is not None
        assert tmax is not None
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
                spk.time[pick], spk.trial[pick], spk.time_limits,
                spk.n_trials, winlen=winlen, step=step)
            frate.append(frt)
            cell_names.append(spk.cell_names[pick])

    frate = np.stack(frate, axis=0)
    frate = _turn_spike_rate_to_xarray(times, frate, spk,
                                        cell_names=cell_names)
    return frate


def _compute_spike_rate_numpy(spike_times, spike_trials, time_limits,
                              n_trials, winlen=0.25, step=0.05):
    halfwin = winlen / 2
    epoch_len = time_limits[1] - time_limits[0]
    n_steps = int(np.floor((epoch_len - winlen) / step + 1))

    fr_t_start = time_limits[0] + halfwin
    fr_tend = time_limits[1] - halfwin + step * 0.001
    times = np.arange(fr_t_start, fr_tend, step=step)
    frate = np.zeros((n_trials, n_steps))

    for step_idx in range(n_steps):
        winlims = times[step_idx] + np.array([-halfwin, halfwin])
        msk = (spike_times >= winlims[0]) & (spike_times < winlims[1])
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
def _spike_density(spk, picks=None, winlen=0.3, gauss_sd=None, kernel=None,
                   sfreq=500.):
    '''Calculates normal (constant) spike density.

    The density is computed by convolving the binary spike representation
    with a gaussian kernel.
    '''
    from scipy.signal import correlate

    if kernel is None:
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


def depth_of_selectivity(frate, by):
    '''Compute depth of selectivity for given category.

    Parameters
    ----------
    frate : xarray
        Xarray with firing rate data.
    by : str
        Name of the dimension to group by and calculate depth of
        selectivity for.

    Returns
    -------
    selectivity : xarray
        Xarray with depth of selectivity.
    '''
    avg_by_probe = frate.groupby(by).mean(dim='trial')
    n_categories = len(avg_by_probe.coords[by])
    r_max = avg_by_probe.max(by)
    numerator = n_categories - (avg_by_probe / r_max).sum(by)
    selectivity = numerator / (n_categories - 1)
    return selectivity, avg_by_probe


def compute_selectivity_windows(spk, windows=None, compare='image',
                                baseline=None, progress=True):
    '''
    Compute selectivity for each cell in specific time windows.

    Parameters
    ----------
    spk : SpikeEpochs
        Spike epochs object.
    windows : dict of tuples
        Dictionary with keys being the names of the windows and values being
        ``(start, end)`` tuples of window time limits in seconds.
    compare : str
        Metadata category to compare. Defaults to ``'image'``.

    Returns
    -------
    selectivity : dict of pandas.DataFrame
        Dictionary of dataframes with selectivity for each cell. Each
        dictionary key and each dataframe corresponds to a time window of given
        name.
    '''
    import pandas as pd
    from scipy.stats import kruskal
    from sarna.utils import progressbar

    if windows is None:
        windows = {'early': (0.1, 0.6), 'late': (0.6, 1.1)}

    columns = ['neuron', 'region', 'kruskal_stat', 'kruskal_pvalue', 'DoS']
    has_region = 'region' in spk.cellinfo
    if not has_region:
        columns.pop(1)

    level_labels = np.unique(spk.metadata[compare])
    n_levels = len(level_labels)
    level_cols = [f'FR_{compare}{lb}' for lb in level_labels]
    level_cols_norm = (list() if baseline is None else
                       [f'nFR_{compare}{lb}' for lb in level_labels])
    columns += level_cols + level_cols_norm

    frate, df = dict(), dict()
    for name, limits in windows.items():
        frate[name] = spk.spike_rate(tmin=limits[0], tmax=limits[1],
                                     step=False)
        df[name] = pd.DataFrame(columns=columns)

    n_cells = len(spk)
    n_windows = len(windows.keys())
    pbar = progressbar(progress, total=n_cells * n_windows)
    for cell_idx in range(n_cells):
        if has_region:
            brain_region = spk.cellinfo.loc[cell_idx, 'region']
        for window in windows.keys():

            if has_region:
                df[window].loc[cell_idx, 'region'] = brain_region
            df[window].loc[cell_idx, 'neuron'] = spk.cell_names[cell_idx]

            if not (frate[window][cell_idx].values > 0).any():
                stat, pvalue = np.nan, 1.
            else:
                data = [arr.values for label, arr in
                        frate[window][cell_idx].groupby(compare)]
                stat, pvalue = kruskal(*data)

            df[window].loc[cell_idx, 'kruskal_stat'] = stat
            df[window].loc[cell_idx, 'kruskal_pvalue'] = pvalue

            # compute DoS
            dos, avg = depth_of_selectivity(frate[window][cell_idx],
                                            by=compare)
            preferred = int(avg.argmax(dim=compare).item())
            df[window].loc[cell_idx, 'DoS'] = dos.item()
            df[window].loc[cell_idx, 'preferred'] = preferred

            # other preferred?
            perc_of_max = avg / avg[preferred]
            other_pref = ((perc_of_max < 1.) & (perc_of_max >= 0.75))
            if other_pref.any().item():
                txt = ','.join([str(x) for x in np.where(other_pref)[0]])
            else:
                txt = ''
            df[window].loc[cell_idx, 'preferred_second'] = txt

            # save firing rate
            if baseline is not None:
                base_fr = baseline[cell_idx].mean(dim='trial').item()
                if base_fr == 0:
                    base_fr = avg[avg > 0].min().item()

            for idx in range(n_levels):
                fr = avg[idx].item()
                level = avg.coords[compare][idx].item()
                df[window].loc[cell_idx, f'FR_{compare}{level}'] = fr

                if baseline is not None:
                    nfr = fr / base_fr
                    df[window].loc[cell_idx, f'nFR_{compare}{level}'] = nfr

            pbar.update(1)

    for window in windows.keys():
        df[window] = df[window].infer_objects()
        df[window].loc[:, 'preferred'] = df[window]['preferred'].astype('int')

    return df, frate