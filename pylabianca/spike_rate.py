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


def depth_of_selectivity(frate, groupby, ignore_below=1e-15):
    '''Compute depth of selectivity for given category.

    Parameters
    ----------
    frate : xarray
        Xarray with firing rate data.
    groupby : str
        Name of the dimension to group by and calculate depth of
        selectivity for.
    ignore_below : float
        Ignore values below this threshold. This is useful when spike density
        is passed in ``frate`` - due to numerical errors, some values may be
        very small or even negative but not exactly zero. Such values can lead
        to depth of selectivity being far greater than 1. Default is ``1e-15``.

    Returns
    -------
    selectivity : xarray
        Xarray with depth of selectivity.
    '''
    if ignore_below > 0:
        frate = frate.copy()
        msk = frate.values < ignore_below
        frate.values[msk] = 0.

    avg_by_probe = frate.groupby(groupby).mean(dim='trial')
    n_categories = len(avg_by_probe.coords[groupby])
    r_max = avg_by_probe.max(dim=groupby)

    singleton = r_max.shape == ()
    if singleton and r_max.item() == 0:
        return 0, avg_by_probe

    numerator = n_categories - (avg_by_probe / r_max).sum(dim=groupby)
    selectivity = numerator / (n_categories - 1)
    selectivity.name = 'depth of selectivity'

    if not singleton:
        bad_selectivity = r_max < ignore_below
        if (bad_selectivity).any():
            selectivity[bad_selectivity] = 0

    return selectivity, avg_by_probe


# TODO: could use njobs!
def compute_selectivity_windows(spk, windows=None, compare='image',
                                baseline=None, test='kruskal', n_perm=2000,
                                progress=True):
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
        Metadata category to compare. Defaults to ``'image'``. If ``None``,
        the average firing rate is compared to the baseline.
    baseline : None | xarray.DataArray
        Baseline firing rate to compare to.
    test : str
        Test to use for computing selectivity. Can be ``'kruskal'`` or
        ``'permut'``. Defaults to ``'kruskal'``.
    n_perm : int
        Number of permutations to use for permutation test.
    progress : bool | str
        Whether to show a progress bar. If string, it can be ``'text'`` for
        text progress bar or ``'notebook'`` for a notebook progress bar.

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

    use_test = kruskal if test == 'kruskal' else permutation_test
    test_args = {'n_perm': n_perm} if test == 'permut' else {}

    if windows is None:
        windows = {'early': (0.1, 0.6), 'late': (0.6, 1.1)}

    columns = ['neuron', 'region', f'{test}_stat', f'{test}_pvalue', 'DoS']
    has_region = 'region' in spk.cellinfo
    if not has_region:
        columns.pop(1)

    if compare is not None:
        level_labels = np.unique(spk.metadata[compare])
        n_levels = len(level_labels)
        level_cols = [f'FR_{compare}{lb}' for lb in level_labels]
        level_cols_norm = (list() if baseline is None else
                        [f'nFR_{compare}{lb}' for lb in level_labels])
        columns += level_cols + level_cols_norm
    else:
        columns += ['FR_baseline', 'FR_condition', 'nFR_condition']

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
            elif compare is not None:
                data = [arr.values for label, arr in
                        frate[window][cell_idx].groupby(compare)]
                stat, pvalue = use_test(*data, **test_args)
            elif compare is None and baseline is not None:
                data = [frate[window][cell_idx].values,
                        baseline[cell_idx].values]
                stat, pvalue = use_test(*data, **test_args)

            df[window].loc[cell_idx, f'{test}_stat'] = stat
            df[window].loc[cell_idx, f'{test}_pvalue'] = pvalue

            # compute DoS
            if compare is not None:
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
                    if compare is not None and (avg > 0).any().item():
                        base_fr = avg[avg > 0].min().item()
                    else:
                        base_fr = 1.

            if compare is not None:
                for idx in range(n_levels):
                    fr = avg[idx].item()
                    level = avg.coords[compare][idx].item()
                    df[window].loc[cell_idx, f'FR_{compare}{level}'] = fr

                    if baseline is not None:
                        nfr = fr / base_fr
                        df[window].loc[cell_idx, f'nFR_{compare}{level}'] = nfr
            else:
                average_fr = frate[window][cell_idx].mean(dim='trial').item()
                df[window].loc[cell_idx, 'FR_baseline'] = base_fr
                df[window].loc[cell_idx, 'FR_condition'] = average_fr
                df[window].loc[cell_idx, 'nFR_condition'] = average_fr / base_fr
            pbar.update(1)
    pbar.close()

    for window in windows.keys():
        df[window] = df[window].infer_objects()
        if compare is not None:
            # TODO: not sure if this is the right way to do in all cases
            df[window].loc[:, 'preferred'] = (
                df[window]['preferred'].astype('int'))

    return df, frate


# should return: matrix of selectivity for each permutation
def compute_selectivity_continuous(frate, compare='image', n_perm=500,
                                   n_jobs=1, min_Hz=False):
    '''
    Compute population selectivity for specific experimental category.

    Parameters
    ----------
    frate : xarray.DataArray
        Firing rate of the neurons.
    compare : str
        Metadata category to compare. Defaults to ``'image'``.
    n_perm : int
        Number of permutations to use for permutation test.
    n_jobs : int
        Number of parallel jobs. No parallel computation is done when
        ``n_jobs=1`` (default).
    min_Hz : 0.1 | bool
        Minimum spiking rate threshold (in Hz). Cells below this threshold will
        be ignored.

    Returns
    -------
    selectivity : dict of pandas.DataFrame
        Dictionary of dataframes with selectivity for each cell. Each
        dictionary key and each dataframe corresponds to a time window of given
        name.
    '''
    import xarray as xr

    # select cells
    if min_Hz:
        msk = frate.mean(dim=('trial', 'time')) >= min_Hz
        frate = frate.sel(cell=msk)

    frate = frate.transpose('trial', 'cell', 'time')

    # permutations
    # ------------
    arrs = [arr for _, arr in frate.groupby(compare)]
    results = permutation_test(
        *arrs, paired=False, n_perm=n_perm,
        return_pvalue=False, return_distribution=True, n_jobs=n_jobs)

    # turn to xarray
    # --------------
    cells = frate.cell.values
    if 'subject' in frate.attrs:
        subj = frate.attrs['subject']
        cells = ['_'.join([subj, x]) for x in cells]

    # perm
    dims = ['perm', 'cell', 'time']
    coords = {'perm': np.arange(n_perm) + 1,
              'time': frate.time,
              'cell': cells}
    results['dist'] = xr.DataArray(data=results['dist'], dims=dims,
                                   coords=coords, name='t value')

    # stat
    coords = {k: coords[k] for k in dims[1:]}
    results['stat'] = xr.DataArray(
        data=results['stat'], dims=dims[1:], coords=coords, name='t value')

    # thresh
    results['thresh'] = np.stack(results['thresh'], axis=0)
    dims2 = ['tail', 'cell', 'time']
    coords.update({'tail': ['pos', 'neg']})
    results['thresh'] = xr.DataArray(
        data=results['thresh'], dims=dims2, coords=coords, name='t value')

    # add cell coords
    for key in results.keys():
        coords = {coord: ('cell', frate.coords[coord].values)
                  for coord in ['region', 'region2', 'channel', 'cluster']}
        results[key] = results[key].assign_coords(coords)

    return results


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


# TODO: add njobs
# TODO: refactor to separate cluster-based and cell selection
# TODO: create more progress bars and pass to cluster_based_test
def cluster_based_selectivity(frate, spk=None, compare='probe',
                              cluster_entry_pval=0.05, n_permutations=1000,
                              n_stat_permutations=0, pbar='notebook',
                              correct_dos_window=0.,
                              window_of_interest=(0.1, 1),
                              min_time_in_window=0.2,
                              min_depth_of_selectivity=0.4,
                              stat_fun=None, format='new'):
    '''Calculate category-sensitivity of neurons using cluster-based tests.

    Performs cluster-based ANOVA on firing rate of each neuron to test
    category-selectivity of the neurons.

    Parameters
    ----------
    frate : xarray.DataArray
        Xarray with spike rate  or spike density containing ``'cell'``,
        ``'trial'`` and ``'time'`` dimensions.
    spk : SpikeEpochs
        Spikes to use when calculating depth of sensitivity in cluster
        time window.
    compare : str
        Dimension labels specified for ``'trial'`` dimension that constitutes
        categories to test selectivity for.
    cluster_entry_pval : float
        P value used as a cluster-entry threshold. The default is ``0.05``.
    pbar : str | tqdm progressbar
        Progressbar to use. The default is ``'notebook'`` which creates a
        ``tqdm.notebook.tqdm`` progressbar.

    Returns
    -------
    df_cluster : pandas.DataFrame
        Dataframe with category-sensitivity information.
    '''
    import pandas as pd
    from .viz import check_modify_progressbar

    n_cells = len(frate)
    corrwin = [-correct_dos_window, correct_dos_window]
    pbar = check_modify_progressbar(pbar, total=n_cells)

    # init dataframe for storing results
    cols = ['pval', 'window', 'in_toi', 'preferred', 'n_preferred']
    if format == 'old':
        cols = ['cluster1_' + postfix for postfix in cols]
        cols = ['cell', 'n_signif_clusters'] + cols
    elif format == 'new':
        cols = ['neuron', 'region', 'region_short', 'cluster'] + cols

    df_cluster = pd.DataFrame(columns=cols)

    for pick, fr_cell in enumerate(frate):
        cell_name = fr_cell.cell.item()
        row_idx = df_cluster.shape[0]

        # TODO - pass relevant arrays
        _, clusters, pval = cluster_based_test(
            fr_cell, compare=compare, cluster_entry_pval=cluster_entry_pval,
            paired=False, stat_fun=stat_fun, n_permutations=n_permutations,
            n_stat_permutations=n_stat_permutations, progress=False)

        # process clusters
        # TODO - separate function
        n_clusters = len(pval)
        n_signif_clusters = 0
        if n_clusters > 0:
            # extract relevant cluster info
            n_signif_clusters = (pval < 0.05).sum()
            time_in_window = [_compute_time_in_window(
                                  clusters[idx], fr_cell.time,
                                  window_of_interest)
                              for idx in range(n_clusters)]
            win_start = np.array([fr_cell.time[clusters[idx]][0]
                                  for idx in range(n_clusters)])

            # select clusters to save in df
            good_clst = (np.array(time_in_window) > 0) | (win_start > 0)
            clst_idx = np.where(good_clst & (pval < 0.05))[0]

            if len(clst_idx) == 0:
                clst_idx = [np.argmax(time_in_window)]
            cluster_p = pval[clst_idx]

            for ord_idx, idx in enumerate(clst_idx):
                # get cluster time window
                clst_msk = clusters[idx]
                this_pval = cluster_p[ord_idx]
                twin = fr_cell.time[clst_msk].values[[0, -1]]
                twin_str = '{:.03f} - {:.03f}'.format(*twin)
                in_toi = time_in_window[idx]

                if format == 'old':
                    prefix = 'cluster{}'.format(ord_idx + 1)
                    df_cluster.loc[row_idx, prefix + '_pval'] = this_pval
                    df_cluster.loc[row_idx, prefix + '_window'] = twin_str
                    df_cluster.loc[row_idx, prefix + '_in_toi'] = in_toi
                elif format == 'new':
                    if 'region' in fr_cell:
                        region_name = fr_cell.region.item()
                        short_name = region_name[1:-1]
                    else:
                        region_name = None
                        short_name = None

                    df_idx = row_idx + ord_idx
                    df_cluster.loc[df_idx, 'neuron'] = cell_name
                    df_cluster.loc[df_idx, 'region'] = region_name
                    df_cluster.loc[df_idx, 'region_short'] = short_name
                    df_cluster.loc[df_idx, 'cluster'] = ord_idx + 1
                    df_cluster.loc[df_idx, 'pval'] = this_pval
                    df_cluster.loc[df_idx, 'window'] = twin_str
                    df_cluster.loc[df_idx, 'in_toi'] = in_toi

                # calculate depth of selectivity for each selective window
                tmin, tmax = twin + corrwin
                if spk is not None:
                    frate_avg = spk.spike_rate(
                        picks=pick, step=False, tmin=tmin, tmax=tmax)

                    # TODO: for GammBur I used only correct trials,
                    #       but it may not always make sense...
                    frate_avg = frate_avg.isel(cell=0)
                    # .sel(trial=frate_avg.ifcorrect)
                else:
                    frate_avg = fr_cell.sel(time=slice(tmin, tmax)).mean(
                        dim='time')  # .sel(trial=frate.ifcorrect)
                depth, preferred = depth_of_selectivity(frate_avg, by=compare)

                # find preferred categories
                perc_pref = preferred / preferred.max()
                pref_idx = np.where(perc_pref >= 0.75)[0]
                perc_pref_sel = perc_pref.values[pref_idx].argsort()[::-1]

                # check if cluster time window can be deemed as selective
                is_sel = ((cluster_p[ord_idx] < 0.05)
                          & (in_toi >= min_time_in_window)
                          & (depth.item() > min_depth_of_selectivity))

                if format == 'old':
                    pref = pref_idx[perc_pref_sel] + 1
                    pref_str = ';'.join([str(x) for x in pref])

                    df_cluster.loc[row_idx, prefix + '_preferred'] = pref_str
                    df_cluster.loc[row_idx, prefix + '_n_preferred'] = len(pref)
                    df_cluster.loc[row_idx, prefix + 'DoS'] = depth.item()
                    df_cluster.loc[row_idx, prefix + '_selective'] = is_sel
                elif format == 'new':
                    # TODO - use category labels in preferred!
                    pref = pref_idx[perc_pref_sel].tolist()
                    pref_str = str(pref)

                    df_cluster.loc[df_idx, 'preferred'] = pref_str
                    df_cluster.loc[df_idx, 'n_preferred'] = len(pref)
                    df_cluster.loc[df_idx, 'DoS'] = depth.item()
                    df_cluster.loc[df_idx, 'selective'] = is_sel

        else:
            if format == 'old':
                df_cluster.loc[row_idx, 'cluster1_selective'] = False

        # store the information in a dataframe
        if format == 'old':
            df_cluster.loc[row_idx, 'cell'] = cell_name
            df_cluster.loc[row_idx, 'n_signif_clusters'] = n_signif_clusters

        pbar.update(1)
    return df_cluster


def _compute_time_in_window(clst_msk, times, window_of_interest):
    twin = times[clst_msk].values[[0, -1]]
    twin[0] = min(max(window_of_interest[0], twin[0]),
                    window_of_interest[1])
    twin[1] = max(min(window_of_interest[1], twin[1]),
                    window_of_interest[0])
    return twin[1] - twin[0]
