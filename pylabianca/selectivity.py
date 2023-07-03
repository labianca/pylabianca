import numpy as np


# TODO: adapt for multiple cells
# TODO: make more universal (do not require 'time' and 'trial' dimensions)
def explained_variance(frate, groupby, kind='omega'):
    """Calculate percentage of explained variance (PEV) effect size.

    Parameters
    ----------
    frate : xarray.DataArray
        Firing rate data.
    groupby : str
        Name of the grouping variable. It is assumed that the variable is
        categorical with at least two levels and is specified as a coordinate
        for the trial dimension.
    kind : str
        Type of effect size to calculate. Can be either 'omega' or 'eta' for
        omega squared and eta squared respectively.

    Returns
    -------
    es : xarray.DataArray
        Explained variance effect size.
    """
    assert kind in ('omega', 'eta'), 'kind must be either "omega" or "eta"'
    is_omega = kind == 'omega'

    global_avg = frate.mean(dim='trial')
    has_time = 'time' in frate.dims
    if has_time:
        n_times = len(frate.coords['time'])

    groups, per_group = np.unique(frate.image, return_counts=True)
    n_groups = len(groups)

    SS_total = ((frate - global_avg) ** 2).sum(dim='trial')
    SS_between = (np.zeros((n_groups, n_times))
                  if has_time else np.zeros(n_groups))

    if is_omega:
        SS_within = SS_between.copy()

    for idx, (label, arr) in enumerate(frate.groupby('image')):
        # are group labels always sorted when using .groupby?
        group_avg = arr.mean(dim='trial')
        SS_between[idx] = per_group[idx] * (group_avg - global_avg) ** 2

        if is_omega:
            within_group = ((arr - group_avg) ** 2).sum(dim='trial')
            SS_within[idx] = within_group

    SS_between = SS_between.sum(axis=0)

    if not is_omega:
        es = SS_between / SS_total
        es.name = 'eta squared'
    else:
        df = n_groups - 1
        n_trials = len(frate.coords['trial'])
        MSE = SS_within.sum(axis=0) / n_trials
        es = (SS_between - df * MSE) / (SS_total + MSE)
        es.name = 'omega squared'

    return es


def depth_of_selectivity(frate, groupby, ignore_below=1e-15):
    '''Compute depth of selectivity for given category.

    Parameters
    ----------
    frate : xarray
        Xarray with firing rate data.
    groupby : str
        Name of the dimension to group by and calculate depth of selectivity
        for.
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


def compute_selectivity_continuous(frate, compare='image', n_perm=500,
                                   n_jobs=1, min_Hz=False):
    '''
    Compute population selectivity for specific experimental category.

    Uses permutation test to compute selectivity p-values.

    Parameters
    ----------
    frate : xarray.DataArray
        Firing rate of the neurons.
    compare : str
        Metadata category to compare. Defaults to ``'image'``.
    n_perm : int
        Number of permutations to use for permutation test. Defaults to
        ``500``.
    n_jobs : int
        Number of parallel jobs. No parallel computation is done when
        ``n_jobs=1`` (default).
    min_Hz : 0.1 | bool
        Minimum spiking rate threshold (in Hz). Cells below this threshold will
        be ignored.

    Returns
    -------
    selectivity : dict of xarray.DataArray
        Dictionary of DataArrays with selectivity for each cell. The following
        keys are used:
        * ``'stat'`` - selectivity statistic (t values), DataArray with
          dimensions ``('cell', 'time')`` (unless time was not present in the
          ``frate``)
        * ``'thresh'`` - 95% significance thresholds from permutation test:
          lower, negative (2.5%) and higher, positive (97.5%) tails. DataArray
          with dimensions ``('tail', 'cell', 'time')`` (unless time was not
          present in the ``frate``)
        * ``'perm'`` - selectivity statistic for each permutation. DataArray
          with dimensions ``('perm', 'cell', 'time')`` (unless time was not
          present in the ``frate``)
    '''
    import xarray as xr
    from .stats import permutation_test

    has_time = 'time' in frate.dims

    # select cells
    if min_Hz:
        reduce_dims = 'trial' if not has_time else ('trial', 'time')
        msk = frate.mean(dim=reduce_dims) >= min_Hz
        frate = frate.sel(cell=msk)

    frate_dims = ['trial', 'cell']
    if has_time:
        frate_dims.append('time')
    frate = frate.transpose(*frate_dims)

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
    dims = ['perm', 'cell']
    coords = {'perm': np.arange(n_perm) + 1,
              'cell': cells}
    if has_time:
        dims.append('time')
        coords['time'] = frate.time.values

    results['dist'] = xr.DataArray(data=results['dist'], dims=dims,
                                   coords=coords, name='t value')

    # stat
    coords = {k: coords[k] for k in dims[1:]}
    results['stat'] = xr.DataArray(
        data=results['stat'], dims=dims[1:], coords=coords, name='t value')

    # thresh
    results['thresh'] = np.stack(results['thresh'], axis=0)
    dims2 = ['tail'] + dims[1:]
    coords.update({'tail': ['pos', 'neg']})
    results['thresh'] = xr.DataArray(
        data=results['thresh'], dims=dims2, coords=coords, name='t value')

    # add cell coords
    for key in results.keys():
        copy_coords = ['region', 'region2', 'anat', 'channel', 'cluster']
        copy_coords = [coord for coord in copy_coords if coord in frate.coords]
        coords = {coord: ('cell', frate.coords[coord].values)
                  for coord in copy_coords}
        results[key] = results[key].assign_coords(coords)

    return results


# TODO: add njobs
# TODO: refactor to separate cluster-based and cell selection
# TODO: create more progress bars and pass to cluster_based_test
# TODO: use calculate_dos to ignore DoS calculation
def cluster_based_selectivity(frate, compare, spk=None,
                              cluster_entry_pval=0.05, n_permutations=1000,
                              n_stat_permutations=0, pbar='notebook',
                              correct_window=0., min_cluster_pval=0.1,
                              calculate_pev=False, calculate_dos=True,
                              calculate_peak_pev=False, stat_fun=None,
                              baseline_window=(-0.5, 0)):
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
    import xarray as xr

    from .stats import cluster_based_test
    from .viz import check_modify_progressbar

    n_cells = len(frate)
    corrwin = [-correct_window, correct_window]
    pbar = check_modify_progressbar(pbar, total=n_cells)

    # init dataframe for storing results
    cols = ['pval', 'window', 'preferred', 'n_preferred', 'FR_preferred']
    cols = ['neuron', 'region', 'region_short', 'anat', 'cluster'] + cols
    if calculate_pev:
        cols += ['pev']
    if calculate_dos:
        cols += ['DoS']
    if calculate_peak_pev:
        cols += ['peak_pev']
    if baseline_window is not None:
        cols += ['FR_vs_baseline']

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
        if n_clusters > 0:

            # select clusters to save in df
            good_clst = pval <= min_cluster_pval
            clst_idx = np.where(good_clst)[0]

            for ord_idx, idx in enumerate(clst_idx):
                # get cluster time window
                clst_msk = clusters[idx]
                this_pval = pval[idx]
                twin = fr_cell.time[clst_msk].values[[0, -1]]
                twin_str = '{:.03f} - {:.03f}'.format(*twin)

                # get anatomical information
                if 'region' in fr_cell.coords:
                    region_name = fr_cell.region.item()

                    if 'region2' in fr_cell.coords:
                        short_name = fr_cell.region2.item()
                    else:
                        short_name = region_name[1:-1]
                else:
                    region_name = None
                    short_name = None

                if 'anat' in fr_cell.coords:
                    anat_name = fr_cell.anat.item()
                else:
                    anat_name = None

                df_idx = row_idx + ord_idx
                df_cluster.loc[df_idx, 'neuron'] = cell_name
                df_cluster.loc[df_idx, 'region'] = region_name
                df_cluster.loc[df_idx, 'region_short'] = short_name
                df_cluster.loc[df_idx, 'anat'] = anat_name
                df_cluster.loc[df_idx, 'cluster'] = ord_idx + 1
                df_cluster.loc[df_idx, 'pval'] = this_pval
                df_cluster.loc[df_idx, 'window'] = twin_str

                # calculate depth of selectivity for each selective window
                tmin, tmax = twin + corrwin
                if spk is not None:
                    frate_avg = spk.spike_rate(
                        picks=pick, step=False, tmin=tmin, tmax=tmax
                        ).isel(cell=0)

                    # TODO: for GammBur I used only correct trials,
                    #       but it may not always make sense...
                    # .sel(trial=frate_avg.ifcorrect)

                    if baseline_window is not None:
                        frate_bsln = spk.spike_rate(
                            picks=pick, step=False, tmin=baseline_window[0],
                            tmax=baseline_window[1]).isel(cell=0)

                else:
                    frate_avg = fr_cell.sel(time=slice(tmin, tmax)).mean(
                        dim='time')  # .sel(trial=frate.ifcorrect)

                    if baseline_window is not None:
                        frate_bsln = fr_cell.sel(
                            time=slice(baseline_window[0], baseline_window[1])
                            ).mean(dim='time')

                depth, preferred = depth_of_selectivity(
                    frate_avg, groupby=compare)
                if isinstance(depth, xr.DataArray):
                    depth = depth.item()

                # TODO - this also could be moved outside
                # find preferred categories
                perc_pref = preferred / preferred.max()
                pref_idx = np.where(perc_pref >= 0.75)[0]
                perc_pref_sel = perc_pref.values[pref_idx].argsort()[::-1]

                # TODO - use category labels in preferred!
                pref = pref_idx[perc_pref_sel].tolist()
                pref_str = str(pref)

                # calculate PEV and peak PEV
                if calculate_pev:
                    pev = explained_variance(
                        frate_avg, compare).item()

                if calculate_peak_pev:
                    peak_pev = explained_variance(
                        fr_cell.sel(time=slice(tmin, tmax)),
                        compare
                        ).max().item()

                if baseline_window is not None:
                    FR_vs_bsln = (preferred.max().item()
                                  / frate_bsln.mean().item())

                df_cluster.loc[df_idx, 'preferred'] = pref_str
                df_cluster.loc[df_idx, 'n_preferred'] = len(pref)
                df_cluster.loc[df_idx, 'FR_preferred'] = preferred.max().item()
                df_cluster.loc[df_idx, 'DoS'] = depth

                if calculate_pev:
                    df_cluster.loc[df_idx, 'PEV'] = pev

                if calculate_peak_pev:
                    df_cluster.loc[df_idx, 'peak_PEV'] = peak_pev

                if baseline_window is not None:
                    df_cluster.loc[df_idx, 'FR_vs_baseline'] = FR_vs_bsln

        pbar.update(1)

    # make sure that the dataframe columns are of the right type
    df_cluster = df_cluster.infer_objects()
    return df_cluster


def assess_selectivity(df_cluster, min_cluster_p=0.05,
                       window_of_interest=(0.1, 1), min_time_in_window=0.2,
                       min_depth_of_selectivity=None, min_pev=None,
                       min_peak_pev=None, min_FR_vs_baseline=None,
                       min_FR_preferred=None):
    pval_msk = df_cluster.pval <= min_cluster_p

    def eval_column(column_name, min_value):
        msk = (np.ones_like(pval_msk) if min_value is None
               else df_cluster[column_name] >= min_value)
        return msk

    if min_time_in_window is not None:
        df_cluster = compute_time_in_window(df_cluster, window_of_interest)
        in_toi_msk = df_cluster.in_toi >= min_time_in_window
    else:
        in_toi_msk = np.ones_like(pval_msk)

    dos_msk = eval_column('DoS', min_depth_of_selectivity)
    pev_msk = eval_column('PEV', min_pev)
    peak_pev_msk = eval_column('peak_PEV', min_peak_pev)
    fr_vs_bsln_msk = eval_column('FR_vs_baseline', min_FR_vs_baseline)
    fr_pref_msk = eval_column('FR_preferred', min_FR_preferred)

    # check if cluster time window can be deemed as selective
    is_sel = (pval_msk & pev_msk & in_toi_msk & dos_msk & peak_pev_msk
              & fr_vs_bsln_msk & fr_pref_msk)

    df_cluster['selective'] = is_sel
    return df_cluster


def _compute_time_in_window(clst_window, window_of_interest):
    # clst_limits = times[clst_msk].values[[0, -1]]
    twin = [0., 0.]
    clst_limits = _parse_window(clst_window)
    twin[0] = min(max(window_of_interest[0], clst_limits[0]),
                      window_of_interest[1])
    twin[1] = max(min(window_of_interest[1], clst_limits[1]),
                      window_of_interest[0])
    return twin[1] - twin[0]


def _parse_window(window_str):
    start, fin = [float(x) for x in window_str.split(' - ')]
    return start, fin


def compute_time_in_window(df_cluster, window_of_interest):
    '''
    Compute time spent in the window of interest for each cluster.

    Parameters
    ----------
    df_cluster : pandas.DataFrame
        Dataframe with cluster information.
    window_of_interest : tuple of float
        Time window of interest.

    Returns
    -------
    df_cluster : pandas.DataFrame
        Dataframe with cluster information, including time spent in the window
        of interest.
    '''
    df_cluster.loc[:, 'in_toi'] = df_cluster.window.apply(
        _compute_time_in_window, args=(window_of_interest,))
    return df_cluster


# TODO: compare with time resolved selectivity
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
