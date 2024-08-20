from numbers import Real

import numpy as np
import pandas as pd

from .utils import (xr_find_nested_dims, cellinfo_from_xarray,
                    _inherit_metadata_from_xarray, assign_session_coord)


# TODO: ! adapt for multiple cells
# TODO: ensure same ``frate`` explanation for all functions
#         Xarray with spike rate  or spike density containing ``'cell'``,
#        ``'trial'`` and ``'time'`` dimensions.
#       - obtained with SpikeEpochs.spike_rate or SpikeEpochs.spike_density
#       - dimensions: ..., ..., ...
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

    groups, per_group = np.unique(frate.coords[groupby], return_counts=True)
    n_groups = len(groups)

    SS_total = ((frate - global_avg) ** 2).sum(dim='trial')
    SS_between = (np.zeros((n_groups, n_times))
                  if has_time else np.zeros(n_groups))

    if is_omega:
        SS_within = SS_between.copy()

    for idx, (label, arr) in enumerate(frate.groupby(groupby)):
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


# TODO: ! adapt for multiple cells
def depth_of_selectivity(frate, groupby):
    '''Compute depth of selectivity for given category.

    Parameters
    ----------
    frate : xarray
        Xarray with firing rate data.
    groupby : str
        Name of the dimension to group by and calculate depth of selectivity
        for.

    Returns
    -------
    selectivity : xarray
        Xarray with depth of selectivity.
    '''

    avg_by_probe = frate.groupby(groupby).mean(dim='trial')
    n_categories = len(avg_by_probe.coords[groupby])
    r_max = avg_by_probe.max(dim=groupby)

    singleton = r_max.shape == ()
    if singleton and r_max.item() == 0:
        return 0, avg_by_probe

    numerator = n_categories - (avg_by_probe / r_max).sum(dim=groupby)
    selectivity = numerator / (n_categories - 1)
    selectivity.name = 'depth of selectivity'

    return selectivity, avg_by_probe


# CONSIDER: could add an attribute informing about condition order
#           (important for t test interpretation for example)
# TODO: change tail order to neg, pos?
def compute_selectivity_continuous(frate, compare='image', n_perm=500,
                                   n_jobs=1):
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

    frate_dims = ['trial', 'cell']
    if has_time:
        frate_dims.append('time')
    frate = frate.transpose(*frate_dims)

    # permutations
    # ------------
    arrs = [arr for _, arr in frate.groupby(compare)]
    stat_name = 't value' if len(arrs) == 2 else 'F value'
    stat_unit = stat_name[0]
    results = permutation_test(
        *arrs, paired=False, n_perm=n_perm,
        return_pvalue=False, return_distribution=True, n_jobs=n_jobs)

    # turn to xarray
    # --------------
    cells = frate.cell.values

    # perm
    dims = ['perm', 'cell']
    coords = {'perm': np.arange(n_perm) + 1,
              'cell': cells}
    if has_time:
        dims.append('time')
        coords['time'] = frate.time.values

    if n_perm > 0:
        results['dist'] = xr.DataArray(data=results['dist'], dims=dims,
                                    coords=coords, name=stat_name)
        use_data = results['stat']
    else:
        use_data = results
        results = dict()

    # stat
    coords = {k: coords[k] for k in dims[1:]}
    results['stat'] = xr.DataArray(
        data=use_data, dims=dims[1:], coords=coords, name=stat_name)

    # thresh
    if n_perm > 0:
        if isinstance(results['thresh'], list) and len(results['thresh']) == 2:
            # two-tail thresholds
            results['thresh'] = np.stack(results['thresh'], axis=0)
            dims2 = ['tail'] + dims[1:]
            coords.update({'tail': ['pos', 'neg']})
        else:
            dims2 = dims[1:]

        results['thresh'] = xr.DataArray(
            data=results['thresh'], dims=dims2, coords=coords, name=stat_name)

    # copy unit information
    # TODO: use a separate utility function
    for key in results.keys():
        results[key].attrs['unit'] = stat_unit
        if 'coord_units' in frate.attrs:
            results[key].attrs['coord_units'] = frate.attrs['coord_units']

    # add cell coords
    copy_coords = xr_find_nested_dims(frate, 'cell')
    if len(copy_coords) > 0:
        for key in results.keys():
            results[key] = _inherit_metadata_from_xarray(
                frate, results[key], 'cell', copy_coords=copy_coords)

    # transform to dataset:
    results = xr.Dataset(results)

    return results


# pbar is now True, which defaults to text tqdm, but could be 'auto'
# TODO: create more progress bars and pass to cluster_based_test
def cluster_based_selectivity(frate, compare, cluster_entry_pval=0.05,
                              n_permutations=1000, n_stat_permutations=0,
                              n_jobs=1, pbar=True, correct_window=0.,
                              min_cluster_pval=0.1, calculate_pev=False,
                              calculate_dos=True, calculate_peak_pev=False,
                              stat_fun=None, baseline_window=(-0.5, 0),
                              copy_cellinfo=True):
    '''Calculate category-sensitivity of neurons using cluster-based tests.

    Performs cluster-based ANOVA on firing rate of each neuron to test
    category-selectivity of the neurons.

    Parameters
    ----------
    frate : xarray.DataArray
        Xarray with spike rate  or spike density containing ``'cell'``,
        ``'trial'`` and ``'time'`` dimensions.
    compare : str
        Dimension labels specified for ``'trial'`` dimension that constitutes
        categories to test selectivity for.
    cluster_entry_pval : float
        P value used as a cluster-entry threshold. The default is ``0.05``.
    n_permutations : int
        Number of permutations to use for permutation test. The default is
        ``1000``.
    n_stat_permutations : int
        Number of permutations to use for non-parametric calculation of the p
        value for the test statistic (see ``stat_fun``). The default is ``0``,
        which means that the p value is calculated parametrically. Using
        non-parametric calculation is slower but more resistant to parametric
        assumptions.
    n_jobs : int
        Number of parallel jobs to use. The default is ``1``. If ``-1``, all
        available CPUs are used. ``n_jobs > 1`` requires the ``joblib``
        package.
    pbar : str | tqdm progressbar
        Progressbar to use. The default is ``'text'`` which creates a
        standard ``tqdm.tqdm`` text progressbar. Can also be ``'notebook'`` for
        a Jupyter notebook progressbar or ``False`` to disable progressbar.
    correct_window : float
        The cluster time window will be extended by this value when calculating
        window average statistics (DoS, PEV). The default is ``0.``.
    min_cluster_pval : float
        Minimum p value for a cluster to be stored in the results dataframe.
    calculate_pev : bool
        Whether to calculate percentage of explained variance (PEV) for each
        cluster. The default is ``False``.
    calculate_dos : bool
        Whether to calculate depth of selectivity (DoS) for each cluster. The
        default is ``True``.
    calculate_peak_pev : bool
        Whether to calculate peak PEV for each cluster. The default is
        ``False``.
    stat_fun : callable
        Function to used in cluster based test for calculating the test
        statistic. The default is ``None``, which means that the test statistic
        is inferred (repeated measures t test or ANOVA).
    baseline_window : tuple of float
        Time window to use as baseline for calculating the baseline-normalized
        firing rate. The default is ``(-0.5, 0)``.
    copy_cellinfo : bool | list of str
        Whether to copy cell information from the xarray to the results
        dataframe. If ``True``, all cell information is copied. If a list of
        strings, only the specified columns are copied. The default is
        ``True``.

    Returns
    -------
    df_cluster : pandas.DataFrame
        Dataframe with category-sensitivity information.
    '''
    import xarray as xr
    from .viz import check_modify_progressbar

    n_cells = len(frate)
    correct_window = [-correct_window, correct_window]
    pbar = check_modify_progressbar(pbar, total=n_cells)

    # this could be simplified or moved to a separate function
    cellinfo = _which_copy_cellinfo(frate, copy_cellinfo)

    # init dataframe for storing results
    df_cluster = _init_df_cluster(
        calculate_pev, calculate_dos, calculate_peak_pev, baseline_window,
        cellinfo=cellinfo)
    df_parts = list()

    def pick_cellinfo(cellinfo, pick):
        return cellinfo.iloc[pick, :] if cellinfo is not None else None

    if n_jobs == 1:
        for pick, fr_cell in enumerate(frate):
            cellinfo_row = pick_cellinfo(cellinfo, pick)
            df_part = _cluster_sel_process_cell(
                fr_cell, compare, cluster_entry_pval, stat_fun, n_permutations,
                n_stat_permutations, min_cluster_pval, df_cluster, correct_window,
                cellinfo_row, calculate_pev, calculate_dos, calculate_peak_pev,
                baseline_window
            )
            df_parts.append(df_part)
            pbar.update(1)
    else:
        from joblib import Parallel, delayed

        df_parts = Parallel(n_jobs=n_jobs)(
            delayed(_cluster_sel_process_cell)(
                frate.isel(cell=pick), compare, cluster_entry_pval, stat_fun,
                n_permutations, n_stat_permutations, min_cluster_pval,
                df_cluster, correct_window, pick_cellinfo(cellinfo, pick),
                calculate_pev, calculate_dos, calculate_peak_pev,
                baseline_window
            ) for pick in range(n_cells)
        )

    # make sure that the dataframe columns are of the right type
    df_cluster = pd.concat(df_parts, ignore_index=True)
    df_cluster = df_cluster.infer_objects()
    return df_cluster


def _cluster_sel_process_cell(fr_cell, compare, cluster_entry_pval,
        stat_fun, n_permutations, n_stat_permutations, min_cluster_pval,
        df_cluster, correct_window, cellinfo_row, calculate_pev,
        calculate_dos, calculate_peak_pev, baseline_window):
    from .stats import cluster_based_test

    # TODO - pass relevant arrays (not sure what I meant here ...)
    _, clusters, pvals = cluster_based_test(
        fr_cell, compare=compare, cluster_entry_pval=cluster_entry_pval,
        paired=False, stat_fun=stat_fun, n_permutations=n_permutations,
        n_stat_permutations=n_stat_permutations, progress=False)

    # process clusters
    df_part = _characterize_clusters(
        fr_cell, clusters, pvals, min_cluster_pval, df_cluster,
        correct_window, cellinfo_row, compare, calculate_dos,
        calculate_pev, calculate_peak_pev, baseline_window
    )

    return df_part


# TODO: the df could have correct types from the start
def _init_df_cluster(calculate_pev, calculate_dos, calculate_peak_pev,
                     baseline_window, cellinfo=None):
    '''Initialize dataframe for storing cluster information.'''
    add_cols = ['pval', 'window', 'preferred', 'n_preferred', 'FR_preferred']
    cols = ['cell']

    if cellinfo is not None and len(cellinfo) > 0:
        cols = cols + cellinfo.columns.tolist()
    cols += add_cols

    if calculate_pev:
        cols += ['PEV']
    if calculate_dos:
        cols += ['DoS']
    if calculate_peak_pev:
        cols += ['peak_PEV']
    if baseline_window is not None:
        cols += ['FR_vs_baseline']

    df_cluster = pd.DataFrame(columns=cols)
    return df_cluster


def _which_copy_cellinfo(frate, copy_cellinfo):
    '''Determine which cell information to copy to the results dataframe.'''
    do_copy_cellinfo = (
        (isinstance(copy_cellinfo, bool) and copy_cellinfo)
        or (isinstance(copy_cellinfo, list) and len(copy_cellinfo) > 0)
    )
    if do_copy_cellinfo:
        cellinfo = cellinfo_from_xarray(frate)

        if cellinfo is not None and len(cellinfo) > 0:
            if isinstance(copy_cellinfo, bool):
                copy_cellinfo = cellinfo.columns.tolist()
            else:
                copy_cellinfo_has_not = [col for col in copy_cellinfo
                                         if col not in cellinfo.columns]
                if len(copy_cellinfo_has_not) > 0:
                    raise ValueError('Some columns in copy_cellinfo are not '
                                     'present in the cellinfo dataframe: '
                                     f'{copy_cellinfo_has_not}')

                cellinfo = cellinfo.loc[:, copy_cellinfo]
        else:
            cellinfo = None
    else:
        cellinfo = None

    return cellinfo


def _characterize_cluster(fr_cell, cluster_mask, cluster_pval, df_cluster,
                          correct_window, ord_idx, cellinfo_row, compare,
                          calculate_dos, calculate_pev, calculate_peak_pev,
                          baseline_window):
    '''Characterize a single cluster.'''
    import xarray as xr

    # get cluster time window
    twin = fr_cell.time[cluster_mask].values[[0, -1]]
    twin_str = '{:.03f} - {:.03f}'.format(*twin)
    tmin, tmax = twin + correct_window

    df_cluster.loc[0, 'cell'] = fr_cell.cell.item()
    df_cluster.loc[0, 'cluster'] = ord_idx + 1
    df_cluster.loc[0, 'pval'] = cluster_pval
    df_cluster.loc[0, 'window'] = twin_str

    # copy info from cellinfo
    if cellinfo_row is not None:
        for col in cellinfo_row.index:
            df_cluster.loc[0, col] = cellinfo_row[col]

    # calculate depth of selectivity for each selective window
    frate_avg = fr_cell.sel(
        time=slice(tmin, tmax)
        ).mean(dim='time')

    if baseline_window is not None:
        frate_bsln = fr_cell.sel(
            time=slice(baseline_window[0], baseline_window[1])
            ).mean(dim='time')

    if calculate_dos:
        depth, preferred = depth_of_selectivity(
            frate_avg, groupby=compare)
        if isinstance(depth, xr.DataArray):
            depth = depth.item()
    else:
        preferred = frate_avg.groupby(compare).mean(dim='trial')

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

    df_cluster.loc[0, 'preferred'] = pref_str
    df_cluster.loc[0, 'n_preferred'] = len(pref)
    df_cluster.loc[0, 'FR_preferred'] = preferred.max().item()

    if calculate_dos:
        df_cluster.loc[0, 'DoS'] = depth

    if calculate_pev:
        df_cluster.loc[0, 'PEV'] = pev

    if calculate_peak_pev:
        df_cluster.loc[0, 'peak_PEV'] = peak_pev

    if baseline_window is not None:
        df_cluster.loc[0, 'FR_vs_baseline'] = FR_vs_bsln

    return df_cluster


def _characterize_clusters(fr_cell, clusters, pvals, min_cluster_pval,
                          df_cluster, correct_window, cellinfo_row,
                          compare, calculate_dos, calculate_pev,
                          calculate_peak_pev, baseline_window):
    '''Characterize all clusters.'''
    df_clst = list()
    n_clusters = len(pvals)
    if n_clusters > 0:
        # select clusters to save in df
        good_clst = pvals <= min_cluster_pval
        clst_idx = np.where(good_clst)[0]

        for ord_idx, idx in enumerate(clst_idx):
            df_part = _characterize_cluster(
                fr_cell, clusters[idx], pvals[idx], df_cluster.copy(),
                correct_window, ord_idx, cellinfo_row, compare,
                calculate_dos, calculate_pev, calculate_peak_pev,
                baseline_window
            )
            df_clst.append(df_part)

    if len(df_clst) == 0:
        df_clst = df_cluster.copy()
    else:
        df_clst = pd.concat(df_clst, ignore_index=True)

    return df_clst


def assess_selectivity(df_cluster, min_cluster_p=0.05,
                       window_of_interest=(0.1, 1), min_time_in_window=0.2,
                       min_depth_of_selectivity=None, min_pev=None,
                       min_peak_pev=None, min_FR_vs_baseline=None,
                       min_FR_preferred=None):
    '''
    Assess selectivity of clusters based on various criteria.

    Parameters
    ----------
    df_cluster : pandas.DataFrame
        Dataframe with cluster information.
    min_cluster_p : float
        Minimum p value for a cluster to be considered selective. The default
        is ``0.05``.
    window_of_interest : tuple of float
        Time window of interest. The default is ``(0.1, 1)``.
    min_time_in_window : float
        Minimum time spent in the window of interest for a cluster to be
        considered selective. The default is ``0.2``.
    min_depth_of_selectivity : float
        Minimum depth of selectivity for a cluster to be considered selective.
    min_pev : float
        Minimum percentage of explained variance (PEV) for a cluster to be
        considered selective.
    min_peak_pev : float
        Minimum peak PEV for a cluster to be considered selective.
    min_FR_vs_baseline : float
        Minimum firing rate vs baseline for a cluster to be considered
        selective.
    min_FR_preferred : float
        Minimum firing rate for the preferred category for a cluster to be
        considered selective.

    Returns
    -------
    df_cluster : pandas.DataFrame
        Dataframe with cluster information, including a column ``'selective'``
        that indicates whether the cluster is selective based on the specified
        criteria.
    '''
    # skip if no clusters, adding empty columns
    n_rows = df_cluster.shape[0]
    if n_rows == 0:
        df_cluster['selective'] = None
        if min_time_in_window is not None:
            df_cluster['in_toi'] = None
        return df_cluster

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
    '''Compute time spent in the window of interest for a cluster.'''
    # clst_limits = times[cluster_mask].values[[0, -1]]
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


# CONSIDER: selectivity as list of names / dict of lists of names ?
# RENAME: select_cells, as it can be used with various other cell-selection
#         criteria
# CONSIDER dataset case ...
# CONSIDER: too complicated, try to simplify
def pick_selective(frate, selectivity, threshold=None, session_coord='sub'):
    # one xarray and session_coord is None: assumes one subject
    #    - if same order of cells -> simple bool selection
    #    - if different order -> select by unique cell names
    #    raise Warning that can be silenced with ``session_coord=False``
    # one xarray and session_coord is not None: assumes multiple subjects
    #    - ...
    #    - ...
    # dictionary of xarrays: assumes multiple subjects/sessions, one per key
    import xarray as xr

    frate_xarr = isinstance(frate, xr.DataArray)

    if threshold is not None:
        selectivity = threshold_selectivity(selectivity, threshold)

    if frate_xarr:
        has_session_coord = session_coord in frate.coords
        same_cells = (frate.cell.values == selectivity.cell.values).all()

        if not has_session_coord:
            same_sessions = True
        else:
            same_sessions = (frate[session_coord].values
                             == selectivity[session_coord].values).all()

        # if same_cells and same_sessions:
        if same_cells and same_sessions:
            fr_sel = frate.isel(cell=np.where(selectivity)[0])
        else:
            # TODO: check if same sessions but with different order (or maybe
            #       that all sessions from frate are in selectivity)
            # if has_session_coord but not same_sessions / cells:
            fr_list = list()
            for ses, fr in frate.groupby(session_coord):
                stat_ses = selectivity.query({'cell': f'{session_coord} == "{ses}"'})
                sel_cell = stat_ses.cell.values[stat_ses.values]
                fr_sel = fr.sel(cell=sel_cell)
                fr_list.append(fr_sel)
            fr_sel = xr.concat(fr_list, dim='cell')
    elif isinstance(frate, dict):
        assert isinstance(selectivity, xr.DataArray)
        assert session_coord is not None

        # iterate over sessions
        fr_sel = dict()
        for ses, sel in selectivity.groupby(session_coord):
            if not sel.any():
                continue

            fr_ses = frate[ses].copy()
            same_cells = (fr_ses.cell.values == sel.cell.values).all()
            # raise warning when not same_cells ?
            # use what is below or fr_ses.sel(cell=sel) ?
            if same_cells:
                fr_sel[ses] = fr_ses.isel(cell=np.where(sel)[0])
            else:
                sel_cell = sel.cell.values[sel.values]
                fr_sel[ses] = fr_ses.sel(cell=sel_cell)

        return fr_sel


# CONSIDER: when threshold is Real, then > threshold and < -threshold
def threshold_selectivity(selectivity, threshold):
    '''Threshold selectivity statistics generating boolean selectivity.

    Parameters
    ----------
    selectivity : xarray.DataArray
        Selectivity statistic.
    threshold : float | xarray.DataArray
        Threshold value. If a float, selectivity values above the threshold are
        considered significant. If an xarray.DataArray, the threshold can be
        different for positive and negative values.

    Returns
    -------
    selected : xarray.DataArray
        Boolean array indicating whether the selectivity is above the
        threshold.
    '''
    import xarray as xr

    if isinstance(threshold, Real):
        return np.abs(selectivity) > threshold
    elif isinstance(threshold, xr.DataArray):
        has_pos, has_neg = False, False
        if 'pos' in threshold.coords['tail']:
            has_pos = True
            above = selectivity > threshold.sel(tail='pos')

        if 'neg' in threshold.coords['tail']:
            has_neg = True
            below = selectivity < threshold.sel(tail='neg')

        has_both = has_pos and has_neg
        selected = (
            (above | below) if has_both else
            above if has_pos else
            below
        )
        return selected
    else:
        raise ValueError('Threshold must be a float or xarray.DataArray.')


def compute_percent_selective(selectivity, threshold=None, dist=None,
                              percentile=None, tail='both', groupby=None):
    '''
    Selectivity can be:
    * boolean xarray (already thresholded)
    * xarray with the selectivity statistic (requires threshold argument)
    * xarray.Dataset containing the selectivity statistic, the threshold and
        the null distribution (will be thrsholded, unless percentile is
        defined)

    if percentiles is not defined (default None) and selectivity is not already
    a boolean array, threshold must be defined. Either in the threshold keyword
    argument or in the selectivity xarray.Dataset as 'thresh' variable.
    '''
    import xarray as xr

    # selectivity has to be DataArray or Dataset
    if not isinstance(selectivity, (xr.DataArray, xr.Dataset)):
        raise TypeError('`selectivity` must be an xarray.DataArray or '
                        f'xarray.Dataset. Got {type(selectivity)}.')

    # test that dims are ['cell'] or ['cell', 'time']
    if not selectivity.dims[0] == 'cell':
        raise ValueError('Selectivity must have "cell" as the first dimension')

    has_perc = percentile is not None
    has_dist = dist is not None

    if isinstance(selectivity, xr.Dataset):
        if 'thresh' in selectivity and threshold is None and not has_perc:
            threshold = selectivity['thresh']
        if 'dist' in selectivity and dist is None:
            has_dist = True
            dist = selectivity['dist']

        selectivity = selectivity['stat']

    if has_perc:
        assert has_dist, ('percentile threshold requires passing a null '
                          'distribution in the "dist" argument or dist '
                          'variable being present in the selectivity '
                          'xarray.Dataset.')
        from .stats import find_percentile_threshold
        threshold = find_percentile_threshold(dist, percentile, tail=tail)

    # if no threshold at this point - assume selectivity is already bool
    if threshold is None:
        assert selectivity.dtype == bool, ('If no threshold or percentile is '
                                           'passed, the selectivity must be a '
                                           'boolean array.')
        sel = selectivity
        perm_sel = None
    else:
        sel = threshold_selectivity(selectivity, threshold)
        if has_dist:
            # if we have a reference distribution (null), we can do the same
            # for the permuted data
            perm_sel = threshold_selectivity(dist, threshold)
        else:
            perm_sel = None
    n_cells = len(selectivity.cell)

    n_total = sel.copy()
    n_total.values = np.ones(n_total.shape, dtype=int)

    if groupby is not None:
        n_tot = n_total.groupby(groupby).sum(dim='cell')
        n_sig = sel.groupby(groupby).sum(dim='cell')
        if has_dist and threshold is not None:
            n_sig_perm = perm_sel.groupby(groupby).sum(dim='cell')
    else:
        n_tot = n_total.sum(dim='cell')
        n_sig = sel.sum(dim='cell')
        if has_dist and threshold is not None:
            n_sig_perm = perm_sel.sum(dim='cell')

    perc_sel = (n_sig / n_tot) * 100.

    if has_dist and threshold is not None:
        from .stats import find_percentile_threshold
        perc_sel_perm = (n_sig_perm / n_tot) * 100.
        perm_thresh = find_percentile_threshold(
            perc_sel_perm, percentile=95, tail='pos', perm_dim=0
        )
        # TODO - return Dataset
        return perc_sel, perm_thresh, perc_sel_perm
    else:
        return perc_sel


# TODO: create apply_dict function (with out_type='dict' or 'xarray' etc.)
def compute_selectivity_multisession(frate, compare=None, select=None,
                                     n_perm=1_000, n_jobs=1):
    import xarray as xr
    assert isinstance(frate, dict)

    all_results = list()
    sessions = list(frate.keys())
    for ses in sessions:
        fr = frate[ses]

        # trial selection
        if select is not None:
            fr = fr.query({'trial': select})

        fr = assign_session_coord(fr, ses, dim_name='cell', ses_name='sub')

        results = compute_selectivity_continuous(
            fr, compare=compare, n_perm=n_perm, n_jobs=n_jobs)
        all_results.append(results)

    # concatenate
    all_results = xr.concat(all_results, dim='cell')

    return all_results


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
