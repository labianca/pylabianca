import numpy as np


# TODO: move to borsar
def permutation_test(*arrays, paired=False, n_perm=1000, progress=False,
                     return_pvalue=True, return_distribution=True, n_jobs=1):
    '''Perform permutation test on the data.

    Parameters
    ----------
    *arrays : array_like
        The arrays for which the permutation test should be performed.
    paired : bool
        Whether a paired (repeated measures) or unpaired test should be used.
    n_perm : int
        Number of permutations to perform.
    progress : bool
        Whether to show progress bar.
    return_pvalue : bool
        Whether to return p values.
    return_distribution : bool
        Whether to return the permutation distribution.
    n_jobs : int
        Number of jobs to run in parallel.

    Returns
    -------
    out : dict | tuple
        Dictionary with keys ``'stat'``, ``'thresh'``, ``'dist'`` and ``'pval'``
        if both ``return_pvalue`` and ``return_distribution`` are ``True``
        (the default). Otherwise a tuple with the first element being the
        statistic and the second being the p value (if ``return_pvalue`` is
        True). If both ``return_pvalue`` and ``return_distribution`` are False,
        then only the statistic is returned.
    '''
    import borsar

    n_groups = len(arrays)
    tail = 'both' if n_groups == 2 else 'pos'
    stat_fun = borsar.stats._find_stat_fun(n_groups=n_groups, paired=paired,
                                            tail=tail)

    has_xarr = all(['DataArray' in str(type(x)) for x in arrays])
    if has_xarr:
        arrays = [x.values for x in arrays]

    thresh, dist = borsar.stats._compute_threshold_via_permutations(
        arrays, paired=paired, tail=tail, stat_fun=stat_fun,
        return_distribution=True, n_permutations=n_perm, progress=progress,
        n_jobs=n_jobs)

    stat = stat_fun(*arrays)

    # this does not make sense for > 1d, but we could make sure
    # that if output is 1d, 1-element array, it is returned as a scalar
    #
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
                pval[is_pos] = ((dist[:, is_pos] >= stat[None, is_pos])
                                .mean(axis=0))
                pval[~is_pos] = ((dist[:, ~is_pos] <= stat[None, ~is_pos])
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
        Xarray with spike rate or spike density containing observations as the
        first dimension (for example trials for between-trials analysis or
        cells for between-cells analysis). If you have both cells and trials
        then you should either:
            * selected one cell at a time, via ``frate.isel(cell=0)`` for
              example
            * the trials dimension should be averaged (for example
              ``frate.mean(dim='trial')``) for a between-cell analysis.
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
        Anova F statistics for every time point.
    clusters : list of numpy.ndarray
        List of cluster memberships.
    pval : numpy.ndarray
        List of p values from anova.
    '''
    from borsar.cluster import permutation_cluster_test_array

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


# ENH: move to sarna/borsar sometime
# ENH: allow for standard arrays (and perm_index)
def cluster_based_test_from_permutations(data, perm_data, tail='both',
                                         adjacency=None, percentile=5):
    '''Performs a cluster-based test from precalculated permutations.

    This function should get data ready for cluster-based permutation test
    WITHOUT the need to perform any statistical tests - so if t test between
    boots of two regions are done for example, then this should be done
    OUTSIDE of this function (the function should receive the t values in
    such case), also boot averages should be done OUTSIDE.

    Parameters
    ----------
    data : xarray.DataArray
        Data array with observations as the first dimension (for example
        trials for between-trials analysis or cells for between-cells
        analysis).
    perm_data : xarray.DataArray
        Data array with permutations. Should contain a dimension named
        ``'perm'``.
    tail : str
        Which tail to use for the test. Can be ``'both'``, ``'pos'`` or
        ``'neg'``.
    adjacency : numpy.ndarray
        Adjacency matrix for clustering. If ``None`` then lattice adjacency is
        used.
    percentile : int or float
        Percentile of the permutation distribution to use for thresholding.
        The percentile threshold is calculated for every data point (time point
        for example) separately. The percentile conforms to the tail selected
        using the ``tail`` parameter. When ``tail`` is ``'both'`` then two
        percentiles are calculated (one for positive and one for negative tail
        ). For example, if ``percentile`` is 5, and the tail is ``'both'`` then
        the 2.5th (``percentile / 2``) and 97.5th (``100 - (percentile / 2)``)
        percentiles are used as thresholds. The percentile should be between
        0 and 100.

    Returns
    -------
    clusters : list of numpy.ndarray
        List of cluster memberships.
    cluster_stats : numpy.ndarray
        Cluster statistics.
    cluster_pval : numpy.ndarray
        Cluster p values.
    '''
    import xarray as xr
    import borsar

    assert isinstance(data, xr.DataArray)
    assert isinstance(perm_data, xr.DataArray)

    perm_dim_name, _ = _find_dim(perm_data)
    thresholds = find_percentile_threshold(
        perm_data, perm_dim=perm_dim_name,
        percentile=percentile, tail=tail, as_xarray=False
    )

    # clusters on actual data
    clusters, cluster_stats = borsar.find_clusters(
        data.values, thresholds, backend='borsar', adjacency=adjacency)

    # check which are pos and which neg:
    is_neg = cluster_stats < 0
    cluster_distr_pos = list()
    cluster_distr_neg = list()

    for perm_idx in perm_data.coords[perm_dim_name].data:
        perm_stat = perm_data.sel({perm_dim_name: perm_idx})

        perm_clusters, perm_cluster_stats = borsar.find_clusters(
            perm_stat.values, thresholds, backend='borsar',
            adjacency=adjacency)

        n_perm_clusters = len(perm_clusters)
        perm_is_neg = perm_cluster_stats < 0

        if n_perm_clusters > 0 and perm_is_neg.any():
            best_neg = perm_cluster_stats.min()
        else:
            best_neg = 0
        cluster_distr_neg.append(best_neg)

        if  n_perm_clusters > 0 and (~perm_is_neg).any():
            best_pos = perm_cluster_stats.max()
        else:
            best_pos = 0
        cluster_distr_pos.append(best_pos)

    cluster_distr_pos = np.array(cluster_distr_pos)
    cluster_distr_neg = np.array(cluster_distr_neg)

    cluster_pval = [(cluster_distr_pos >= clst_stat).mean() if clst_stat > 0
                    else (cluster_distr_neg <= clst_stat).mean()
                    for clst_stat in cluster_stats]
    cluster_pval = np.array(cluster_pval)

    if tail == 'both':
        # we are essentially performing two tests (one for pos
        # and one for neg clusters) so we have to correct...
        cluster_pval = np.minimum(cluster_pval * 2, 1)

    return clusters, cluster_stats, cluster_pval


# TODO: ! move the xarray "clothing" function somewhere to utils
#         something like this is used in many places of pylabianca
# TODO: use neg, pos thresholds order - this would first require a change in
#       borsar
# TODO: perm_data has to be DataArray, not Dataset !
def find_percentile_threshold(perm_data, percentile=None, tail='both',
                              perm_dim=None, as_xarray=True):
    assert tail in ['both', 'pos', 'neg']
    percentile = 5 if percentile is None else percentile

    if as_xarray:
        import xarray as xr

    _, perm_dim_idx = _find_dim(perm_data, perm_dim=perm_dim)

    if tail == 'both':
        percentiles = [100 - (percentile / 2), (percentile / 2)]
        thresholds = np.percentile(perm_data, percentiles, axis=perm_dim_idx)
        if not as_xarray:
            thresholds = [thresholds[0], thresholds[1]]

    elif tail == 'pos':
        thresholds = np.percentile(
            perm_data, 100 - percentile, axis=perm_dim_idx)
        if not as_xarray:
            thresholds = [thresholds, None]
    elif tail == 'neg':
        thresholds = np.percentile(perm_data, percentile, axis=perm_dim_idx)
        if not as_xarray:
            thresholds = [None, thresholds[1]]

    if as_xarray:
        tail_coords = ['pos', 'neg'] if tail == 'both' else [tail]
        perm_data_dims = list(perm_data.dims)
        perm_data_dims.pop(perm_dim_idx)

        dims = ['tail'] + perm_data_dims
        coords = {'tail': tail_coords}
        for dim_idx, dim_name in enumerate(perm_data_dims):
            coords[dim_name] = perm_data.coords[dim_name].data

        thresholds = xr.DataArray(thresholds, dims=dims, coords=coords)

    return thresholds


# CONSIDER: move to utils (very similar code is also used somewhere else
#                          in pylabianca)
def _find_dim(perm_data, perm_dim=None):
    if perm_dim is None:
        dim_names = ['perm', 'permutation']
        has_dim = [dim_name in perm_data.dims for dim_name in dim_names]
        assert any(has_dim)
        perm_dim = dim_names[np.where(has_dim)[0][0]]

    if isinstance(perm_dim, str):
        perm_dim_idx = perm_data.dims.index(perm_dim)
    else:
        # TODO: assert it is an integer
        perm_dim_idx = perm_dim
        perm_dim = perm_data.dims[perm_dim_idx]

    return perm_dim, perm_dim_idx
