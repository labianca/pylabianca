import numpy as np


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


# ENH: move to sarna/borsar sometime
def cluster_based_test_from_permutations(data, perm_data, tail='both',
                                         adjacency=None):
    '''Performs a cluster-based test from precalculated permutations.
    
    This function should get data ready for cluster-based permutation test
    WITHOUT the need to perform any statistical tests - so if t test between
    boots of two regions are done for example, then this should be done
    OUTSIDE of this function (the function should receive the t values in
    such case), also boot averages should be done OUTSIDE.
    '''
    import xarray as xr
    import borsar

    assert isinstance(data, xr.DataArray)
    assert isinstance(perm_data, xr.DataArray)
    assert tail in ['both', 'pos', 'neg']
    assert 'perm' in perm_data.dims

    percentiles = [97.5, 2.5] if tail == 'both' else [95, 5]
    perm_dim = perm_data.dims.index('perm')
    n_perms = perm_data.shape[perm_dim]
    thresholds = np.percentile(perm_data, percentiles, axis=perm_dim)

    if tail == 'both':
        thresholds = [thresholds[0], thresholds[1]]
    elif tail == 'pos':
        thresholds = [thresholds[0], None]
    elif tail == 'neg':
        thresholds = [None, thresholds[1]]

    # clusters on actual data
    clusters, cluster_stats = borsar.find_clusters(
        data.values, thresholds, backend='borsar', adjacency=adjacency)

    # check which are pos and which neg:
    is_neg = cluster_stats < 0
    cluster_distr_pos = list()
    cluster_distr_neg = list()

    for perm_idx in perm_data.coords['perm'].data:
        perm_stat = perm_data.sel(perm=perm_idx)

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
        # we are essentially performing two test (one for pos
        # and one for neg clusters) so we have to correct...
        cluster_pval = np.minimum(cluster_pval * 2, 1)

    return clusters, cluster_stats, cluster_pval
