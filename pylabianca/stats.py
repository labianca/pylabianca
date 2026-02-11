import numpy as np


# TODO: move to borsar
# CONSIDER: n_perm=0 returns just the statistic, it will make selectivity
#           code a bit easier
def permutation_test(*arrays, paired=False, n_perm=1_000, progress=False,
                     return_pvalue=True, return_distribution=True,
                     permutation_vectors=False, n_jobs=1):
    '''Perform permutation test on the data.

    Parameters
    ----------
    *arrays : array_like
        The arrays for which the permutation test should be performed. The
        observations are assumed to be in the first dimension. This dimension
        is permuted.
    paired : bool
        Whether a paired (repeated measures) or unpaired test should be used.
    n_perm : int
        Number of permutations to perform. If ``0`` then only the statistic is
        returned. Defaults to ``1_000``.
    progress : bool
        Whether to show progress bar.
    return_pvalue : bool
        Whether to return p values.
    return_distribution : bool
        Whether to return the permutation distribution.
    permutation_vectors : bool
        Whether to return permutation vectors that allow to reconstruct
        condition labels for each permutation. This is useful when
        defining selectivity using multiple criteria apart from result of
        one statistical test (for example: multiple tests or additional
        criteria, like increase in post-stimulus firing rate for the preferred
        condition.
    n_jobs : int
        Number of jobs to run in parallel.

    Returns
    -------
    out : dict | tuple | numpy.ndarray
        Dictionary with keys ``'stat'``, ``'thresh'``, ``'dist'`` and ``'pval'``
        if both ``return_pvalue`` and ``return_distribution`` are ``True``
        (the default). Otherwise a tuple with the first element being the
        statistic and the second being the p value (if ``return_pvalue`` is
        True). If both ``return_pvalue`` and ``return_distribution`` are False,
        then only the statistic is returned (numpy.ndarray). Only statistic is
        returned also when `n_perm=0`.
    '''
    import borsar

    if permutation_vectors and not return_distribution:
        raise ValueError('permutation_vectors=True requires setting '
                         'return_distribution=True')

    n_groups = len(arrays)
    tail = 'both' if n_groups == 2 else 'pos'
    stat_fun = borsar.stats._find_stat_fun(n_groups=n_groups, paired=paired,
                                            tail=tail)

    has_xarr = all(['DataArray' in str(type(x)) for x in arrays])
    if has_xarr:
        arrays = [x.values for x in arrays]

    if n_perm > 0:
        out = borsar.stats._compute_threshold_via_permutations(
            arrays, paired=paired, tail=tail, stat_fun=stat_fun,
            return_distribution=True, n_permutations=n_perm, progress=progress,
            return_permutations=permutation_vectors, n_jobs=n_jobs)

        if permutation_vectors:
            thresh, dist, perm_vec = out
        else:
            thresh, dist = out
    else:
        return_pvalue = False
        return_distribution = False

    stat = stat_fun(*arrays)

    # CONSIDER: if we perform just one test, we could return a scalar
    # TODO: use borsar functions for this or just put into separate function
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

        if permutation_vectors:
            out['perm_vec'] = perm_vec

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
# ENH: allow for standard arrays (and perm_index) - separate the inner
#      machinery into a separate xarray-agnostic function
# ENH: return borsar.Clusters object as output
def cluster_based_test_from_permutations(data, perm_data, tail='both',
                                         adjacency=None, percentile=5,
                                         threshold=None):
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
    threshold : float
        Threshold to use for clustering. If ``None`` then the threshold is
        calculated using the ``percentile`` parameter.

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

    if threshold is None:
        thresholds = find_percentile_threshold(
            perm_data, perm_dim=perm_dim_name,
            percentile=percentile, tail=tail, as_xarray=False
        )
    else:
        thresholds = threshold

    # clusters on actual data
    clusters, cluster_stats = borsar.find_clusters(
        data.values, thresholds, backend='borsar', adjacency=adjacency)

    # check which are pos and which neg:
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


# CONSIDER: move the xarray "clothing" function somewhere to utils
#           something like this is used in compute_selectivity_continuous
# CONSIDER: especially when using one-tail one may expect that percentile
#           95 works adequately, not 5
#           we later do 100 - percentile, but that seems counter-intuitive
# TODO: use neg, pos thresholds order - this would first require a change in
#       borsar
def find_percentile_threshold(perm_data, percentile=None, tail='both',
                              perm_dim=None, as_xarray=True,
                              check_correctness=True):
    '''Find percentile thresholds for the permutation data.

    Parameters
    ----------
    perm_data : xarray.DataArray
        The permutation data for which the thresholds should be calculated.
        Should contain a dimension named ``'perm'`` or ``'permutation'``. If
        different name is used for the permutation dimension, it should be
        specified using the ``perm_dim`` argument.
    percentile : float, optional
        The "top" percentile value to use for thresholding. For example, a
        percentile of 5 means that the top 5% of the distribution will be used
        as the threshold. The meaning of "top" depends on the
        ``tail`` argument. If ``tail`` is ``'pos'``, then the top 5% means the
        highest 5% of the distribution. If ``tail`` is ``'neg'``, then the top
        5% means the lowest 5% of the distribution. If ``tail`` is ``'both'``,
        then the top 5% means the highest 2.5% and the lowest 2.5% of the
        distribution. This approach is used so that the percentile retains
        the same statistical meaning regardless of the tail used.
        If ``None`` then the default of 5 is used (i.e. the 5th "top"
        percentile). The percentile should be between 0 and 100.
    tail : str, optional
        The tail to use for thresholding. Can be ``'both'``, ``'pos'`` or
        ``'neg'``. The default is ``'both'``.
    perm_dim : str or int, optional
        The name or index of the dimension containing permutations. If
        ``None`` then the function will try to find a dimension named
        ``'perm'`` or ``'permutation'``.
    as_xarray : bool, optional
        Whether to return the thresholds as an xarray.DataArray. If ``False``,
        then a list of two values is returned (for positive and negative tails).
        The default is ``True``.
    check_correctness : bool, optional
        Whether to check for common errors in percentile and tail definition.
        The default is ``True``.

    Returns
    -------
    thresholds : xarray.DataArray or list
        The thresholds for the positive and negative tails. If
        ``as_xarray`` is ``True``, then an xarray.DataArray is returned with
        dimensions ``('tail', ...)`` where ``...`` are the dimensions of the
        original data without the permutation dimension. If ``as_xarray`` is
        ``False``, then a list of two values is returned (for positive and
        negative tails).
    '''
    import xarray as xr

    msg = 'perm_data should be xarray.DataArray'
    assert isinstance(perm_data, xr.DataArray), msg
    percentile = 5 if percentile is None else percentile

    if check_correctness:
        _catch_common_percentile_errors(percentile, perm_data, tail)

    _, perm_dim_idx = _find_dim(perm_data, perm_dim=perm_dim)
    if perm_dim_idx == -1:
        raise ValueError('The array has to contain a dimension named "perm"'
                         ' or "permutation". Otherwise specify the dimension'
                         ' name or index using the perm_dim argument.')

    if tail == 'both':
        percentiles = [100 - (percentile / 2), (percentile / 2)]
        thresholds = np.percentile(perm_data, percentiles, axis=perm_dim_idx)
        if not as_xarray:
            thresholds = [thresholds[0], thresholds[1]]

    elif tail == 'pos':
        thresholds = np.percentile(
            perm_data, [100 - percentile], axis=perm_dim_idx)
        if not as_xarray:
            thresholds = [thresholds[0], None]
    elif tail == 'neg':
        thresholds = np.percentile(perm_data, [percentile], axis=perm_dim_idx)
        if not as_xarray:
            thresholds = [None, thresholds[0]]

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
        if not any(has_dim):
            return None, -1
        perm_dim = dim_names[np.where(has_dim)[0][0]]

    if isinstance(perm_dim, str):
        perm_dim_idx = perm_data.dims.index(perm_dim)
    else:
        # TODO: assert it is an integer
        perm_dim_idx = perm_dim
        perm_dim = perm_data.dims[perm_dim_idx]

    return perm_dim, perm_dim_idx


def _catch_common_percentile_errors(percentile, dist, tail):
    """
    Check for common errors in percentile and tail definition.

    Parameters
    ----------
    percentile : float
        The percentile value to check.
    dist : xarray.DataArray
        The distribution to check against.
    tail : str
        The tail to check.

    Raises
    ------
    ValueError
        If the percentile is not between 0 and 100 or if the tail is not one of
        'both', 'pos', or 'neg'.
    """
    if not (0 <= percentile <= 100):
        raise ValueError('Percentile must be between 0 and 100.')

    if tail not in ['both', 'pos', 'neg']:
        raise ValueError('Tail must be one of "both", "pos", or "neg".')

    # also - warn if percentile is too low (for example 0.05 likely means that
    # the user wanted percentile of 5, and not 0.05)
    if percentile < 1:
        import warnings
        per_text = f'{percentile:.2f} %'
        warnings.warn('Percentile is very low ({per_text}). Remember that it '
                      'is a percentile, not a fraction.')

    # additionally - if tail is 'both' (the default), check if the distribution
    # indeed contain positive and negative values - if not, warn the user
    # that the default tail might not be appropriate in their case
    if tail == 'both':
        if dist.min() >= 0:
            import warnings
            warnings.warn('The distribution does not contain negative values. '
                          'Consider using "pos" tail for thresholding.')
        elif dist.max() <= 0:
            import warnings
            warnings.warn('The distribution does not contain positive values. '
                          'Consider using "neg" tail for thresholding.')
