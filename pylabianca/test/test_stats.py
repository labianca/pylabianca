import numpy as np
import xarray as xr
import pytest

from functools import partial
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_rel, ttest_ind, f_oneway

import pylabianca as pln


def test_permutation_test():
    # current borsar does not have ttest_ind_no_p with equal_var=False
    ttest_ind_eq = partial(ttest_ind, equal_var=True)

    arr1 = np.random.rand(15, 10)
    arr2 = np.random.rand(15, 10)
    arr3 = np.random.rand(12, 10)
    arr4 = np.random.rand(17, 10)

    is_paired = [True, False, False]
    array_comb = [(arr1, arr2), (arr1, arr3), (arr2, arr3, arr4)]
    stat_functions = [ttest_rel, ttest_ind_eq, f_oneway]

    for arrays, paired, stat_fun in zip(array_comb, is_paired, stat_functions):

        stats_perm, pvals_perm = pln.stats.permutation_test(
            *arrays, paired=paired, return_distribution=False)

        stats, pvals = stat_fun(*arrays)

        np.testing.assert_almost_equal(stats_perm, stats)
        pval_abs_diff = np.abs(pvals - pvals_perm)

        assert (pval_abs_diff < 0.25).all()
        assert (pval_abs_diff < 0.1).mean() > 0.6


def test_permutation_test_n_perm_0():
    arr1 = np.random.rand(10, 20)
    arr2 = np.random.rand(15, 20)
    stats = pln.stats.permutation_test(arr1, arr2, n_perm=0)

    assert isinstance(stats, np.ndarray)
    assert stats.shape == (20,)


def test_cluster_based_test_from_permutations():
    from borsar import Clusters

    n_trials, n_times = 50, 100
    times = np.linspace(-0.5, 1.5, num=n_times)
    data = np.random.rand(n_trials, n_times)

    # create conditions and add effect to one of them
    conditions = np.array([1] * 25 + [2] * 25)
    np.random.shuffle(conditions)
    effect_cond_mask = conditions == 1
    data[effect_cond_mask, 35:55] += 1

    # smooth the data
    data = gaussian_filter1d(data, sigma=20)

    # create xarray
    arr = xr.DataArray(
        data, dims=['trial', 'time'],
        coords={'time': times, 'cond': ('trial', conditions)}
    )

    # standard cluster-based test
    n_permutations = 250
    _, clst, _ = pln.stats.cluster_based_test(
        arr, compare='cond', n_permutations=n_permutations,
        return_clusters=False)

    # compute the effect
    cond = conditions.copy()
    msk = cond == 1
    stat, _ = ttest_ind(data[msk, :], data[~msk, :])

    # compute stats from permuted data
    stat_perm = np.zeros((n_permutations, n_times))
    for perm_idx in range(n_permutations):
        np.random.shuffle(cond)
        msk = cond == 1
        this_stat, _ = ttest_ind(data[msk, :], data[~msk, :])
        stat_perm[perm_idx, :] = this_stat

    # turn to xarrays
    stat = xr.DataArray(stat, dims=['time'], coords={'time': times})
    stat_perm = xr.DataArray(stat_perm, dims=['perm', 'time'],
                             coords={'time': times})

    # compute test from permuted stats
    with pytest.warns(FutureWarning, match='will change in the next version'):
        clst2, _, pval2 = pln.stats.cluster_based_test_from_permutations(
            stat, stat_perm)
    pval2_unsorted = pval2.copy()

    # sort for comparison
    srt_idx = pval2.argsort()
    pval2 = pval2[srt_idx]
    clst2 = [clst2[idx] for idx in srt_idx]

    assert (clst2[0] == clst[0]).mean() >= 0.9

    # test Clusters output with metadata
    clst = pln.stats.cluster_based_test_from_permutations(
        stat, stat_perm, return_clusters=True, tail='both')
    assert isinstance(clst, Clusters)
    assert clst.dimnames == ['time']
    np.testing.assert_array_equal(clst.dimcoords[0], times)
    np.testing.assert_array_equal(clst.stat, stat.values)
    np.testing.assert_array_equal(clst.pvals, pval2_unsorted)

    assert clst.description['tail'] == 'both'
    assert clst.description['percentile'] == 5
    assert clst.description['threshold'] is None
    assert clst.description['n_permutations'] == n_permutations
    assert 'perm_dim' not in clst.description


def test_find_percentile_threshold():
    n_perm, n_cells, n_times = 100, 10, 67
    data = np.random.rand(n_perm, n_cells, n_times)
    data = xr.DataArray(data, dims=['perm', 'cell', 'time'])

    # pos tail
    # --------
    thresh = pln.stats.find_percentile_threshold(
        data, percentile=2.5, tail='pos', perm_dim=0, as_xarray=True
    )

    # currently there is always tail dimension:
    assert thresh.shape == (1, n_cells, n_times)
    assert thresh.dims == ('tail', 'cell', 'time')
    assert thresh.coords['tail'][0].item() == 'pos'
    assert (thresh.data == np.percentile(data, [97.5], axis=0)).all()

    # neg tail
    # --------
    thresh = pln.stats.find_percentile_threshold(
        data, percentile=2.3, tail='neg', perm_dim=0, as_xarray=True
    )

    assert thresh.shape == (1, n_cells, n_times)
    assert thresh.dims == ('tail', 'cell', 'time')
    assert thresh.coords['tail'][0].item() == 'neg'
    assert (thresh.data == np.percentile(data, [2.3], axis=0)).all()


def test_cluster_based_test_return_clusters_object():
    from borsar import Clusters
    from pylabianca.testing import gen_random_xarr

    n_trials, n_cells, n_times = 40, 35, 60
    arr = gen_random_xarr(n_cells, n_trials, n_times,
                        trial_condition_levels=[1, 2])

    effect_cond_mask = arr.coords['cond'] == 1
    has_eff = np.random.rand(n_cells) < 0.7
    idx = np.ix_(has_eff, effect_cond_mask, np.arange(20, 35))
    arr.data[idx] += 0.2

    np.random.seed(12)
    args = dict(compare='cond', n_permutations=100, progress=False)
    with pytest.warns(FutureWarning, match='will change in the next version'):
        stat, clusters, pval = pln.stats.cluster_based_test( arr, **args)

    np.random.seed(12)
    clst = pln.stats.cluster_based_test(arr, return_clusters=True, **args)

    assert isinstance(clst, Clusters)
    assert clst.dimnames == ['cell', 'time']
    np.testing.assert_array_equal(clst.dimcoords[0], arr.cell.values)
    np.testing.assert_array_equal(clst.dimcoords[1], arr.time.values)
    np.testing.assert_allclose(clst.stat, stat)

    # make sure the configuration is stored in .description
    assert clst.description['n_permutations'] == 100
    assert clst.description['compare'] == 'cond'
    assert clst.description['levels'] == [1, 2]
    assert clst.description['tail'] == 'both'
    assert not clst.description['paired']

    # the order of clusters may be different (at least the test failed here,
    # the order may be different due to sorting [or re-sorting?], when there
    # are multiple clusters with identical p value)
    # so we compare the number of cluster members (across time) per cell
    n_memb_clst = clst.clusters.any(axis=0).sum(axis=-1)
    n_memb_arr = np.array(clusters).any(axis=0).sum(axis=-1)
    np.testing.assert_array_equal(n_memb_clst, n_memb_arr)

    assert (clst.pvals == pval).all()

    # make sure this also works after aggregation:
    agg = pln.aggregate(arr, groupby='cond')
    args['return_clusters'] = True
    clst = pln.stats.cluster_based_test(agg, **args)
    assert len(clst.dimnames) == 1
    assert clst.dimnames[0] == 'time'

    agg = pln.aggregate(arr.transpose('trial', 'cell', 'time'), groupby='cond')
    clst = pln.stats.cluster_based_test(agg, **args)
    assert len(clst.dimnames) == 1
    assert clst.dimnames[0] == 'time'
    assert clst.description['paired']

    # test raising error when compare coord is not found
    args['compare'] = 'nothing'
    match = '``compare`` \("nothing"\) was not found'
    with pytest.raises(ValueError, match=match):
        clst = pln.stats.cluster_based_test(arr, **args)
