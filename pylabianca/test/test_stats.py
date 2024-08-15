import numpy as np
from functools import partial
from scipy.stats import ttest_rel, ttest_ind, f_oneway

import pylabianca as pln


def test_permutation_test():
    # current borsar does not have ttest_ind_no_p with equal_var=False
    ttest_ind_eq = partial(ttest_ind, equal_var=True)

    arr1 = np.random.rand(15, 10)
    arr2 = np.random.rand(15, 10)
    arr3 = np.random.rand(12, 10)
    arr4 = np.random.rand(17, 10)

    paireds = [True, False, False]
    array_comb = [(arr1, arr2), (arr1, arr3), (arr2, arr3, arr4)]
    stat_funs = [ttest_rel, ttest_ind_eq, f_oneway]


    for arrays, paired, stat_fun in zip(array_comb, paireds, stat_funs):

        stats_perm, pvals_perm = pln.stats.permutation_test(
            *arrays, paired=paired, return_distribution=False)

        stats, pvals = stat_fun(*arrays)

        np.testing.assert_almost_equal(stats_perm, stats)
        pval_abs_diff = np.abs(pvals - pvals_perm)

        assert (pval_abs_diff < 0.25).all()
        assert (pval_abs_diff < 0.1).mean() > 0.6
