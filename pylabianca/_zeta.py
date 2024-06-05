import numpy as np
from scipy import stats


from .utils import _deal_with_picks

# top-level API
# zeta_test

def cumulative_spikes_norm(spikes, reference_time, n_trials):
    '''
    Calculate cumulative spike distribution against trial time.
    To compare conditions and detect differences based on total number of
    spikes the distribution is normalized by the number of trials.

    Parameters
    ----------
    spikes : np.ndarray
        Spike times.
    reference_time : np.ndarray
        Time points to interpolate to.
    n_trials : int
        Number of trials.

    Returns
    -------
    spikes_fraction_interpolated : np.ndarray
        Interpolated cumulative spike distribution.
    '''

    n_spikes = len(spikes)
    step = 1 / n_trials
    endpoint = n_spikes / n_trials
    spikes_fraction = np.arange(step, endpoint + 0.5 * step, step)

    tmax = reference_time[-1]
    spikes = np.concatenate([[0.], spikes, [tmax]], axis=0)
    spikes_fraction = np.concatenate(
        [[0.], spikes_fraction, [endpoint]], axis=0)

    spikes_fraction_interpolated = np.interp(
        x=reference_time, xp=spikes, fp=spikes_fraction
    )

    return spikes_fraction_interpolated


def cumulative_diff_two_conditions(spikes1, spikes2, n_trials1, n_trials2,
                                   reference_time):
    # introduce minimum jitter to identical spikes
    spikes1_all = np.sort(spikes1)
    spikes2_all = np.sort(spikes2)

    fraction1 = cumulative_spikes_norm(spikes1_all, reference_time, n_trials1)
    fraction2 = cumulative_spikes_norm(spikes2_all, reference_time, n_trials2)

    fraction_diff = fraction1 - fraction2
    fraction_diff -= fraction_diff.mean()

    return fraction_diff


def diff_func(x):
    return x[0] - x[1]


def var_func(x):
    return np.var(x, axis=0)


def std_func(x):
    return np.std(x, axis=0)


def cumulative_n_conditions(spikes, n_trials, reference_time,
                            reduction=diff_func):
    # introduce minimum jitter to identical spikes
    spikes = [np.sort(spks) for spks in spikes]

    n_cond = len(spikes)
    n_points = reference_time.shape[0]
    fractions = np.zeros((n_cond, n_points))
    for idx, (spks, n_tri) in enumerate(zip(spikes, n_trials)):
        fractions[idx] = cumulative_spikes_norm(
            spks, reference_time, n_tri)

    reduced_fraction = reduction(fractions)
    reduced_fraction -= reduced_fraction.mean()

    return reduced_fraction


def ZETA_2cond(times, trials, reference_time, condition_idx, n_cnd,
               n_permutations, n_samples, reduction=diff_func):
    # CONSIDER turning to list of arrays - one array per trial
    fraction_diff = run_ZETA_2cond(
        times, trials, reference_time, condition_idx, n_cnd,
        reduction=reduction
    )

    permutations = permute_zeta_2cond(
        n_permutations, n_samples, times, trials, condition_idx, n_cnd,
        reference_time, reduction=reduction
    )

    return fraction_diff, permutations


def run_ZETA_2cond(times, trials, reference_time, cnd_values, n_cnd,
                   reduction=diff_func):
    times_per_cond, n_trials = group_spikes_by_cond_no_numba(
        times, trials, cnd_values, n_cnd)

    fraction_diff = cumulative_n_conditions(
        times_per_cond, n_trials, reference_time, reduction=reduction)
    return fraction_diff


def permute_zeta_2cond(
        n_permutations, n_samples, times, trials, condition_idx, n_cnd,
        reference_time, reduction=diff_func):
    permutations = np.zeros((n_permutations, n_samples), dtype=times.dtype)
    condition_idx_perm = condition_idx.copy()
    for perm_idx in range(n_permutations):
        np.random.shuffle(condition_idx_perm)
        permutations[perm_idx] = run_ZETA_2cond(
            times, trials, reference_time, condition_idx_perm, n_cnd,
            reduction=reduction)

    return permutations


def group_spikes_by_cond(times, trials, cnd_values, uni_cnd):
    n_trials = list()
    times_per_cond = list()
    for cnd in uni_cnd:
        sel_cnd = np.where(cnd_values == cnd)[0]
        n_trials.append(len(sel_cnd))
        times_per_cond.append(
            _select_spikes_numba(times, trials, sel_cnd))

    return times_per_cond, n_trials


def group_spikes_by_cond_no_numba(times, trials, cnd_values, n_cnd):
    n_trials = list()
    times_per_cond = list()
    for cnd in range(n_cnd):
        sel_cnd = np.where(cnd_values == cnd)[0]
        n_trials.append(len(sel_cnd))
        this_mask = np.in1d(trials, sel_cnd)
        times_per_cond.append(times[this_mask])

    return times_per_cond, n_trials


def get_condition_indices_and_unique(cnd_values):
    uni_cnd = np.unique(cnd_values)
    n_cnd = uni_cnd.shape[0]
    n_trials = cnd_values.shape[0]

    n_trials_per_cond = np.zeros(n_cnd, dtype='int32')
    cnd_idx_per_tri = np.zeros(n_trials, dtype='int32')

    for cnd_idx, cnd in enumerate(uni_cnd):
        msk = cnd_values == cnd
        n_trials_per_cond[cnd_idx] = msk.sum()
        cnd_idx_per_tri[msk] = cnd_idx

    return cnd_idx_per_tri, n_trials_per_cond, uni_cnd, n_cnd


def _prepare_ZETA_numpy_and_numba(spk, compare, tmax):
    from .spikes import SpikeEpochs
    if not isinstance(spk, SpikeEpochs):
        raise TypeError('Currently ``spk`` must be a SpikeEpochs instance.'
                        ' One-sample ZETA test is not yet implemented. Got '
                        '{} instead.'.format(type(spk)))

    assert compare in spk.metadata.columns
    condition_values = spk.metadata[compare].values

    if condition_values.dtype == 'object':
        condition_values = condition_values.astype('str')

    n_trials_max = spk.n_trials
    if tmax is None:
        tmax = spk.time_limits[-1]

    return condition_values, n_trials_max, tmax


def _get_times_and_trials(spk, pick, tmin, tmax):
    times = spk.time[pick]
    sel_time = (times >= tmin) & (times < tmax)
    times = times[sel_time]
    trials = spk.trial[pick][sel_time]

    reference_time = np.sort(
        np.concatenate([[0.], times, [tmax]], axis=0)
    )

    return times, trials, reference_time


def ZETA(spk, compare, picks=None, tmin=0., tmax=None, backend='numpy',
         n_permutations=100, significance='gumbel', return_dist=False,
         reduction=None):

    if backend == 'numba':
        from ._numba import (get_condition_indices_and_unique_numba
                             as _unique_func)
        from ._zeta_numba import ZETA_numba_2cond, ZETA_numba_ncond
    else:
        _unique_func = get_condition_indices_and_unique

    condition_values, n_trials_max, tmax = _prepare_ZETA_numpy_and_numba(
        spk, compare, tmax)
    condition_idx, n_trials_per_cond, _, n_cnd = (
        _unique_func(condition_values)
    )
    if backend == 'numpy' and reduction is None:
        if n_cnd == 2:
            reduction = diff_func
        elif n_cnd > 2:
            reduction = var_func

    picks = _deal_with_picks(spk, picks)
    n_cells = len(picks)

    if return_dist:
        cumulative_diffs = list()
        permutation_diffs = list()
    real_abs_max = np.zeros(n_cells)
    perm_abs_max = np.zeros((n_cells, n_permutations))

    # TODO: add joblib parallelization if necessary
    for pick_idx, pick in enumerate(picks):
        times, trials, reference_time = _get_times_and_trials(
            spk, pick, tmin, tmax)
        n_samples = reference_time.shape[0]

        # TRY: put everything below in a jit-compiled function
        # TODO: be a bit smarter by selecting the relevant function
        #       beforehand, not checking it every iteration
        if backend == 'numba':
            if n_cnd == 2:
                fraction_diff, permutations = ZETA_numba_2cond(
                    times, trials, reference_time, n_trials_max,
                    n_trials_per_cond, condition_idx, n_cnd, n_permutations,
                    n_samples)
            elif n_cnd > 2:
                fraction_diff, permutations = ZETA_numba_ncond(
                    times, trials, reference_time, n_trials_max,
                    n_trials_per_cond, condition_idx, n_cnd, n_permutations,
                    n_samples)
        elif backend == 'numpy':
            fraction_diff, permutations = ZETA_2cond(
                times, trials, reference_time, condition_idx, n_cnd,
                n_permutations, n_samples, reduction=reduction)

        # center the cumulative diffs and find max(abs) values
        # TODO: could be done earlier (so for numba - within the
        #       compiled function)
        real_abs_max[pick_idx] = np.max(np.abs(fraction_diff))
        perm_abs_max[pick_idx] = np.max(np.abs(permutations), axis=-1)

        if return_dist:
            cumulative_diffs.append(fraction_diff)
            permutation_diffs.append(permutations)

    # asses significance through gumbel or by only comparing to permutations
    z_scores, p_values = compute_pvalues(
        real_abs_max, perm_abs_max, significance=significance)

    if not return_dist:
        return z_scores, p_values
    else:
        other = dict(trace=cumulative_diffs, perm_trace=permutation_diffs,
                     max=real_abs_max, perm_max=perm_abs_max)
        return z_scores, p_values, other


def gumbel(mean, std, x):
    """"Calculate p-value and z-score for maximum value of N samples drawn from
    a Gaussian

    Parameters
    ----------
    mean : float
        Mean of the maxima distribution.
    std : float
        Standard deviation of the maxima distribution.
    x : np.ndarray
        Maximum values to calculate p-values for.

    Returns
    -------
    z_stat : np.ndarray
        Z-scores for the maximum values.
    gumbel_p : np.ndarray
        P-values for the maximum values.
    """

    # derive beta parameter from std dev
    beta = np.sqrt(6) * std / np.pi

    # derive mode from mean, beta and E-M constant
    mode = mean - beta * np.euler_gamma

    # calculate cumulative density at X (or multiple Xs)
    gumbel_cdf = np.exp(-np.exp(-((x - mode) / beta)))

    # define p-value
    gumbel_p = 1 - gumbel_cdf

    # transform to output z-score
    z_stat = -stats.norm.ppf(gumbel_p / 2)

    # approximation for large X
    inf_msk = np.isinf(z_stat)
    if inf_msk.any():
        gumbel_p[inf_msk] = np.exp(mode - x[inf_msk] / beta)
        z_stat[inf_msk] = -stats.norm.ppf(gumbel_p[inf_msk] / 2)

    # return
    return z_stat, gumbel_p


def compute_pvalues(real_abs_max, perm_abs_max, significance='gumbel'):
    if significance in ['gumbel', 'both']:
        perm_mean = np.mean(perm_abs_max, axis=-1)
        perm_std = np.std(perm_abs_max, axis=-1)
        z_scores, p_values_gumbel = gumbel(perm_mean, perm_std, real_abs_max)
    if significance in ['empirical', 'both']:
        p_values_empirical = (perm_abs_max >= real_abs_max[:, None]).mean()

    if significance == 'gumbel':
        p_values = p_values_gumbel
    elif significance == 'empirical':
        p_values = p_values_empirical
        z_scores = None
    elif significance == 'both':
        p_values = {'gumbel': p_values_gumbel, 'empirical': p_values_empirical}

    return z_scores, p_values
