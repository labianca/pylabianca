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
        x=reference_time, xp=spikes, fp=spikes_fraction #,
        # left=step, right=endpoint
    )

    return spikes_fraction_interpolated


def cumulative_diff_two_conditions(spikes1, spikes2, n_trials1, n_trials2,
                                   reference_time, tmax):
    # introduce minimum jitter to identical spikes
    spikes1_all = np.sort(spikes1)
    spikes2_all = np.sort(spikes2)

    fraction1 = cumulative_spikes_norm(spikes1_all, reference_time, n_trials1)
    fraction2 = cumulative_spikes_norm(spikes2_all, reference_time, n_trials2)

    fraction_diff = fraction1 - fraction2
    return fraction_diff


def run_ZETA(times, trials, reference_time, cnd_values, uni_cnd, tmax):
    times_per_cond, n_trials = group_spikes_by_cond_no_numba(
        times, trials, cnd_values, uni_cnd)

    fraction_diff = cumul_diff_two_conditions(
        times_per_cond[0], times_per_cond[1], n_trials[0], n_trials[1],
        reference_time, tmax)
    return fraction_diff


def group_spikes_by_cond(times, trials, cnd_values, uni_cnd):
    n_trials = list()
    times_per_cond = list()
    for cnd in uni_cnd:
        sel_cnd = np.where(cnd_values == cnd)[0]
        n_trials.append(len(sel_cnd))
        times_per_cond.append(
            _select_spikes_numba(times, trials, sel_cnd))

    return times_per_cond, n_trials


def group_spikes_by_cond_no_numba(times, trials, cnd_values, uni_cnd):
    n_trials = list()
    times_per_cond = list()
    for cnd in uni_cnd:
        sel_cnd = np.where(cnd_values == cnd)[0]
        n_trials.append(len(sel_cnd))
        this_mask = np.in1d(trials, sel_cnd)
        times_per_cond.append(times[this_mask])

    return times_per_cond, n_trials


def get_condition_indices_and_unique(cnd_values):
    uni_cnd = np.unique(cnd_values)
    cnd_idx_per_tri = np.zeros(cnd_values.shape[0], dtype='int32')

    for cnd_idx, cnd in enumerate(uni_cnd):
        msk = cnd_values == cnd
        cnd_idx_per_tri[msk] = cnd_idx

    return cnd_idx_per_tri, uni_cnd


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


def ZETA_test(spk, pick, compare='load', tmin=0., tmax=None,
              n_permutations=100):
    cnd_values, n_trials_max, tmax = _prepare_ZETA_numpy_and_numba(
        spk, compare, tmax)
    cnd_idx_per_tri, uni_cnd = get_condition_indices_and_unique(cnd_values)

    # LOOP over picks
    times, trials, reference_time = _get_times_and_trials(
        spk, pick, tmin, tmax)

    fraction_diff = run_ZETA(times, trials, reference_time, cnd_values,
                             uni_cnd, tmax)

    permutations = np.zeros((n_permutations, len(fraction_diff)))

    for perm_idx in range(n_permutations):
        cnd_perm = cnd_values.copy()
        np.random.shuffle(cnd_perm)
        permutations[perm_idx] = run_ZETA(
            times, trials, reference_time, cnd_perm, uni_cnd, tmax)

    return fraction_diff, permutations


# some of the things done here would be done one level up and passed to a
# private function
def ZETA_test_numba(spk, compare, picks=None, tmin=0., tmax=None,
                    n_permutations=100, significance='gumbel',
                    return_dist=False):
    from ._numba import (
        _get_trial_boundaries_numba, get_condition_indices_and_unique_numba)
    from ._zeta_numba import _run_ZETA_numba, permute_zeta_2cond_diff_numba

    condition_values, n_trials_max, tmax = _prepare_ZETA_numpy_and_numba(
        spk, compare, tmax)
    condition_idx, n_trials_per_cond, uni_cnd, n_cnd = (
        get_condition_indices_and_unique_numba(condition_values)
    )
    picks = _deal_with_picks(spk, picks)
    n_cells = len(picks)

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
        ZETA_numba_2cond(times, trials, reference_time, n_trials_max,
                         n_trials_per_cond, condition_idx, n_cnd,
                         n_permutations, n_samples)

        # center the cumulative diffs and find max(abs) values
        # TODO: could be done earlier (so for numba - within the
        #       compiled function)
        fraction_diff -= fraction_diff.mean()
        permutations -= permutations.mean(axis=-1, keepdims=True)

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

    # %% define Gumbel parameters from mean and std
    # derive beta parameter from std dev
    beta = np.sqrt(6) * std / np.pi

    # derive mode from mean, beta and E-M constant
    mode = mean - beta * np.euler_gamma

    # calculate cum dens at X
    gumbel_cdf = np.exp(-np.exp(-((x - mode) / beta)))

    # define p-value
    gumbel_p = 1 - gumbel_cdf

    # transform to output z-score
    z_stat = -stats.norm.ppf(np.divide(gumbel_p, 2))

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
