import numpy as np
from scipy import stats


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
    n_trials_max = spk.n_trials
    assert compare in spk.metadata.columns
    cnd_values = spk.metadata[compare].values
    # TODO: if cnd_values are string then we'll need to make sure the array is
    #       not object type

    if tmax is None:
        tmax = spk.time_limits[-1]

    return cnd_values, n_trials_max, tmax


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
def ZETA_test_numba(spk, pick, compare='load', tmin=0., tmax=None,
                    n_permutations=100):
    from ._numba import (
        _get_trial_boundaries_numba, get_condition_indices_and_unique_numba)
    from ._zeta_numba import _run_ZETA_numba

    cnd_values, n_trials_max, tmax = _prepare_ZETA_numpy_and_numba(
        spk, compare, tmax)
    condition_idx, n_trials_per_cond, uni_cnd, n_cnd = (
        get_condition_indices_and_unique_numba(cnd_values)
    )

    times, trials, reference_time = _get_times_and_trials(
        spk, pick, tmin, tmax)

    # TRY: put everything below in jit-compiled function
    trial_boundaries, trial_ids = _get_trial_boundaries_numba(
        trials, n_trials_max)
    n_spk_tri = np.diff(trial_boundaries)
    n_trials_real = trial_ids.shape[0]

    fraction_diff = _run_ZETA_numba(
        times, trial_boundaries, trial_ids, n_spk_tri, n_trials_real,
        n_trials_per_cond, condition_idx, n_cnd,
        reference_time, tmax)

    permutations = np.zeros((n_permutations, len(fraction_diff)))
    condition_idx_perm = condition_idx.copy()
    for perm_idx in range(n_permutations):
        np.random.shuffle(condition_idx_perm)
        permutations[perm_idx] = _run_ZETA_numba(
            times, trial_boundaries, trial_ids, n_spk_tri, n_trials_real,
            n_trials_per_cond, condition_idx_perm, n_cnd,
            reference_time, tmax)

    return fraction_diff, permutations


def gumbel(mean, std, x):
    """"Calculate p-value and z-score for maximum value of N samples drawn from Gaussian
           dblP,dblZ = getGumbel(mean,std,x)

                input:
                - mean: mean of distribution of maximum values
                - std: standard deviation of distribution of maximum values
                - x: maximum value to express in quantiles of Gumbel

                output:
                - arrP; p-value for dblX (chance that sample originates from distribution given by mean/std)
                - arrZ; z-score corresponding to P

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
