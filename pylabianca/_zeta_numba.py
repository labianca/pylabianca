import numpy as np
from numba import njit

from ._numba import _get_trial_boundaries_numba, var_2d_axis_0


# try compiling this too
def ZETA_numba_2cond(times, trials, reference_time,  n_trials,
                     n_trials_per_cond, condition_idx, n_conditions,
                     rnd, n_samples):
    trial_boundaries, trial_ids = _get_trial_boundaries_numba(
        trials, n_trials)
    n_spikes_per_trial = np.diff(trial_boundaries)
    n_trials_nonempty = trial_ids.shape[0]

    fraction_diff = _zeta_numba_2cond(
        times, trial_boundaries, trial_ids, n_spikes_per_trial,
        n_trials_nonempty, n_trials_per_cond, condition_idx, n_conditions,
        reference_time)

    permutations = _permute_zeta_numba_2cond(
        rnd, n_samples, times, trial_boundaries, trial_ids,
        n_spikes_per_trial, n_trials_nonempty, n_trials_per_cond,
        condition_idx, n_conditions, reference_time)

    return fraction_diff, permutations


@njit(cache=True)
def ZETA_numba_Ncond(times, trials, reference_time,  n_trials,
                     n_trials_per_cond, condition_idx, n_conditions,
                     rnd, n_samples):
    trial_boundaries, trial_ids = _get_trial_boundaries_numba(
        trials, n_trials)
    n_spikes_per_trial = np.diff(trial_boundaries)
    n_trials_nonempty = trial_ids.shape[0]

    fraction_diff = _zeta_numba_Ncond(
        times, trial_boundaries, trial_ids, n_spikes_per_trial,
        n_trials_nonempty, n_trials_per_cond, condition_idx, n_conditions,
        reference_time)

    permutations = _permute_zeta_numba_Ncond(
        rnd, n_samples, times, trial_boundaries, trial_ids,
        n_spikes_per_trial, n_trials_nonempty, n_trials_per_cond,
        condition_idx, n_conditions, reference_time)

    return fraction_diff, permutations


# _run_ZETA_numba could:
# - [ ] return max abs value too
# - [ ] this would translate into what permute_zeta_... returns
@njit(cache=True)
def _zeta_numba_2cond(times, trial_boundaries, trial_ids, n_spikes_per_trial,
                      n_trials_nonempty, n_trials_per_cond, condition_idx,
                      n_conditions, reference_time):

    times_per_cond = group_spikes_by_cond_numba(
        times, trial_boundaries, trial_ids, n_spikes_per_trial,
        n_trials_nonempty, condition_idx, n_conditions)

    fraction_diff = cumulative_diff_two_conditions_numba(
        times_per_cond[0], times_per_cond[1],
        n_trials_per_cond[0], n_trials_per_cond[1],
        reference_time)

    return fraction_diff


@njit(cache=True)
def _zeta_numba_Ncond(times, trial_boundaries, trial_ids, n_spikes_per_trial,
                      n_trials_nonempty, n_trials_per_cond, condition_idx,
                      n_conditions, reference_time):

    times_per_cond = group_spikes_by_cond_numba(
        times, trial_boundaries, trial_ids, n_spikes_per_trial,
        n_trials_nonempty, condition_idx, n_conditions
    )

    fraction_diff = cumulative_sel_multi_conditions_numba(
        times_per_cond, n_trials_per_cond, reference_time
    )

    return fraction_diff


@njit(cache=True)
def group_spikes_by_cond_numba(times, trial_boundaries, trial_ids,
                               n_spikes_per_trial, n_trials, condition_idx,
                               n_conditions):
    spikes_per_cnd = np.zeros(n_conditions, dtype='int32')
    tri_cnd = np.zeros(n_trials, dtype='int32')
    for idx in range(n_trials):
        tri = trial_ids[idx]
        cnd_idx = condition_idx[tri]
        tri_cnd[idx] = cnd_idx
        this_n = n_spikes_per_trial[idx]
        spikes_per_cnd[cnd_idx] += this_n

    temp_arr = np.array([0.], dtype=times.dtype)
    out_list = [temp_arr]
    for idx in range(n_conditions):
        out_list.append(np.zeros(spikes_per_cnd[idx], dtype=times.dtype))
    out_list = out_list[1:]

    c_idx = np.zeros(n_conditions, dtype='int32')
    for idx in range(n_trials):
        cnd_idx = tri_cnd[idx]
        bnd1 = trial_boundaries[idx]
        bnd2 = trial_boundaries[idx + 1]

        out_list[cnd_idx][c_idx[cnd_idx]:c_idx[cnd_idx]
                          + n_spikes_per_trial[idx]] = times[bnd1:bnd2]
        c_idx[cnd_idx] += n_spikes_per_trial[idx]

    return out_list


# FURTHER improve - omit concatenate?
@njit(cache=True)
def cumulative_spikes_norm_numba(spikes, reference_time, n_trials):
    '''
    Calculate cumulative spike distribution against trial time (numba version).
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
    zero = np.array([0.])

    n_spikes = len(spikes)
    step = 1 / n_trials
    endpoint = n_spikes / n_trials
    spikes_fraction = np.arange(step, endpoint + 0.5 * step, step)

    tmax = reference_time[-1]
    spikes = np.concatenate((zero, spikes, np.array([tmax])))
    spikes_fraction = np.concatenate(
        (zero, spikes_fraction, np.array([endpoint]))
    )

    spikes_fraction_interpolated = np.interp(
        reference_time, spikes, spikes_fraction
    )

    return spikes_fraction_interpolated


# cumulative_spikes_norm_numba
@njit(cache=True)
def cumulative_diff_two_conditions_numba(
        spikes1, spikes2, n_trials1, n_trials2, reference_time):
    spikes1_all = np.sort(spikes1)
    spikes2_all = np.sort(spikes2)

    fraction1 = cumulative_spikes_norm_numba(
        spikes1_all, reference_time, n_trials1)
    fraction2 = cumulative_spikes_norm_numba(
        spikes2_all, reference_time, n_trials2)

    fraction_diff = fraction1 - fraction2
    fraction_diff -= fraction_diff.mean()

    return fraction_diff


@njit(cache=True)
def cumulative_sel_multi_conditions_numba(spikes_list, n_trials,
                                          reference_time):
    n_cond = n_trials.shape[0]
    len_ref = reference_time.shape[0]
    spikes_all = [np.sort(spk) for spk in spikes_list]

    cumulative_fraction = np.zeros(
        (n_cond, len_ref), dtype=spikes_all[0].dtype
    )
    for idx in range(n_cond):
        cumulative_fraction[idx] = cumulative_spikes_norm_numba(
                spikes_all[idx], reference_time, n_trials[idx]
        )

    fraction_sel = var_2d_axis_0(cumulative_fraction)
    fraction_sel -= fraction_sel.mean()

    return fraction_sel


@njit(cache=True)
def _permute_zeta_numba_2cond(
        rnd, n_samples, times, trial_boundaries, trial_ids,
        n_spikes_per_trial, n_trials_nonempty, n_trials_per_cond,
        condition_idx, n_conditions, reference_time):
    n_permutations = rnd.shape[0]
    permutations = np.zeros((n_permutations, n_samples), dtype=times.dtype)

    for perm_idx in range(n_permutations):
        np.random.seed(rnd[perm_idx])
        condition_idx_perm = np.random.permutation(condition_idx)
        permutations[perm_idx] = _zeta_numba_2cond(
            times, trial_boundaries, trial_ids, n_spikes_per_trial,
            n_trials_nonempty, n_trials_per_cond, condition_idx_perm,
            n_conditions, reference_time)

    return permutations


@njit(cache=True)
def _permute_zeta_numba_Ncond(
        rnd, n_samples, times, trial_boundaries, trial_ids,
        n_spikes_per_trial, n_trials_nonempty, n_trials_per_cond,
        condition_idx, n_conditions, reference_time):
    n_permutations = rnd.shape[0]
    permutations = np.zeros((n_permutations, n_samples), dtype=times.dtype)
    for perm_idx in range(n_permutations):
        np.random.seed(rnd[perm_idx])
        condition_idx_perm = np.random.permutation(condition_idx)
        permutations[perm_idx] = _zeta_numba_Ncond(
            times, trial_boundaries, trial_ids, n_spikes_per_trial,
            n_trials_nonempty, n_trials_per_cond, condition_idx_perm,
            n_conditions, reference_time)

    return permutations
