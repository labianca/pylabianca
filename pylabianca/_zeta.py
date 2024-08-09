import numpy as np
from scipy import stats

from .utils import _deal_with_picks, _get_trial_boundaries


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


def diff_func(x):
    return x[0] - x[1]


def var_func(x):
    return np.var(x, axis=0)


def cumulative_n_conditions(spikes, n_trials, reference_time,
                            reduction=diff_func):
    spikes = [np.sort(spk) for spk in spikes]

    n_cond = len(spikes)
    n_points = reference_time.shape[0]
    fractions = np.zeros((n_cond, n_points))
    for idx, (spk, n_tri) in enumerate(zip(spikes, n_trials)):
        fractions[idx] = cumulative_spikes_norm(
            spk, reference_time, n_tri)

    reduced_fraction = reduction(fractions)
    reduced_fraction -= reduced_fraction.mean()

    return reduced_fraction


def ZETA_numpy(times, reference_time, condition_idx, n_cnd,
               n_permutations, n_samples, reduction=diff_func):
    # CONSIDER turning to list of arrays - one array per trial
    fraction_diff = _zeta_numpy(
        times, reference_time, condition_idx, n_cnd,
        reduction=reduction
    )

    permutations = _permute_zeta_numpy(
        n_permutations, n_samples, times, condition_idx, n_cnd,
        reference_time, reduction=reduction
    )

    return fraction_diff, permutations


def _zeta_numpy(times, reference_time, cnd_values, n_cnd,
                reduction=diff_func):
    times_per_cond, n_trials = group_spikes_by_cond_no_numba(
        times, cnd_values, n_cnd)

    fraction_diff = cumulative_n_conditions(
        times_per_cond, n_trials, reference_time, reduction=reduction)
    return fraction_diff


def _permute_zeta_numpy(
        n_permutations, n_samples, times, condition_idx, n_cnd,
        reference_time, reduction=diff_func):
    permutations = np.zeros((n_permutations, n_samples), dtype=times.dtype)
    condition_idx_perm = condition_idx.copy()
    for perm_idx in range(n_permutations):
        np.random.shuffle(condition_idx_perm)
        permutations[perm_idx] = _zeta_numpy(
            times, reference_time, condition_idx_perm, n_cnd,
            reduction=reduction)

    return permutations


def group_spikes_by_cond_no_numba(times, cnd_values, n_cnd):
    n_trials = list()
    times_per_cond = list()
    for cnd in range(n_cnd):
        sel_cnd = np.where(cnd_values == cnd)[0]
        n_trials.append(len(sel_cnd))
        times_per_cond.append(np.concatenate(times[sel_cnd]))

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


# TEST: if tmin, tmax selection before trial boundaries is faster
#       this would require changing _get_trial_boundaries to work on
#       trial id vector
def _to_array_of_arrays(spk_epochs, pick, tmin=None, tmax=None):
    """Convert spike times + trials to array of arrays (one per trial).

    Parameters
    ----------
    spk_epochs : SpikeEpochs
        Spike data.
    pick : int
        Cell index.
    tmin : float, optional
        Minimum time to consider. Default is ``None``.
    tmax : float, optional
        Maximum time to consider. Default is ``None``.

    Returns
    -------
    trial_list : np.ndarray
        Array of arrays with spike times for each trial. ``trial_list[idx]``
        gives array of spike times for trial ``idx``.
    """
    limit_time = tmin is not None and tmax is not None
    max_trials = spk_epochs.n_trials
    trial_list = np.empty(max_trials, dtype='object')
    tri_limits, tri_ids = _get_trial_boundaries(spk_epochs, pick)
    tri_enum = 0
    for tri in range(max_trials):
        if tri in tri_ids:
            lim1, lim2 = tri_limits[tri_enum:tri_enum + 2]
            tri_spikes = spk_epochs.time[pick][lim1:lim2]

            if limit_time:
                sel_time = (tri_spikes >= tmin) & (tri_spikes < tmax)
                tri_spikes = tri_spikes[sel_time]

            trial_list[tri] = tri_spikes
            tri_enum += 1
        else:
            trial_list[tri] = np.array([])

    return trial_list


def _get_times_and_trials(spk, pick, tmin, tmax, subsample, backend):
    """Convert spike data to adequate format for ZETA test with given backend.
    """
    if backend == 'numba':
        times = spk.time[pick]
        sel_time = (times >= tmin) & (times < tmax)
        times = times[sel_time]
        trials = spk.trial[pick][sel_time]

        reference_time = np.sort(
            np.concatenate([[0.], times, [tmax]], axis=0)
        )
    else:
        trials = None
        times = _to_array_of_arrays(spk, pick, tmin=tmin, tmax=tmax)

        reference_time = np.sort(
            np.concatenate([[0.], *times, [tmax]], axis=0)
        )

    if subsample > 1:
        reference_time = reference_time[::subsample]

    return times, trials, reference_time


# TODO: CONSIDER renaming return_dist to return_traces / return_dict, etc.
# MOVE to pylabianca.selectivity ?
def ZETA(spk, compare, picks=None, tmin=0., tmax=None, backend='numpy',
         n_permutations=100, significance='gumbel', return_dist=False,
         subsample=1, reduction=None):
    """ZETA test for comparing cumulative spike distributions between
    conditions.

    Parameters
    ----------
    spk : SpikeEpochs
        Spike data.
    compare : str
        Metadata column name containing condition labels to compare.
    picks : int | str | list of int | list of str,  optional
        Cell indices or names to test. If None, all cells are tested.
    tmin : float, optional
        Minimum time to consider. Default is ``0.``.
    tmax : float, optional
        Maximum time to consider. Default is ``None``, which uses the maximum
        time in the data.
    backend : {'numpy', 'numba'}, optional
        Backend to use. Default is 'numpy'.
    n_permutations : int, optional
        Number of permutations to perform. Default is ``100``.
    significance : {'gumbel', 'empirical', 'both'}, optional
        Method to assess significance. ``'gumbel'`` estimates p-values using
        the Gumbel distribution. ``'empirical'`` compares the real maximum
        value to the permutation distribution. ``'both'`` returns both
        estimates. Default is ``'gumbel'``
    return_dist : bool, optional
        Whether to return the cumulative traces and permutation traces.
        Default is ``False``.
    subsample : int, optional
        Subsample factor for the reference time. Default is ``10``.
    reduction : callable, optional
        Reduction function to apply to the cumulative traces. Default is
        ``None``, which uses the difference function for two conditions and
        variance for more than two conditions.

    Returns
    -------
    z_scores : np.ndarray
        Z-scores - one per cell. Used only if ``significance`` is
        ``'gumbel'``, otherwise ``None``.
    p_values : np.ndarray
        P-values - one per cell.
    other : dict
        Dictionary containing additional output if ``return_dist`` is ``True``.
        Contains the following keys:

        - trace : list of np.ndarray
            Cumulative traces for each cell.
        - perm_trace : list of np.ndarray
            Permutation traces for each cell.
        - max : np.ndarray
            Maximum values for each cell.
        - perm_max : np.ndarray
            Maximum values for each permutation.
        - ref_time : list of np.ndarray
            Reference times for each cell.
    """

    if backend == 'numba':
        from ._numba import (get_condition_indices_and_unique_numba
                             as _unique_func)
    else:
        _unique_func = get_condition_indices_and_unique

    condition_values, n_trials_max, tmax = _prepare_ZETA_numpy_and_numba(
        spk, compare, tmax)
    condition_idx, n_trials_per_cond, _, n_cnd = _unique_func(condition_values)

    if backend == 'numpy' and reduction is None:
        if n_cnd == 2:
            reduction = diff_func
        elif n_cnd > 2:
            reduction = var_func
    elif backend == 'numba':
        if n_cnd == 2:
            from ._zeta_numba import ZETA_numba_2cond as numba_func
        elif n_cnd > 2:
            from ._zeta_numba import ZETA_numba_Ncond as numba_func

    picks = _deal_with_picks(spk, picks)
    n_cells = len(picks)

    if return_dist:
        cumulative_diffs = list()
        permutation_diffs = list()
        reference_times = list()

    real_abs_max = np.zeros(n_cells)
    perm_abs_max = np.zeros((n_cells, n_permutations))

    # TODO: add joblib parallelization if necessary
    for pick_idx, pick in enumerate(picks):
        times, trials, reference_time = _get_times_and_trials(
            spk, pick, tmin, tmax, subsample, backend)
        n_samples = reference_time.shape[0]

        if backend == 'numba':
            fraction_diff, permutations = numba_func(
                times, trials, reference_time, n_trials_max,
                n_trials_per_cond, condition_idx, n_cnd, n_permutations,
                n_samples)
        elif backend == 'numpy':
            fraction_diff, permutations = ZETA_numpy(
                times, reference_time, condition_idx, n_cnd,
                n_permutations, n_samples, reduction=reduction)

        # center the cumulative diffs and find max(abs) values
        # TODO: could be done earlier (so for numba - within the
        #       compiled function)
        real_abs_max[pick_idx] = np.max(np.abs(fraction_diff))
        perm_abs_max[pick_idx] = np.max(np.abs(permutations), axis=-1)

        if return_dist:
            cumulative_diffs.append(fraction_diff)
            permutation_diffs.append(permutations)
            reference_times.append(reference_time)

    # asses significance through gumbel or by only comparing to permutations
    z_scores, p_values = compute_pvalues(
        real_abs_max, perm_abs_max, significance=significance)

    if not return_dist:
        return z_scores, p_values
    else:
        other = dict(trace=cumulative_diffs, perm_trace=permutation_diffs,
                     max=real_abs_max, perm_max=perm_abs_max,
                     ref_time=reference_times)
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

    # transform to z-score
    z_stat = -stats.norm.ppf(gumbel_p / 2)

    # approximation for large X
    inf_msk = np.isinf(z_stat)
    if inf_msk.any():
        gumbel_p[inf_msk] = np.exp(mode - x[inf_msk] / beta)
        z_stat[inf_msk] = -stats.norm.ppf(gumbel_p[inf_msk] / 2)

    # return
    return z_stat, gumbel_p


def compute_pvalues(real_abs_max, perm_abs_max, significance='gumbel'):
    """Compute p-values for the maxima of the real data compared to
    permutations.

    Parameters
    ----------
    real_abs_max : np.ndarray
        Maximum values of the real data.
    perm_abs_max : np.ndarray
        Maximum values of the permutations.
    significance : {'gumbel', 'empirical', 'both'}, optional
        Method to assess significance. ``'gumbel'`` estimates p-values using
        the Gumbel distribution. ``'empirical'`` compares the real maximum
        value to the permutation distribution. ``'both'`` returns both
        estimates. Default is ``'gumbel'``.

    Returns
    -------
    z_scores : np.ndarray | None
        Z-scores - one per cell. Used only if ``significance`` is
        ``'gumbel'`` or ``'both'``, otherwise ``None``.
    p_values : np.ndarray | dict
        P-values - one per cell. If ``significance`` is ``'both'``, returns
        a dictionary with keys ``'gumbel'`` and ``'empirical'``.
    """
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
