import numpy as np
from numba import jit, njit
from numba.extending import overload


@njit
def _compute_spike_rate_numba(spike_times, spike_trials, time_limits,
                              n_trials, winlen=0.25, step=0.05):
    half_win = winlen / 2
    window_limits = np.array([-half_win, half_win])
    epoch_len = time_limits[1] - time_limits[0]
    n_steps = int(np.floor((epoch_len - winlen) / step + 1))

    fr_t_start = time_limits[0] + half_win
    fr_tend = time_limits[1] - half_win + step * 0.001
    times = np.arange(fr_t_start, fr_tend, step=step)
    frate = np.zeros((n_trials, n_steps))

    for step_idx in range(n_steps):
        winlims = times[step_idx] + window_limits
        msk = (spike_times >= winlims[0]) & (spike_times < winlims[1])
        tri = spike_trials[msk]
        in_tri, count = _monotonic_unique_counts(tri)
        frate[in_tri, step_idx] = count / winlen

    return times, frate


@njit
def _monotonic_unique_counts(values):
    n_val = len(values)
    if n_val == 0:
        return np.array([0], dtype='int64'), np.array([0], dtype='int64')

    uni = np.zeros(n_val, dtype='int64')
    cnt = np.zeros(n_val, dtype='int64')

    idx = 0
    val = values[0]
    uni[0] = val
    current_count = 1

    for next_val in values[1:]:
        if next_val == val:
            current_count += 1
        else:
            cnt[idx] = current_count
            idx += 1
            val = next_val
            uni[idx] = val
            current_count = 1

    cnt[idx] = current_count
    uni = uni[:idx + 1]
    cnt = cnt[:idx + 1]
    return uni, cnt



def numba_compare_times(spk, cell_idx1, cell_idx2, spk2=None):
    times1 = (spk.timestamps[cell_idx1] / spk.sfreq).astype('float64')

    use_spk = spk if spk2 is None else spk2
    times2 = (use_spk.timestamps[cell_idx2] / spk.sfreq).astype('float64')
    distances = np.zeros(len(times1), dtype='float64')

    res = _numba_compare_times(times1, times2, distances)
    return res


@njit
def _numba_compare_times(times1, times2, distances):
    n_times1 = times1.shape[0]
    n_times2 = times2.shape[0]

    tm2_idx = 0
    for idx1 in range(n_times1):
        time = times1[idx1]
        min_distance = times2.max()
        for idx2 in range(tm2_idx, n_times2):
            this_distance = np.abs(time - times2[idx2])
            if this_distance < min_distance:
                min_distance = this_distance
            else:
                break
        distances[idx1] = min_distance
        tm2_idx = max(idx2 - 1, 0)
    return distances


@njit
def _xcorr_hist_auto_numba(times, bins, batch_size=1_000):
    '''Compute auto-correlation histogram for a single cell.

    [a little more about memory efficiency and using monotonic relationship to
     our advantage etc.]'''
    n_times = times.shape[0]
    distances = [0.01]
    n_bins = len(bins) - 1
    max_lag = max(abs(bins[0]), abs(bins[-1]))
    counts = np.zeros(n_bins, dtype='int')

    in_batch = 0
    for idx1 in range(n_times):
        time1 = times[idx1]

        # move forward till we fall out of max_lag
        max_lag_ok = True
        idx2 = idx1
        while max_lag_ok and idx2 < (n_times - 1):
            idx2 += 1
            distance = times[idx2] - time1
            max_lag_ok = distance <= max_lag

            if max_lag_ok:
                distances.append(distance)
                distances.append(-distance)
                in_batch += 2

        if in_batch >= batch_size + 1 or idx1 == (n_times - 1):
            these_counts, _ = numba_histogram(distances[1:], bins)
            counts += these_counts
            in_batch = 0
            distances = [0.01]

    return counts


@njit
def _xcorr_hist_cross_numba(times, times2, bins, batch_size=1_000):
    '''Compute cross-correlation histogram for a single cell.

    [a little more about memory efficiency and using monotonic relationship to
     our advantage etc.]'''
    n_times = times.shape[0]
    n_times2 = times2.shape[0]
    max_lag = max(abs(bins[0]), abs(bins[-1]))
    distances = [0.01]
    n_bins = len(bins) - 1
    counts = np.zeros(n_bins, dtype='int')

    tm_idx_low = 0
    tm_idx_high = 1
    in_batch = 0

    for idx1 in range(n_times):
        time1 = times[idx1]

        # move forward till tm_idx_high
        new_tm_idx_low = tm_idx_low
        for idx2 in range(tm_idx_low, tm_idx_high):
            if not idx1 == idx2:
                distance = times2[idx2] - time1

                if distance < -max_lag:
                    new_tm_idx_low = idx2
                else:
                    distances.append(distance)
                    in_batch += 1

        tm_idx_low = new_tm_idx_low
        max_lag_ok = True
        while max_lag_ok and idx2 < (n_times2 - 1):
            idx2 += 1
            if not idx1 == idx2:
                distance = times2[idx2] - time1
                max_lag_ok = distance <= max_lag

                if max_lag_ok:
                    tm_idx_high = idx2
                    distances.append(distance)
                    in_batch += 1

        if in_batch >= batch_size + 1 or idx1 == (n_times - 1):
            these_counts, _ = numba_histogram(distances[1:], bins)
            counts += these_counts
            in_batch = 0
            distances = [0.01]

    return counts


@njit
def compute_bin(x, bin_edges):
    '''Copied from https://numba.pydata.org/numba-examples/examples/density_estimation/histogram/results.html'''
    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin

    bin = int(n * (x - a_min) / (a_max - a_min))

    if bin < 0 or bin >= n:
        return None
    else:
        return bin


@njit
def numba_histogram(a, bin_edges):
    '''Copied from https://numba.pydata.org/numba-examples/examples/density_estimation/histogram/results.html'''
    n_bins = len(bin_edges) - 1
    hist = np.zeros(n_bins, dtype=np.intp)

    # for x in a.flat:
    for x in a:
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += 1

    return hist, bin_edges


# FIXME: this function assumes non-overlapping epochs
@njit
def _epoch_spikes_numba(timestamps, event_times, tmin, tmax):
    trial_idx = [-1]
    n_in_trial = [0]
    time = [np.array([0.2])]

    t_idx_low = 0
    t_idx_hi = 0
    n_spikes = len(timestamps)
    n_epochs = event_times.shape[0]
    this_epo = (timestamps[0] < (event_times + tmax)).argmax()
    epo_indices = np.arange(this_epo, n_epochs, dtype=np.int16)

    for epo_idx in epo_indices:
        # find spikes that fit within the epoch
        still_looking = True
        event_time = event_times[epo_idx]
        t_low = event_time + tmin
        t_high = event_time + tmax
        current_idx = t_idx_hi if t_idx_hi < t_low else t_idx_low

        while still_looking and current_idx < n_spikes:
            if timestamps[current_idx] >= t_low:
                t_idx_low = current_idx
                still_looking = False
            current_idx += 1

        still_looking = True
        while still_looking and current_idx < n_spikes:
            if timestamps[current_idx] >= t_high:
                t_idx_hi = current_idx
                still_looking = False
            current_idx += 1

        # select these spikes and center wrt event time
        tms = timestamps[t_idx_low:t_idx_hi] - event_time
        n_spk_in_tri = len(tms)
        if n_spk_in_tri > 0:
            time.append(tms)
            trial_idx.append(epo_idx)
            n_in_trial.append(n_spk_in_tri)

    trial = create_trials_from_short(trial_idx[1:], n_in_trial[1:])
    time = concat_times(time[1:], n_in_trial[1:])

    return trial, time


@njit
def create_trials_from_short(trial_idx, n_in_trial):
    n_all = sum(n_in_trial)
    trial = np.empty(n_all, dtype=np.int16)
    idx = 0
    for tri_idx, n_fill in zip(trial_idx, n_in_trial):
        idx_end = idx + n_fill
        trial[idx:idx_end] = tri_idx
        idx = idx_end

    return trial


@njit
def concat_times(times, n_in_trial):
    n_all = sum(n_in_trial)
    time = np.empty(n_all, dtype=np.float64)
    idx = 0
    for tms, n_fill in zip(times, n_in_trial):
        idx_end = idx + n_fill
        time[idx:idx_end] = tms
        idx = idx_end

    return time


@njit
def _select_spikes_numba(spikes, trials, tri_sel):
    '''Assumes both trials and tri_sel are sorted.'''
    tri_sel_idx = 0
    current_tri = tri_sel[tri_sel_idx]
    msk = np.zeros(len(trials), dtype='bool')
    for idx, tri in enumerate(trials):
        if tri < current_tri:
            continue

        if tri == current_tri:
            msk[idx] = True
        elif tri > current_tri:
            too_low = True
            while too_low:
                tri_sel_idx += 1
                current_tri = tri_sel[tri_sel_idx]
                too_low = tri > current_tri
            if tri == current_tri:
                msk[idx] = True

    return spikes[msk]


@njit
def _monotonic_find_first(values, find_val):
    n_val = values.shape[0]
    for idx in range(n_val):
        if values[idx] == find_val:
            return idx
    return -1


def _get_trial_boundaries(spk, cell_idx):
    return _get_trial_boundaries_numba(spk.trial[cell_idx], spk.n_trials)


@njit
def _get_trial_boundaries_numba(trials, n_trials):
    '''
    Numba implementation of get_trial_boundaries.

    Parameters
    ----------
    trials : np.ndarray
        Trial indices for each spike.
    n_trials : int
        Number of trials (actual number of trials, not the number of trials
        that spikes of given cell appear in).

    Returns
    -------
    trial_boundaries : np.ndarray
        Spike indices where trials start.
    trial_ids : np.ndarray
        Trial indices (useful in case spikes did not appear in some of the
        trials for given cell).
    '''
    n_spikes = trials.shape[0]
    trial_boundaries = np.zeros(n_trials + 1, dtype='int32')
    trial_ids = np.zeros(n_trials + 1, dtype='int32')
    idx = -1
    boundaries_idx = -1
    prev_trial = -1
    while idx < (n_spikes - 1):
        idx += 1
        this_trial = trials[idx]
        if this_trial > prev_trial:
            boundaries_idx += 1
            trial_ids[boundaries_idx] = this_trial
            trial_boundaries[boundaries_idx] = idx
            prev_trial = this_trial

    boundaries_idx += 1
    trial_ids = trial_ids[:boundaries_idx]
    boundaries_idx += 1
    trial_boundaries = trial_boundaries[:boundaries_idx]
    trial_boundaries[-1] = n_spikes

    return trial_boundaries, trial_ids


@njit
def get_condition_indices_and_unique_numba(cnd_values):
    n_trials = cnd_values.shape[0]
    uni_cnd = np.unique(cnd_values)
    n_cnd = uni_cnd.shape[0]
    cnd_idx_per_tri = np.zeros(n_trials, dtype='int32')
    n_trials_per_cond = np.zeros(n_cnd, dtype='int32')

    for idx in range(n_trials):
        cnd_val = cnd_values[idx]
        cnd_idx = _monotonic_find_first(uni_cnd, cnd_val)
        cnd_idx_per_tri[idx] = cnd_idx
        n_trials_per_cond[cnd_idx] += 1

    return cnd_idx_per_tri, n_trials_per_cond, uni_cnd, n_cnd


@njit
def depth_of_selectivity_numba(arr, groupby):
    avg_by_cond = groupby_mean(arr, groupby)
    n_categories = avg_by_cond.shape[0]
    selectivity = depth_of_selectivity_numba_low_level(
        avg_by_cond, n_categories
    )

    return selectivity, avg_by_cond


@njit
def depth_of_selectivity_numba_low_level(avg_by_cond, n_categories):
    r_max = max_2d_axis_0(avg_by_cond)
    numerator = n_categories - (avg_by_cond / r_max).sum(axis=0)
    return numerator / (n_categories - 1)


@njit
def groupby_mean(arr, groupby):
    cnd_idx_per_tri, n_trials_per_cond, _, n_cnd = (
        get_condition_indices_and_unique_numba(groupby)
    )
    avg_by_cnd = _groupby_mean_low_level(
        arr, cnd_idx_per_tri, n_trials_per_cond, n_cnd)
    return avg_by_cnd


@njit
def max_2d_axis_0(arr):
    out = np.zeros(arr.shape[1], dtype=arr.dtype)
    for idx in range(arr.shape[1]):
        out[idx] = arr[:, idx].max()
    return out


@njit
def _groupby_mean_low_level(arr, cnd_idx_per_tri, n_trials_per_cond, n_cnd):
    n_trials = arr.shape[0]
    nd2 = arr.shape[1]
    avg_by_cnd = np.zeros((n_cnd, nd2), dtype=arr.dtype)
    for idx in range(n_trials):
        cnd_idx = cnd_idx_per_tri[idx]
        avg_by_cnd[cnd_idx] += arr[idx]

    for cnd_idx in range(n_cnd):
        avg_by_cnd[cnd_idx] /= n_trials_per_cond[cnd_idx]

    return avg_by_cnd
