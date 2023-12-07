import numpy as np
import numba
from numba.extending import overload


@numba.jit(nopython=True)
def _compute_spike_rate_numba(spike_times, spike_trials, time_limits,
                              n_trials, winlen=0.25, step=0.05):
    half_win = winlen / 2
    window_limits = np.array([-half_win, half_win])
    epoch_len = time_limits[1] - time_limits[0]
    n_steps = int(np.floor((epoch_len - winlen) / step + 1))

    fr_tstart = time_limits[0] + half_win
    fr_tend = time_limits[1] - half_win + step * 0.001
    times = np.arange(fr_tstart, fr_tend, step=step)
    frate = np.zeros((n_trials, n_steps))

    for step_idx in range(n_steps):
        winlims = times[step_idx] + window_limits
        msk = (spike_times >= winlims[0]) & (spike_times < winlims[1])
        tri = spike_trials[msk]
        intri, count = _monotonic_unique_counts(tri)
        frate[intri, step_idx] = count / winlen

    return times, frate


@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
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

        if in_batch >= batch_size or idx1 == (n_times - 1):
            these_counts, _ = numba_histogram(distances[1:], bins)
            counts += these_counts
            in_batch = 0
            distances = [0.01]

    return counts


@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
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
