import numpy as np
import numba
from numba.extending import overload


@numba.jit(nopython=True)
def _compute_spike_rate_numba(spike_times, spike_trials, time_limits,
                              n_trials, winlen=0.25, step=0.05):
    halfwin = winlen / 2
    epoch_len = time_limits[1] - time_limits[0]
    n_steps = int(np.floor((epoch_len - winlen) / step + 1))

    fr_tstart = time_limits[0] + halfwin
    fr_tend = time_limits[1] - halfwin + step * 0.001
    times = np.arange(fr_tstart, fr_tend, step=step)
    frate = np.zeros((n_trials, n_steps))

    for step_idx in range(n_steps):
        winlims = times[step_idx] + np.array([-halfwin, halfwin])
        msk = (spike_times >= winlims[0]) & (spike_times < winlims[1])
        tri = spike_trials[msk]
        intri, count = np.unique(tri, return_counts=True)
        frate[np.asarray(intri), step_idx] = np.asarray(count) / winlen

    return times, frate


# TODO: this np.unique implementation is taken from:
#       https://github.com/numba/numba/pull/2959
#       when it gets merged in numba, I should remove it here
#       (I just modified the `np_unique_counts_impl` to return arrays)
@overload(np.unique)
def np_unique(a, return_counts=False):
    def np_unique_impl(a, return_counts=False):
        b = np.sort(a.ravel())
        head = list(b[:1])
        tail = [x for i, x in enumerate(b[1:]) if b[i] != x]
        return np.array(head + tail)

    def np_unique_wcounts_impl(a, return_counts=False):
        b = np.sort(a.ravel())
        unique = list(b[:1])
        counts = [1 for _ in unique]
        for x in b[1:]:
            if x != unique[-1]:
                unique.append(x)
                counts.append(1)
            else:
                counts[-1] += 1
        return np.array(unique), np.array(counts)

    if not return_counts:
        return np_unique_impl
    else:
        return np_unique_wcounts_impl


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
