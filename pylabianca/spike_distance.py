import numpy as np

from .utils import (_deal_with_picks, _turn_spike_rate_to_xarray,
                    _symmetric_window_samples, _gauss_kernel_samples,
                    spike_centered_windows)


# TODO: the implementation and the API are suboptimal
def compare_spike_times(spk, cell_idx1, cell_idx2, tol=0.002):
    '''Test concurrence of spike times for SpikeEpochs.

    Parameters
    ----------
    spk : SpikeEpochs
        SpikeEpochs object.
    cell_idx1 : int
        Index of the first cell to compare. All spikes of this cell will be
        checked.
    cell_idx2 : int
        Index of the second cell to compare. Spikes of the first cell will
        be tested for concurrence with spikes coming from this cell.
    tol : float
        Concurrence tolerance in seconds. Spikes no further that this will be
        deemed coocurring.

    Returns
    -------
    float
        Percentage of spikes from first cell that concur with spikes from the
        second cell.
    '''
    from .spikes import SpikeEpochs, Spikes

    if isinstance(spk, SpikeEpochs):
        tri1, tms1 = spk.trial[cell_idx1], spk.time[cell_idx1]
        tri2, tms2 = spk.trial[cell_idx2], spk.time[cell_idx2]

        n_spikes = len(tri1)
        if_match = np.zeros(n_spikes, dtype='bool')
        for idx in range(n_spikes):
            this_time = tms1[idx]
            this_tri = tri1[idx]
            corresp_tri = np.where(tri2 == this_tri)[0]
            match = False
            if len(corresp_tri) > 0:
                corresp_tm = tms2[corresp_tri]
                match = (np.abs(corresp_tm - this_time) < tol).any()
            if_match[idx] = match
        return if_match.mean()
    elif isinstance(spk, Spikes):
        tms1 = spk.timestamps[cell_idx1] / spk.sfreq
        tms2 = spk.timestamps[cell_idx2] / spk.sfreq
        time_diffs = np.abs(tms1[:, None] - tms2[None, :])
        closest_time1 = time_diffs.min(axis=1)
        return (closest_time1 < tol).mean()


def numpy_compare_times(spk, cell_idx1, cell_idx2):
    tms1 = spk.timestamps[cell_idx1] / spk.sfreq
    tms2 = spk.timestamps[cell_idx2] / spk.sfreq
    time_diffs = np.abs(tms1[:, None] - tms2[None, :])
    closest_time1 = time_diffs.min(axis=1)
    return closest_time1


def numpy_compare_times(spk, cell_idx1, cell_idx2):
    tms1 = spk.timestamps[cell_idx1] / spk.sfreq
    tms2 = spk.timestamps[cell_idx2] / spk.sfreq
    time_diffs = np.abs(tms1[:, None] - tms2[None, :])
    closest_time1 = time_diffs.min(axis=1)
    return closest_time1


def spike_xcorr_density(spk, cell_idx, picks=None, sfreq=500, winlen=0.1,
                        kernel_winlen=0.025):
    from .spike_rate import _spike_density

    # create kernel
    gauss_sd = kernel_winlen / 6 * sfreq
    win_smp, trim = _symmetric_window_samples(kernel_winlen, sfreq)
    kernel = _gauss_kernel_samples(win_smp, gauss_sd) * sfreq

    # calculate spike density
    picks = _deal_with_picks(spk, picks)
    tms, cnt = _spike_density(spk, picks=picks, sfreq=sfreq, kernel=kernel)
    cnt = cnt.transpose((1, 0, 2))

    # cut out spike-centered windows
    windows, tri = spike_centered_windows(
        spk, cell_idx, cnt, tms, sfreq, winlen=winlen)

    # correct autocorrelation if present:
    if cell_idx in picks:
        idx = np.where(np.asarray(picks) == cell_idx)[0][0]
        trim = int((windows.shape[-1] - len(kernel)) / 2)
        windows[idx, :, trim:-trim] -= kernel

    # turn to xarray
    t_per_smp = 1 / sfreq
    win_diff = [-winlen / 2, winlen / 2]
    time = np.arange(win_diff[0], win_diff[1] + 0.01 * t_per_smp,
                     step=t_per_smp)
    cell_names = [spk.cell_names[idx] for idx in picks]
    xcorr = _turn_spike_rate_to_xarray(time, windows, spk, tri=tri,
                                       cell_names=cell_names)

    return xcorr


def spike_xcorr_elephant(spk, cell_idx1, cell_idx2, sfreq=500, winlen=0.1,
                         kernel_winlen=0.025, shift_predictor=False):
    from scipy.signal import correlate
    import quantities as pq
    from elephant.conversion import BinnedSpikeTrain
    from elephant.spike_train_correlation import cross_correlation_histogram

    # create kernel
    gauss_sd = kernel_winlen / 6 * sfreq
    win_smp, trim = _symmetric_window_samples(kernel_winlen, sfreq)
    kernel = _gauss_kernel_samples(win_smp, gauss_sd) * sfreq

    # bin spikes
    binsize = 1 / sfreq
    spk1 = spk.to_neo(cell_idx1)
    spk2 = spk.to_neo(cell_idx2)
    bst1 = BinnedSpikeTrain(spk1, bin_size=binsize * pq.s)
    bst2 = BinnedSpikeTrain(spk2, bin_size=binsize * pq.s)

    # window length
    win_len = int(winlen / 2 / binsize)

    cch_list = list()
    n_tri = bst1.shape[0]
    if shift_predictor:
        n_tri -= 1

    # iterate through trials
    for tri in range(n_tri):
        if not shift_predictor:
            tri1, tri2 = tri, tri
        else:
            tri1, tri2 = tri, tri + 1

        cch, lags = cross_correlation_histogram(
            bst1[tri1], bst2[tri2], window=[-win_len, win_len], kernel=kernel)
        cch_list.append(np.array(cch)[:, 0])

    # add last trial if shift predictor
    if shift_predictor:
        cch, lags = cross_correlation_histogram(
            bst1[-1], bst2[-2], window=[-50, 50], kernel=kernel)
        cch_list.append(np.array(cch)[:, 0])

    cch_list = np.stack(cch_list, axis=0)
    lags = lags * binsize

    cell_name = '{}-{}'.format(spk.cell_names[cell_idx1],
                               spk.cell_names[cell_idx2])
    cch = _turn_spike_rate_to_xarray(lags, cch_list[None, :], spk,
                                     cell_names=[cell_name])
    return cch