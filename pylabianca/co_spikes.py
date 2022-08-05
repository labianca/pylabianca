import numpy as np

from . import utils, viz


def xcorr(x, y, maxlag=None):
    '''Compute un-normalized cross-correlation for required max lag.

    This is equivalent to convolution but without reversing the kernel.

    Parameters
    ----------
    x : numpy.ndarray
        First 1d array to crosscorrelate.
    y : numpy.ndarray
        Second 1d array to crosscorrelate.
    maxlab : int
        Maximum lag in samples.

    Returns
    -------
    lags : numpy.ndarray
        1d array of lags (in samples).
    corr : numpy.ndarray
        1d array of crosscorrelations.
    '''
    from scipy.signal import correlate, correlation_lags

    corr = correlate(x, y)
    lags = correlation_lags(len(x), len(y))

    if maxlag is not None:
        from borsar.utils import find_range

        rng = find_range(lags, [-maxlag, maxlag])
        corr, lags = corr[rng], lags[rng]

    return lags, corr


def shuffled_spike_xcorr(spk, cell_idx1, cell_idx2, sfreq=500,
                         gauss_winlen=0.025, max_lag=0.1, pbar=True,
                         n_shuffles=1_000):
    '''FIXME'''
    pbar = viz.check_modify_progressbar(pbar, total=n_shuffles)
    max_lag_smp = int(np.round(sfreq * max_lag))

    time, bin_spk = spk.to_raw(picks=[cell_idx1, cell_idx2], sfreq=sfreq)
    n_trials, n_cells, n_times = bin_spk.shape

    # add maxlag separation between trials
    add_sep = np.zeros(shape=(n_trials, n_cells, max_lag_smp))
    bin_spk = np.concatenate([bin_spk, add_sep], axis=2)
    n_times = bin_spk.shape[-1]

    win, _ = utils._symmetric_window_samples(gauss_winlen, sfreq=sfreq)
    sd_smp = int(np.round((sfreq * gauss_winlen) / 6))
    gauss = utils._gauss_kernel_samples(win, sd_smp)

    # unroll to cells x (trials * times)
    concat = bin_spk.transpose(1, 0, 2).reshape(n_cells, n_trials * n_times)

    # turn spikes into continuous signal
    cnt = correlate(concat, gauss[None, :], mode='same')
    lags, corr = xcorr(cnt[0], cnt[1], maxlag=max_lag_smp)
    lags = lags / sfreq

    n_lags = len(lags)
    corr_shuffled = np.zeros(shape=(n_shuffles, n_lags))

    cnt1 = cnt[0]
    # shuffle trials n_times
    for shuffle_idx in range(n_shuffles):
        tri = np.arange(n_trials)
        np.random.shuffle(tri)

        # unroll to cells x (trials * times)
        bin_spk_shuffled = bin_spk[tri, 1]
        concat_shuffled = bin_spk_shuffled.reshape(n_trials * n_times)

        # turn spikes into continuous signal
        cnt_shuffled = correlate(concat_shuffled, gauss, mode='same')
        _, this_corr = xcorr(cnt1, cnt_shuffled, maxlag=max_lag_smp)
        corr_shuffled[shuffle_idx] = this_corr
        pbar.update(1)

    avg_shfl = corr_shuffled.mean(axis=0)
    std_shfl = corr_shuffled.std(axis=0)

    return lags, (corr - avg_shfl) / std_shfl
