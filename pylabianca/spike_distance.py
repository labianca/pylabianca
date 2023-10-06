import itertools
import numpy as np

from . import utils, viz
from .utils import (_deal_with_picks, _turn_spike_rate_to_xarray,
                    _symmetric_window_samples, _gauss_kernel_samples,
                    spike_centered_windows)


# TODO: ``tol=None`` could return the distances without thresholding
# TODO: consider other name like ``find_spike_distances``
# TODO: test for presence of numba and use the numba implementation
#       (could also use ``backend`` argument)
# TODO: multiprocessing could be useful
# TODO: the implementation and the API are suboptimal
# TODO: remove the numpy backend?
def compare_spike_times(spk, cell_idx1, cell_idx2, spk2=None, backend='numba',
                        tol=None):
    '''Test concurrence of spike times for Spikes or SpikeEpochs.

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
    backend : str
        Backend to use for the computation. Currently only 'numba' is
        supported.
    tol : float
        Concurrence tolerance in seconds. Spikes no further that this will be
        deemed co-occurring. Default is ``None``, which means no concurrence
        threshold will be applied and a distance matrix will be returned.

    Returns
    -------
    distance: float | array of float
        If ``tol`` is ``None`` an array of time distances between closest
        spikes of the two cells is returned. More precisely, for each spike
        from cell 1 a distance to closest cell 2 spike is given. If ``tol`` is
        not ``None`` then a percentage of spikes from first cell that concur
        with spikes from the second cell is given.
    '''
    from .spikes import SpikeEpochs, Spikes

    if isinstance(spk, SpikeEpochs):
        raise NotImplementedError('Sorry compare_spike_times does not work '
                                  'with SpikeEpochs yet.')
    elif isinstance(spk, Spikes):
        if backend == 'numba':
            from ._numba import numba_compare_times
            distances = numba_compare_times(spk, cell_idx1, cell_idx2,
                                            spk2=spk2)
            if tol is not None:
                distances = (distances < tol).mean()
            return distances
        else:
            if spk2 is not None:
                raise NotImplementedError(
                    'Sorry compare_spike_times does not support spk2 '
                    'in the numpy backend.'
                )

            # TODO: rework this
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


def compute_spike_coincidence_matrix(spk, spk2=None, tol=0.002, progress=True):
    if progress:
        from tqdm import tqdm

    n_cells = len(spk)
    n_cells2 = n_cells if spk2 is None else len(spk2)

    similarity = np.zeros((n_cells, n_cells2))
    iter_over = tqdm(range(n_cells)) if progress else range(n_cells)

    for cell1 in iter_over:
        for cell2 in range(n_cells2):
            if spk2 is None and cell1 == cell2:
                continue

            simil = compare_spike_times(spk, cell1, cell2, spk2=spk2, tol=tol)
            similarity[cell1, cell2] = simil

    return similarity


def _construct_bins(sfreq, max_lag, bins=None):
    '''Construct bins for cross-correlation histogram.

    Parameters
    ----------
    sfreq : float
        Sampling frequency of the bins. The bin width will be ``1 / sfreq``
        seconds. Used only when ``bins is None``.
    max_lag : float
        Maximum lag in seconds. Used only when ``bins is None``.
    bins : numpy array | None
        Array representing edges of the histogram bins. If ``None`` (default)
        the bins are constructed based on ``sfreq`` and ``max_lag``.

    Returns
    -------
    bins : numpy array
        Array representing edges of the histogram bins.
    '''
    if bins is not None:
        return bins
    else:
        smp_time = 1 / sfreq
        bins = np.arange(-max_lag, max_lag + smp_time / 2, step=smp_time)

        # check if middle bin is zero-centered
        bin_width = np.diff(bins[:2])[0]
        mid_bin = np.abs(bins).argmin()
        diff_to_centered = np.abs(bins[mid_bin] - (bin_width / 2))

        if diff_to_centered > (bin_width / 10):
            bins = bins - diff_to_centered
            bins = np.concatenate([bins, [bins[-1] + bin_width]])

    return bins


# TODO: add option to shuffle trials?
# TODO: add option to shift trials?
def _xcorr_hist_trials(spk, cell_idx1, cell_idx2, sfreq=500., max_lag=0.2,
                       bins=None):
    '''
    Compute cross-correlation histogram between two cells of SpikeEpochs.

    Parameters
    ----------
    spk : pylabianca.spikes.SpikeEpochs
        SpikeEpochs object to use.
    sfreq : float
        Sampling frequency of the bins. The bin width will be ``1 / sfreq``
        seconds. Used only when ``bins is None``. Defaults to ``500.``.
    max_lag : float
        Maximum lag in seconds. Used only when ``bins is None``. Defaults
        to ``0.2``.
    bins : numpy array | None
        Array representing edges of the histogram bins. If ``None`` (default)
        the bins are constructed based on ``sfreq`` and ``max_lag``.

    Returns
    -------
    bins : numpy array
        Array representing edges of the histogram bins.
    '''
    from .utils import _get_trial_boundaries

    bins = _construct_bins(sfreq, max_lag, bins=bins)
    max_tri = max([tri.max() for tri in spk.trial])
    xcorr = np.zeros((max_tri + 1, len(bins) - 1), dtype=int)

    # TODO: this could be done once at the beginning
    trial_boundaries1, tri_num1 = _get_trial_boundaries(spk, cell_idx1)
    if cell_idx1 == cell_idx2:
        autocorr = True
        trial_boundaries2, tri_num2 = trial_boundaries1, tri_num1
    else:
        autocorr = False
        trial_boundaries2, tri_num2 = _get_trial_boundaries(spk, cell_idx2)

    idx2 = 0
    for idx1, tri in enumerate(tri_num1):

        if not autocorr:
            # check if the other unit has spikes in this trial
            has_tri = tri in tri_num2[idx2:]
            if has_tri:
                idx2 = np.where(tri_num2[idx2:] == tri)[0][0] + idx2
            else:
                continue

        times1 = spk.time[cell_idx1][
            trial_boundaries1[idx1]:trial_boundaries1[idx1 + 1]]

        if autocorr:
            times2 = times1
        else:
            times2 = spk.time[cell_idx2][
                trial_boundaries2[idx2]:trial_boundaries2[idx2 + 1]]

        time_diffs = times2[:, None] - times1[None, :]
        n_diffs = time_diffs.shape[0]
        ind = np.diag_indices(n_diffs)

        if autocorr:
            time_diffs[ind] = np.nan

        time_diffs = time_diffs.ravel()
        this_hist, _ = np.histogram(time_diffs, bins=bins)
        xcorr[tri, :] = this_hist

    return xcorr, bins


# TODO: max_lag cannot be longer than epoch time window and preferrably
#       it is < 1 / 2 epoch length ...
# TODO: when gauss_fwhm is set, calculate trim before and adjust max_lag
#       so that it is left in after convolution
def xcorr_hist_trials(spk, picks=None, picks2=None, sfreq=500., max_lag=0.2,
                      bins=None, gauss_fwhm=None):
    '''FIXME'''
    from .spikes import SpikeEpochs
    from .utils import _deal_with_picks, _turn_spike_rate_to_xarray
    assert isinstance(spk, SpikeEpochs)

    bins = _construct_bins(sfreq, max_lag, bins=bins)

    picks = _deal_with_picks(spk, picks)
    if picks2 is None:
        # picks and picks2 has to be all combinations of picks
        pick_pairs = itertools.product(picks, picks)
    else:
        picks2 = _deal_with_picks(spk, picks2)
        pick_pairs = itertools.product(picks, picks2)

    cell = list()  # pair name, but set as cell attr in xarray
    cell1_idx, cell2_idx = list(), list()
    cell1_name = list()
    cell2_name = list()
    auto = list()  # is autocorrelation

    # LOOP
    xcorrs = list()
    for idx1, idx2  in pick_pairs:
        # compute xcorr histogram for cell pair
        xcorr, _ = _xcorr_hist_trials(spk, idx1, idx2, sfreq=sfreq,
                                      max_lag=max_lag, bins=bins)

        name1 = spk.cell_names[idx1]
        name2 = spk.cell_names[idx2]
        is_auto = idx1 == idx2
        name = f'{name1} x {name2}'

        cell.append(name)
        cell1_name.append(name1)
        cell2_name.append(name2)
        cell1_idx.append(idx1)
        cell2_idx.append(idx2)

        xcorrs.append(xcorr)

    # stack cell pairs as first dimension
    xcorrs = np.stack(xcorrs, axis=0)

    # calc bin_centers
    bin_widths = np.diff(bins)
    bin_centers = bins[:-1] + bin_widths

    # smooth with gaussian if needed
    # TODO: DRY with _spike_density
    if gauss_fwhm is not None:
        from scipy.signal import correlate
        from .spike_rate import _gauss_sd_from_FWHM

        gauss_sd = _gauss_sd_from_FWHM(gauss_fwhm)
        winlen = gauss_sd * 6
        gauss_sd = gauss_sd * sfreq
        win_smp, trim = _symmetric_window_samples(winlen, sfreq)
        kernel = _gauss_kernel_samples(win_smp, gauss_sd) * sfreq

        bin_centers = bin_centers[trim:-trim]
        xcorrs = correlate(xcorrs, kernel[None, None, :], mode='valid')

    # construct xarr
    xcorrs = _turn_spike_rate_to_xarray(
        bin_centers, xcorrs, spk, cell_names=cell, x_dim_name='lag')
    xcorrs.name = 'count'
    xcorrs.attrs['coord_units'] = {'lag': 's'}

    # add cell1_idx etc.
    xcorrs.assign_coords(
        {'cell1_name': ('cell', cell1_name),
         'cell2_name': ('cell', cell2_name),
         'cell1_idx': ('cell', cell1_idx),
         'cell2_idx': ('cell', cell2_idx)}
    )

    return xcorrs


def _spike_xcorr_density(spk, cell_idx, picks=None, sfreq=500, winlen=0.1,
                         kernel_winlen=0.025):
    from .spike_rate import _spike_density

    # create kernel
    gauss_sd = kernel_winlen / 6 * sfreq
    win_smp, trim = _symmetric_window_samples(kernel_winlen, sfreq)
    kernel = _gauss_kernel_samples(win_smp, gauss_sd) * sfreq

    # calculate spike density
    picks = _deal_with_picks(spk, picks)
    tms, cnt = _spike_density(spk, picks=picks, sfreq=sfreq, kernel=kernel)

    # cut out spike-centered windows
    windows = spike_centered_windows(
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
    windows = windows.transpose('channel', 'spike', 'time')
    xcorr = _turn_spike_rate_to_xarray(
        time, windows, spk, tri=windows.trial.values,
        cell_names=cell_names
    )

    return xcorr


def _spike_xcorr_elephant(spk, cell_idx1, cell_idx2, sfreq=500, winlen=0.1,
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
    cch = _turn_spike_rate_to_xarray(
        lags, cch_list[None, :], spk, cell_names=[cell_name],
        copy_cellinfo=False)
    return cch


def xcorr_signal(x, y, maxlag=None):
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
    from scipy.signal import correlate
    from .viz import check_modify_progressbar

    pbar = check_modify_progressbar(pbar, total=n_shuffles)
    max_lag_smp = int(np.round(sfreq * max_lag))

    time, bin_spk = spk.to_raw(picks=[cell_idx1, cell_idx2], sfreq=sfreq)
    n_trials, n_cells, n_times = bin_spk.shape

    # add maxlag separation between trials
    add_sep = np.zeros(shape=(n_trials, n_cells, max_lag_smp))
    bin_spk = np.concatenate([bin_spk, add_sep], axis=2)
    n_times = bin_spk.shape[-1]

    win, _ = _symmetric_window_samples(gauss_winlen, sfreq=sfreq)
    sd_smp = int(np.round((sfreq * gauss_winlen) / 6))
    gauss = _gauss_kernel_samples(win, sd_smp)

    # unroll to cells x (trials * times)
    concat = bin_spk.transpose(1, 0, 2).reshape(n_cells, n_trials * n_times)

    # turn spikes into continuous signal
    cnt = correlate(concat, gauss[None, :], mode='same')
    lags, corr = xcorr_signal(cnt[0], cnt[1], maxlag=max_lag_smp)
    lags = lags / sfreq

    if n_shuffles > 0:
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
            _, this_corr = xcorr_signal(cnt1, cnt_shuffled, maxlag=max_lag_smp)
            corr_shuffled[shuffle_idx] = this_corr
            pbar.update(1)

        avg_shfl = corr_shuffled.mean(axis=0)
        std_shfl = corr_shuffled.std(axis=0)
    else:
        avg_shfl = 0
        std_shfl = 1

    return lags, (corr - avg_shfl) / std_shfl


def find_coincidence_clusters(similarity, threshold=0.3):
    import borsar

    adj = similarity >= threshold
    suspicious = adj.any(axis=0) | adj.any(axis=1)
    suspicious_idx = np.where(suspicious)[0]
    adj_susp = adj[suspicious_idx[:, None], suspicious_idx[None, :]]

    fake_signal = np.ones(len(suspicious_idx))
    clusters, counts = borsar.cluster.find_clusters(
        fake_signal, 0.5, adjacency=adj_susp, backend='mne')
    return suspicious_idx, clusters, counts


# TODO: clean up zasady comments
# TODO: clean up a bit, many indexing levels make it a bit confusing to read
#       and modify
# TODO: some time, maybe add more control (kwargs)
# TODO: allow to drop one from pair when different channels, different
#       alignments, high waveform similarity (?) - in case of ground ref this
#       may happen...
def drop_duplicated_units(spk, similarity, return_clusters=False,
                          verbose=False, different_alignment_threshold=0.4,
                          different_channel_threshold=0.4):
    # %% zasady
    # podobieństwo 1.0, ta sama elektroda -> duplikat, usuwamy dowolny
    #
    # jeżeli ten sam kanał b duże podobienstwo (np. > 0.5), różny alignment
    # - wybierz neuron, który ma więcej spike'ów? (czyli de facto - wybierz
    # ten, którego mniej spike'ów zawiera się w drugim)

    # duży klaster, wiele elektrod, wysokie podobieństwa (ok > 0.4)
    # oraz wiele par ma wysoką korelację waveformów ->
    # zrobić osort ze średnia referencją
    #
    # inny kanał, bardzo podobny średni waveform (np. cross-corr > 0.9) ->
    # wybierz ten co ma więcej spike'ow
    #
    # gdy tylko kilka neuronów w klastrze:
    # różne kanały, >= 30% ko-spike'ów, wysoki peak kroskorelacji średnich
    # wavefromow -> wybierz ten, co ma więcej spike'ów
    import scipy

    # init drop vector
    drop = np.zeros(len(spk.timestamps), dtype='bool')

    # cluster by similarity over 0.3 threshold
    threshold = min(different_alignment_threshold, different_channel_threshold)
    suspicious_idx, clusters, counts = find_coincidence_clusters(
        similarity, threshold=threshold)

    # go through pairs within each cluster and apply selection rules
    for cluster_idx in range(len(clusters)):
        idxs = suspicious_idx[clusters[cluster_idx]]
        simil_part = similarity[idxs[:, None], idxs[None, :]]

        # 1. remove duplicates (any similarity == 1.)
        msg = 'Removed {}-{} pair - identical units.'
        if (simil_part == 1).any():
            s1, s2 = np.where(simil_part == 1.)
            for ix in range(len(s1)):
                identical = min([s1[ix], s2[ix]])
                drop[idxs[identical]] = True

                if verbose:
                    name1 = spk.cell_names[idxs[s1[ix]]]
                    name2 = spk.cell_names[idxs[s2[ix]]]
                    print(msg.format(name1, name2))

        # 2. pairs with similarity >= 0.4, same channels, different alignment
        info = spk.cellinfo.loc[idxs, :]
        if (simil_part >= different_alignment_threshold).any():
            msg = ('Removed {}-{} pair - same channel, different alignment, '
                   'spike coincidence: {:.3f}.')
            s1, s2 = np.where(simil_part > different_alignment_threshold)
            for ix in range(len(s1)):
                ix1, ix2 = s1[ix], s2[ix]
                same_chan = info.channel.iloc[ix1] == info.channel.iloc[ix2]
                same_align = (info.alignment.iloc[ix1]
                              == info.alignment.iloc[ix2])
                if same_chan and not same_align:
                    s1_in_s2 = simil_part[ix1, ix2]
                    s2_in_s1 = simil_part[ix2, ix1]
                    rel_idx = np.argmax([s1_in_s2, s2_in_s1])
                    drop_idx = [ix1, ix2][rel_idx]
                    drop[idxs[drop_idx]] = True

                    if verbose:
                        name1 = spk.cell_names[idxs[ix1]]
                        name2 = spk.cell_names[idxs[ix2]]
                        this_simil = max([s1_in_s2, s2_in_s1])
                        print(msg.format(name1, name2, this_simil))

        # 2. pairs with similarity > 0.3, different channels,
        #    very similar waveforms

        msg = ('Removed {}-{} pair - different channel, very similar waveform, '
                'spike coincidence: {:.3f}, max waveform xcorr: {:.3f}.')
        s1, s2 = np.where(simil_part >= different_channel_threshold)
        avg_wave = [spk.waveform[idx].mean(axis=0) for idx in idxs]
        for ix in range(len(s1)):
            ix1, ix2 = s1[ix], s2[ix]
            same_chan = info.channel.iloc[ix1] == info.channel.iloc[ix2]
            same_align = info.alignment.iloc[ix1] == info.alignment.iloc[ix2]
            if not same_chan:
                s1_in_s2 = simil_part[ix1, ix2]
                s2_in_s1 = simil_part[ix2, ix1]
                rel_idx = np.argmax([s1_in_s2, s2_in_s1])
                drop_idx = [ix1, ix2][rel_idx]

                # but only if waveform max cross-corr is > 0.9
                wave1 = avg_wave[ix1]
                wave2 = avg_wave[ix2]
                corr = scipy.signal.correlate(wave1, wave2)
                auto1 = (wave1 * wave1).sum()
                auto2 = (wave2 * wave2).sum()
                norm = np.sqrt(auto1 * auto2)
                corr = corr / norm
                corr_maxabs = np.abs(corr).max()

                if corr_maxabs > 0.99:
                    drop[idxs[drop_idx]] = True

                    if verbose:
                        name1 = spk.cell_names[idxs[ix1]]
                        name2 = spk.cell_names[idxs[ix2]]
                        this_simil = max([s1_in_s2, s2_in_s1])
                        print(msg.format(name1, name2, this_simil, corr_maxabs))

    if not return_clusters:
        return drop
    else:
        return drop, clusters, counts, suspicious_idx


# TODO: move to viz
def plot_high_similarity_cluster(spk, similarity, clusters, suspicious_idx,
                                 cluster_idx=0, drop=None, figsize=(14, 9)):
    '''Plot similarity matrix with waveforms in top column and leftmost rows.

    Parameters
    ----------
    spk : Spikes | SpikeEpochs
        Object containing spikes.
    similarity : numpy.array
        Numpy array with coincidence similarity.
    clusters : list of numpy.array
        List of boolean arrays - where each array identifies cluster members
        with ``True`` values.
    suspicious_idx : numpy.array
        Array of indices mapping from elements in clusters to indices of
        units in ``spk`` (FIXME - better description).
    cluster_idx : int
        Which cluster to plot. Defaults to ``0`` (the first cluster).
    drop : numpy.array | None
        Boolean array specifying which units will be dropped. Optional,
        is used to color unit titles, defaults to ``None``, when the titles are
        not colored in red.
    figsize : tuple
        Two-element tuple specifying the size of the figure (in inches, as
        matploltib likes it).

    Returns
    -------
    fig : matplotlib.Figure
        Figure object.
    '''
    import matplotlib.pyplot as plt

    idxs = suspicious_idx[clusters[cluster_idx]]

    n_cells = len(idxs)
    simil_part = similarity[idxs[:, None], idxs[None, :]]

    fig, ax = plt.subplots(nrows=n_cells + 1, ncols=n_cells + 1,
                           figsize=figsize)

    title_fontsize = (12 if n_cells < 9 else 10 if n_cells < 13
                      else 8 if n_cells < 17 else 5)

    for idx, cell_idx in enumerate(idxs):
        spk.plot_waveform(cell_idx, ax=ax[0, idx + 1])
        spk.plot_waveform(cell_idx, ax=ax[idx + 1, 0])

        info = spk.cellinfo.loc[cell_idx, :]
        n_spikes = len(spk.timestamps[cell_idx])
        alg_txt = (f'\n{info.alignment}'
                   if not info.alignment == 'unknown'
                   else '')
        title = (f'{info.channel}\ncluster {info.cluster}' + alg_txt +
                 f'\n{n_spikes} spikes')
        color = (('red' if drop[cell_idx] else 'black') if drop is not None
                 else 'black')

        ax[0, idx + 1].set_title(title, color=color, fontsize=title_fontsize)

    for this_ax in ax.ravel():
        this_ax.set_ylabel('')
        this_ax.set_xlabel('')
        this_ax.set_xticks([])
        this_ax.set_yticks([])

    ax[0, 0].axis(False)
    max_val = simil_part.max()
    for row_idx in range(n_cells):
        for col_idx in range(n_cells):
            if row_idx == col_idx:
                ax[row_idx + 1, col_idx + 1].axis(False)
                continue

            value = simil_part[row_idx, col_idx]
            val_perc = value / max_val
            txt_col = 'black' if val_perc > 0.5 else 'white'
            ax[row_idx + 1, col_idx + 1].text(0.5, 0.5, f'{value:0.3f}',
                                            horizontalalignment='center',
                                            verticalalignment='center',
                                            color=txt_col)

            color = plt.cm.viridis(val_perc)
            ax[row_idx + 1, col_idx + 1].set_facecolor(color)

    return fig
