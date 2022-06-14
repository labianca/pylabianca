import numpy as np

from .utils import (_deal_with_picks, _turn_spike_rate_to_xarray,
                    _symmetric_window_samples, _gauss_kernel_samples,
                    spike_centered_windows)


# TODO: ``tol=None`` could return the distances without thresholding
# TODO: consider other name like ``find_spike_distances``
# TODO: test for presence of numba and use the numba implementation
#       (could also use ``backend`` argument)
# TODO: multiprocessing could be useful
# TODO: the implementation and the API are suboptimal
def compare_spike_times(spk, cell_idx1, cell_idx2, backend='numba', tol=None):
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
        thresholding will be applied and a distance matrix will be returned.

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
        # TODO: rework this
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
        if backend == 'numba':
            from ._numba import numba_compare_times
            distances = numba_compare_times(spk, cell_idx1, cell_idx2)
            if tol is not None:
                distances = (distances < tol).mean()
            return distances
        else:
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
    xcorr = _turn_spike_rate_to_xarray(time, windows, spk, tri=windows.trial,
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
    cch = _turn_spike_rate_to_xarray(
        lags, cch_list[None, :], spk, cell_names=[cell_name],
        copy_cellinfo=False)
    return cch


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
    import borsar

    # init drop vector
    drop = np.zeros(len(spk.timestamps), dtype='bool')

    # cluster by similarity over 0.3 threshold
    adj = similarity >= min(different_alignment_threshold,
                            different_channel_threshold)
    suspicious = adj.any(axis=0) | adj.any(axis=1)
    suspicious_idx = np.where(suspicious)[0]
    adj_susp = adj[suspicious_idx[:, None], suspicious_idx[None, :]]

    fake_signal = np.ones(len(suspicious_idx))
    clusters, counts = borsar.cluster.find_clusters(
        fake_signal, 0.5, adjacency=adj_susp, backend='mne')

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


# TODO: clean up
def plot_high_similarity_cluster(spk, similarity, clusters, suspicious_idx,
                                 cluster_idx=0, drop=None):
    import matplotlib.pyplot as plt

    idxs = suspicious_idx[clusters[cluster_idx]]

    n_cells = len(idxs)
    simil_part = similarity[idxs[:, None], idxs[None, :]]

    _, ax = plt.subplots(nrows=n_cells + 1, ncols=n_cells + 1)

    for idx, cell_idx in enumerate(idxs):
        spk.plot_waveform(cell_idx, ax=ax[0, idx + 1])
        spk.plot_waveform(cell_idx, ax=ax[idx + 1, 0])

        info = spk.cellinfo.loc[cell_idx, :]
        n_spikes = len(spk.timestamps[cell_idx])
        title = f'{info.channel}, {info.alignment}\n{n_spikes} spikes'

        if drop is None:
            ax[0, idx + 1].set_title(title)
        else:
            color = 'red' if drop[cell_idx] else 'black'
            ax[0, idx + 1].set_title(title, color=color)

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
