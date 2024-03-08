import numpy as np


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