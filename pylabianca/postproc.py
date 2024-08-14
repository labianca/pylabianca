import os
import os.path as op

import numpy as np


def turn_to_percentiles(arr):
    from scipy.stats import percentileofscore
    percentiles = np.vectorize(lambda x: percentileofscore(arr, x))(arr)
    return percentiles / 100.


def plot_scores(spk_sel, score):
    from string import ascii_letters
    from matplotlib import pyplot as plt

    n_uni = len(spk_sel)
    letters = ascii_letters[1:n_uni + 1]
    ax = plt.figure(constrained_layout=True, figsize=(12, 6)).subplot_mosaic(
        'a' * n_uni + '\n' + letters,
         gridspec_kw={'height_ratios': [3, 1]}
    )

    # plot scores
    ax['a'].plot(score, marker='o')

    # plot best score in red
    best_idx = np.nanargmax(score)
    ax['a'].plot(best_idx, score[best_idx], marker='o', markersize=18,
                 markerfacecolor='r')

    # add waveforms at the bottom
    letters = list(letters)
    for ix in range(len(spk_sel)):
        spk_sel.plot_waveform(ix, ax=ax[letters[ix]], labels=False)
        ax[letters[ix]].set_title(f'{len(spk_sel.timestamps[ix])} spikes')
        ax[letters[ix]].set(yticks=[], xticks=[])

    ax['a'].set_ylabel('Weighted score', fontsize=14)
    return ax['a'].figure


def find_coincidence_clusters(similarity, threshold=0.3):
    import borsar

    adj = similarity >= threshold
    suspicious = adj.any(axis=0) | adj.any(axis=1)
    suspicious_idx = np.where(suspicious)[0]
    adj_suspicious = adj[suspicious_idx[:, None], suspicious_idx[None, :]]

    fake_signal = np.ones(len(suspicious_idx))
    clusters, counts = borsar.cluster.find_clusters(
        fake_signal, 0.5, adjacency=adj_suspicious, backend='mne')
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
        matplotlib likes it).

    Returns
    -------
    fig : matplotlib.Figure
        Figure object.
    '''
    import matplotlib.pyplot as plt

    idxs = suspicious_idx[clusters[cluster_idx]]

    n_cells = len(idxs)
    similarity_part = similarity[idxs[:, None], idxs[None, :]]

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
    max_val = similarity_part.max()
    for row_idx in range(n_cells):
        for col_idx in range(n_cells):
            if row_idx == col_idx:
                ax[row_idx + 1, col_idx + 1].axis(False)
                continue

            value = similarity_part[row_idx, col_idx]
            val_perc = value / max_val
            txt_col = 'black' if val_perc > 0.5 else 'white'
            ax[row_idx + 1, col_idx + 1].text(0.5, 0.5, f'{value:0.3f}',
                                            horizontalalignment='center',
                                            verticalalignment='center',
                                            color=txt_col)

            color = plt.cm.viridis(val_perc)
            ax[row_idx + 1, col_idx + 1].set_facecolor(color)

    return fig


def mark_duplicates(spike_data_dir, first_channel, fig_dir=None,
                    data_format='standard', coincidence_threshold=0.1,
                    min_channels=1, weights=None, alignment_sample=94):
    """"
    Mark duplicate units based on coincidence in spike sorting results.

    Osort sorts spikes on each microwire separately, so it is common to have
    duplicated units across microwires. This function calculates coincidence
    between units from each microwire bundle and marks them as duplicates if
    they exceed a given threshold (``coincidence_threshold``). Units within
    each coincidence cluster are then scored based on a set of measures
    (``weights``) and the best unit is chosen. All units from the cluster that
    exceed the coincidence threshold with this selected unit are marked for
    removal. The process is repeated iteratively until no more units within the
    coincidence cluster exceed the threshold. The results are saved in a table
    and figures are produced showing the coincidence matrix and the scores of
    the units.

    Parameters
    ----------
    spike_data_dir : str
        Path to the directory with exported osort spike sorting results.
    first channel : int
        To calculate coincidence only in 8-channel packs (corresponding to
        8 wires coming from one Behnke-Fried electrode), you have to specify
        the first channel. If you leave it as None, the first channel will be
        estimated from selected units, which may be incorrect if you did not
        select any units from the first channel.
    save_fig : bool
        Whether to plot figures.
    fig_dir : str
        Directory to save figures to.
    data format : str
        'standard' or 'mm' (depends on how you exported the curated units).
    coincidence_threshold : float
        Minimum coincidence threshold for coincidence cluster formation.
    min_channels : int
        Minimum number of different-channel units in a coincidence cluster to
        classify it as a duplicate cluster.
    weights : dict | None
        Weights used in calculating the score for each unit (unit with the best
        score is chosen). A dictionary with following fields:
            isi - % inter spike intervals < 3 ms (lower is better)
            n_spikes  - number of spikes
            snr - signal to noise ratio (mean / std) at alignment point
            std - standard deviation calculated separately for each waveform
                  sample and then averaged (lower is better)
            dns - average "perceptual" waveform density (for each time sample
                  20 best pixels from waveform density are chosen and averaged;
                  these values are then averaged across the whole waveform;
                  the values are divided by maximum density to de-bias against
                  waveforms with little spikes and thus make this value closer
                  to the density we see in the figures.
        Defaults to ``{'isi': 0.15, 'n_spikes': 0.35, 'snr': 0.0, 'std': 0.0,
        'dns': 0.5}``.
    alignment_sample : int
        Alignment sample index. Defaults to 94 which should be true for osort
        files.
    """
    import pandas as pd

    from scipy.stats import rankdata
    from tqdm import tqdm
    import h5io

    import pylabianca as pln


    if weights is None:
        weights = {'isi': 0.15, 'n_spikes': 0.35, 'snr': 0.0,
                   'std': 0.0, 'dns': 0.5}

    save_fig = fig_dir is not None
    if save_fig:
        from matplotlib import pyplot as plt
        assert fig_dir is not None

    # TODO - add check that highest channel number is in a pack
    if not op.exists(fig_dir):
        from pathlib import Path
        parent_path = Path(fig_dir).parent
        if not op.exists(parent_path):
            os.mkdir(parent_path)
        os.mkdir(fig_dir)

    # read the file
    print('Reading files - including all waveforms...')
    use_usenegative = False if data_format == 'mm' else True
    spk = pln.io.read_osort(spike_data_dir, waveform=True, format=data_format,
                            use_usenegative=use_usenegative)
    print('done.')

    # remove units with no spikes
    n_spikes = spk.n_spikes()
    no_spikes = n_spikes == 0
    if (no_spikes).any():
        spk.pick_cells(~no_spikes)
        n_spikes = n_spikes[~no_spikes]

    unit_channel = spk.cellinfo.channel.str.slice(1).astype('int')
    last_channel = unit_channel.max()
    n_packs = (last_channel - first_channel + 1) // 8

    # check channel packs
    # -------------------
    # find which wires belong to which wire pack (one electrode)
    # (assuming each wire pack is 8, starting from first_channel)
    packs = list()
    current_ch_idx = first_channel
    for idx in range(n_packs):
        this_pack = np.arange(current_ch_idx, current_ch_idx + 8)
        packs.append(this_pack)
        current_ch_idx += 8

    units_in_pack = [np.where(np.in1d(unit_channel, pack))[0]
                    for pack in packs]

    # calculate measures
    # ------------------
    # isi, fr, snr, std, dns

    print('Calculating FR, SNR, ISI, STD and DNS...')

    n_cells = len(spk)
    spk_epo = spk.to_epochs()

    measure_names = ['n_spikes', 'snr', 'isi', 'std', 'dns']
    measures = {name: np.zeros(n_cells, dtype='float')
                for name in measure_names}
    measures['n_spikes'] = n_spikes

    for ix in tqdm(range(n_cells)):
        # SNR
        waveform_amplitude = np.mean(
            spk.waveform[ix][:, alignment_sample], axis=0)
        waveform_std = np.std(spk.waveform[ix][:, alignment_sample], axis=0)
        measures['snr'][ix] = np.abs(waveform_amplitude) / waveform_std

        # ISI
        isi_val = np.diff(spk_epo.time[ix])
        prop_below_3ms = (isi_val < 0.003).mean()
        measures['isi'][ix] = prop_below_3ms

        # STD
        avg_std = np.std(spk.waveform[ix], axis=0).mean()
        measures['std'][ix] = avg_std

        # DNS
        measures['dns'][ix] = pln.viz.calculate_perceptual_waveform_density(
            spk, ix)

    measures_prc = {name: turn_to_percentiles(measures[name])
                    for name in measure_names}
    print('Done.')


    # compute coincidence
    # -------------------
    # or read from disk if already computed

    fname = 'coincidence_per_pack.hdf5'
    files = os.listdir(spike_data_dir)
    has_similarity = fname in files

    group_idx = 0
    df_columns=['pack', 'group', 'subgroup', 'channel', 'cluster', 'n_spikes',
                'snr', 'isi', 'std', 'dns', 'drop']

    if not has_similarity:
        # turn to function
        print('Calculating similarity matrix, this can take a few minutes...')
        similarity_per_pack = list()

        for pack_idx in units_in_pack:
            if len(pack_idx) < 2: similarity_per_pack.append(None)
            else:
                this_spk = spk.copy().pick_cells(pack_idx)
                similarity = (
                    pln.spike_distance.compute_spike_coincidence_matrix(
                        this_spk)
                )
                similarity_per_pack.append(similarity)

        print('done.')

        # save to disk
        h5io.write_hdf5(op.join(spike_data_dir, fname), similarity_per_pack,
                        overwrite=True)
    else:
        similarity_per_pack = h5io.read_hdf5(op.join(spike_data_dir, fname))
        # add safety checks

    # drop = np.zeros(len(spk), dtype='bool')
    # drop_perc = drop.copy()
    df_list = list()

    reordered_names = ['n_spikes', 'snr', 'dns', 'isi', 'std']
    weights_sel = np.array(
        [weights[name] for name in reordered_names]
    )[None, :]

    no_drop_similarity = list()
    for pack_idx, (pack_units_idx, similarity) in enumerate(
            zip(units_in_pack, similarity_per_pack)):

        if similarity is None:
            continue

        spk_pack = spk.copy().pick_cells(pack_units_idx)
        suspicious_idx, clusters, counts = find_coincidence_clusters(
                similarity, threshold=coincidence_threshold)

        # detect likely REF clusters and select units
        # -------------------------------------------
        check_clst_idx = np.where(counts >= min_channels)[0]

        for cluster_idx in check_clst_idx:
            print(f'Processing pack {pack_idx}, cluster {cluster_idx}...')  # x
            in_cluster_cell_indices = suspicious_idx[clusters[cluster_idx]]
            cell_idx = pack_units_idx[in_cluster_cell_indices]
            channels = spk.cellinfo.loc[cell_idx, 'channel'].unique()
            if len(channels) < min_channels:
                continue

            group_idx += 1

            # get, rank and weight measures
            # -----------------------------
            measures_sel = np.stack(
                [measures[name][cell_idx]
                for name in ['n_spikes', 'snr', 'dns']
                ] +
                [(1 - measures_prc[name][cell_idx])
                for name in ['isi', 'std']],
                axis=1
            )
            measures_perc_sel = np.stack(
                [measures_prc[name][cell_idx]
                for name in ['n_spikes', 'snr', 'dns']
                ] +
                [(1 - measures_prc[name][cell_idx])
                for name in ['isi', 'std']]
            , axis=1)

            score = (measures_perc_sel * weights_sel).sum(axis=1)
            ranks = np.stack(
                [rankdata(measures_sel[:, ix])
                for ix in range(5)]
            , axis=1)
            score_ranks = (ranks * weights_sel).sum(axis=1)

            # save drop cells info according to percentiles and ranks
            # save_idx = cell_idx.copy()
            # save_idx_perc = np.delete(save_idx, score.argmax())
            # save_idx_ranks = np.delete(save_idx, score_ranks.argmax())
            # drop[save_idx_perc] = True
            # drop_perc[save_idx_ranks] = True

            # fill in the dataframe
            # ---------------------
            this_df = (
                spk.cellinfo.loc[cell_idx, ['channel', 'cluster']]
                .reset_index(drop=True)
            )
            this_df.loc[:, 'group'] = group_idx
            this_df.loc[:, 'subgroup'] = np.nan
            this_df.loc[:, 'pack'] = pack_idx + 1

            # add measures
            for ix, measure_name in enumerate(reordered_names):
                this_df.loc[:, measure_name] = measures_sel[:, ix]

            incl_idx = np.ix_(in_cluster_cell_indices, in_cluster_cell_indices)
            in_cluster_similarity = similarity[incl_idx]

            # iteratively drop according to score_ranks
            subgroup_id = 1
            drop_msk = np.zeros(in_cluster_similarity.shape[0], dtype='bool')
            left_similarity = in_cluster_similarity[~drop_msk][:, ~drop_msk]

            while (left_similarity > coincidence_threshold).any():
                this_df.loc[~drop_msk, 'subgroup'] = np.nan
                any_left = this_df.subgroup.isna().any()
                while any_left:
                    subgroup_idx = np.where(this_df.subgroup.isna())[0]
                    best_idx = score_ranks[subgroup_idx].argmax()
                    not_drop_idx = subgroup_idx[best_idx]

                    other_subgroup_idx = np.delete(subgroup_idx, best_idx)
                    other_related = in_cluster_similarity[
                        other_subgroup_idx, not_drop_idx]
                    drop = other_related > coincidence_threshold
                    other_related = in_cluster_similarity[
                        not_drop_idx, other_subgroup_idx]
                    drop = drop | (other_related > coincidence_threshold)
                    drop_idx = other_subgroup_idx[drop]
                    this_df.loc[drop_idx, 'subgroup'] = subgroup_id
                    this_df.loc[drop_idx, 'drop'] = True

                    this_df.loc[not_drop_idx, 'subgroup'] = subgroup_id
                    this_df.loc[not_drop_idx, 'drop'] = False

                    # select the best unit
                    any_left = this_df.subgroup.isna().any()
                    subgroup_id += 1

                # aggregate no_drop similarity
                drop_msk = this_df.loc[:, 'drop'].values.astype('bool')
                left_similarity = in_cluster_similarity[
                    ~drop_msk][:, ~drop_msk]
            no_drop_similarity.append(left_similarity)

            df_list.append(this_df)

            # produce and save plots
            # ----------------------
            if save_fig:
                fname_base = f'pack_{pack_idx:02g}_cluster_{cluster_idx:02g}'
                fname = fname_base + '_01_coincid.png'
                fig = plot_high_similarity_cluster(
                    spk_pack, similarity, clusters, suspicious_idx,
                    cluster_idx=cluster_idx
                )
                fig.savefig(op.join(fig_dir, fname), dpi=300)
                plt.close(fig)

                spk_sel = spk.copy().pick_cells(cell_idx)
                # fname = fname_base + '_02_score_percentiles.png'
                # fig = plot_scores(spk_sel, score)
                # fig.savefig(op.join(fig_dir, fname), dpi=300)
                # plt.close(fig)

                fname = fname_base + '_03_score_within_cluster_ranks.png'
                fig = plot_scores(spk_sel, score_ranks)
                fig.savefig(op.join(fig_dir, fname), dpi=300)
                plt.close(fig)

                # plot after if more than one subgroup
                if left_similarity.shape[0] > 1:
                    fname = fname_base + '_02_coincid_after.png'
                    these_clusters = [clst.copy() for clst in clusters]
                    this_clst_msk = these_clusters[cluster_idx]
                    this_clst_idx = np.where(this_clst_msk)[0]
                    this_clst_msk[this_clst_idx[drop_msk]] = False

                    fig = plot_high_similarity_cluster(
                        spk_pack, similarity, these_clusters, suspicious_idx,
                        cluster_idx=cluster_idx)
                    fig.savefig(op.join(fig_dir, fname), dpi=300)
                    plt.close(fig)

        if save_fig:
            ignored_clst_idx = np.where(counts < min_channels)[0]

            if len(ignored_clst_idx) > 0:
                print('Plotting ignored clusters...')
                ignored_dir = op.join(fig_dir, 'ignored_clusters')
                if not op.isdir(ignored_dir):
                    os.mkdir(ignored_dir)

            for cluster_idx in ignored_clst_idx:
                fname_base = f'pack_{pack_idx:02g}_cluster_{cluster_idx:02g}'
                fname = fname_base + '_01_coincid.png'
                fig = plot_high_similarity_cluster(
                    spk_pack, similarity, clusters, suspicious_idx,
                    cluster_idx=cluster_idx)
                fig.savefig(op.join(fig_dir, 'ignored_clusters', fname),
                            dpi=300)
                plt.close(fig)

    print('Saving dataframe to figures location...')
    df = pd.concat(df_list).reset_index(drop=True)
    df = df.loc[:, df_columns]
    df.loc[:, 'n_spikes'] = df.n_spikes.astype('int')
    df.to_csv(op.join(fig_dir, 'table.tsv'), sep='\t')

    print('All done.')

# # %% write _drop.txt file
# # for switchorder the file is read when using switchorder.read_spk() and unit
# # rejections are applied
# fname = data_dir + '_drop.txt'

# lines_to_write = list()
# channels_with_drops = np.unique(df.channel.values)

# for chan in channels_with_drops:
#     df_sel = df.query(f'channel == "{chan}"')
#     msk = df_sel.loc[:, 'drop'].values
#     if msk.any():
#         clusters = df_sel.loc[msk, 'cluster'].values.tolist()
#         line = f'{chan}, {clusters}\n'
#         lines_to_write.append(line)

# with open(fname, 'w') as file:
#     file.writelines(lines_to_write)

# # %% draw given cluster as graph
# import networkx as nx

# pack_idx = 3
# cluster_idx = 0
# plot_edge_threshold = 0.15 # coincidence_threshold

# pack_units_idx = units_in_pack[pack_idx]
# similarity = similarity_per_pack[pack_idx]

# spk_pack = spk.copy().pick_cells(pack_units_idx)
# suspicious_idx, clusters, counts = (
#     pln.spike_distance.find_coincidence_clusters(
#         similarity,
#         threshold=coincidence_threshold
#     )
# )

# idx = suspicious_idx[clusters[cluster_idx]]
# similarity_clst = similarity[idx][:, idx]

# # construct graph
# G = nx.DiGraph()

# n_units = similarity_clst.shape[0]
# for idx1 in range(n_units):
#     for idx2 in range(n_units):
#         fromto = similarity_clst[idx1, idx2]
#         if fromto > plot_edge_threshold:
#             G.add_edges_from([(idx1, idx2)], weight=fromto)


# # plot'
# pos = nx.spring_layout(G, k=1.5)
# weights = [G[u][v]['weight'] for u,v in G.edges()]
# nx.draw_networkx(G, pos, arrows=True, width=weights)

# # %%
# spk = swo.read_spk(subject, waveform=False, norm=norm)