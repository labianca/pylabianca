# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:21:13 2022

@author: mmagnuski
"""
import os
import os.path as op

import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy.stats import rankdata
from tqdm import tqdm

import pylabianca as pln


# TODO:
# - [x] compute coicidence only for 8-channel packs
#       (but detecting 8-packs on exported spikes may be sometimes difficult
#        - when there are no spikes for first channel, there will be no file
#          for it, which could lead to offsets and bad results)

# SETTINGS
# --------

# first channel
# to calculate coincidence only in 8-channel packs, you have to specify
# the first channel - in case you did not select any units from the first
# channel
# if you leave it as None, the first channel will be estimated from
# selected units, which may be incorrect
first_channel = 64

# the directory with sorting results - that is after manual curation and
# export using updateSORTINGresults_mm matlab function located in
# psy_screenning-\helpers\sorting_utils)
data_dir = (r'D:\Dropbox\PROJ\Labianka\sorting\Kasia_debug\sub-U03_ses-main_task-retrocue_run-01_ieeg')

# whether to plot figures and where to save them
save_fig = True
save_fig_dir = (r'D:\Dropbox\PROJ\Labianka\sorting\Kasia_debug\figures')

# data format - 'standard' or 'mm' (depends on how you exported the curated units)
data_format = 'standard'

# minimum coincidence threshold for coincidence cluster formation
coincidence_threshold = 0.3

# minimum number of different-channel units in a coincidence cluster to
# classify it as a reference cluster
min_ref_channels = 4

# weights
# -------
# weights used in calculating the final score (unit with best score is chosen)
# isi - % inter spike intervals < 3 ms (lower is better)
# fr  - firing rate
# snr - signal to noise ratio (mean / std) at alignment point
# std - standard deviation calculated separately for each waveform sample
#       and then averaged (lower is better)
# dns - average "perceptual" waveform density (for each time sample 20 best
#       pixels from waveform density are chosen and averaged; these values
#       are then averaged across the whole waveform; the values are divided
#       by maximum density to de-bias against waveforms with little spikes
#       and thus make this value closer to the density we see in the figures)
weights = {'isi': 0.15, 'fr': 0.35, 'snr': 0.0, 'std': 0.0, 'dns': 0.5}

# alignment sample index - 94 should be true for all our osort files
algn_smpl = 94


# %% MAIN SCRIPT

# read the file
print('Reading files - including all waveforms...')
spk = pln.io.read_osort(data_dir, waveform=True, format=data_format)
print('done.')

# remove units with no spikes
n_spikes = spk.n_spikes()
no_spikes = n_spikes == 0
if (no_spikes).any():
    spk.pick_cells(~no_spikes)


def turn_to_percentiles(arr):
    percentiles = np.vectorize(
        lambda x: scipy.stats.percentileofscore(arr, x))(arr)
    return percentiles / 100.


def plot_scores(spk_sel, score):
    from string import ascii_letters

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

    ax['a'].set_ylabel('Weighted score', fontsize=14)
    return ax['a'].figure


# check channel packs
# -------------------
channels = spk.cellinfo.channel.unique()
channel_number = np.sort([int(ch[1:]) for ch in channels])

if first_channel is None:
    print('Estimating first channel from selected units.')
    first_channel = channel_number[0]
    print('Lowest channel number:', first_channel)

n_packs = int(np.ceil((channel_number[-1] - first_channel) / 8))
first_pack_channel = first_channel + np.arange(0, n_packs) * 8

# find which units belong to which pack
unit_channel = spk.cellinfo.channel.str.slice(1).astype('int')

units_in_pack = [np.where(np.in1d(unit_channel, frst + np.arange(8)))[0]
                 for frst in first_pack_channel]


# calculate measures
# ------------------
# isi, fr, snr, std, dns

print('Calculating FR, SNR, ISI, STD and DNS...')
fr = [len(tms) for tms in spk.timestamps]
fr_prc = turn_to_percentiles(fr)

n_spikes = len(spk)
spk_epo = spk.to_epochs()

snr = np.zeros(n_spikes, dtype='float')
isi, std, dns = snr.copy(), snr.copy(), snr.copy()

for ix in tqdm(range(n_spikes)):
    # SNR
    waveform_ampl = np.mean(spk.waveform[ix][:, algn_smpl], axis=0)
    waveform_std = np.std(spk.waveform[ix][:, algn_smpl], axis=0)
    snr[ix] = np.abs(waveform_ampl) / waveform_std

    # ISI
    isi_val = np.diff(spk_epo.time[ix])
    prop_below_3ms = (isi_val < 0.003).mean()
    isi[ix] = prop_below_3ms

    # STD
    avg_std = np.std(spk.waveform[ix], axis=0).mean()
    std[ix] = avg_std

    # DNS
    dns[ix] = pln.viz.calculate_perceptual_waveform_density(spk, ix)

snr_prc = turn_to_percentiles(snr)
isi_prc = turn_to_percentiles(isi)
std_prc = turn_to_percentiles(std)
dns_prc = turn_to_percentiles(dns)

print('Done.')


# compute coincidence
# -------------------
# or read from disk if already computed
import h5io

fname = 'coincidence_per_pack.hdf5'
files = os.listdir(data_dir)
has_simil = fname in files

if not has_simil:
    print('Calculating similarity matrix, this can take a few minutes...')
    similarity_per_pack = list()

    for pack_idx in units_in_pack:
        if len(pack_idx) < 2:
            similarity_per_pack.append(None)
        else:
            this_spk = spk.copy().pick_cells(pack_idx)
            similarity = pln.spike_distance.compute_spike_coincidence_matrix(
                this_spk)
            similarity_per_pack.append(similarity)

    print('done.')

    # save to disk
    h5io.write_hdf5(op.join(data_dir, fname), similarity_per_pack,
                    overwrite=True)
else:
    similarity_per_pack = h5io.read_hdf5(op.join(data_dir, fname))


drop = np.zeros(len(spk), dtype='bool')
drop_perc = drop.copy()
for pack_idx, (pack_units_idx, simil) in enumerate(
        zip(units_in_pack, similarity_per_pack)):

    if simil is None:
        continue

    spk_pack = spk.copy().pick_cells(pack_units_idx)
    suspicious_idx, clusters, counts = (
        pln.spike_distance.find_coincidence_clusters(
            simil,
            threshold=coincidence_threshold
        )
    )

    # detect likely REF clusters and select units
    # -------------------------------------------
    check_clst_idx = np.where(counts >= min_ref_channels)[0]

    for cluster_idx in check_clst_idx:
        print(f'Processing cluster {cluster_idx}...')
        cell_idx = pack_units_idx[suspicious_idx[clusters[cluster_idx]]]
        channels = spk.cellinfo.loc[cell_idx, 'channel'].unique()
        if len(channels) < min_ref_channels:
            continue

        # get percentile scores
        score = (snr_prc[cell_idx] * weights['snr']
                 + fr_prc[cell_idx] * weights['fr']
                 + (1 - isi_prc[cell_idx]) * weights['isi']
                 + (1 - std_prc[cell_idx]) * weights['std']
                 + dns_prc[cell_idx] * weights['dns'])

        # scores from within-cluster ranks
        fr_ranks = rankdata(np.array(fr)[cell_idx])
        snr_ranks = rankdata(snr[cell_idx])
        isi_ranks = rankdata(1 - isi[cell_idx])
        std_ranks = rankdata(1 - std[cell_idx])
        dns_ranks = rankdata(dns[cell_idx])

        score_ranks = (snr_ranks * weights['snr'] +
                       fr_ranks  * weights['fr'] +
                       isi_ranks * weights['isi'] +
                       std_ranks * weights['std'] +
                       dns_ranks * weights['dns'])

        # save drop cells info according to percentiles and ranks
        save_idx = cell_idx.copy()
        save_idx_perc = np.delete(save_idx, score.argmax())
        save_idx_ranks = np.delete(save_idx, score_ranks.argmax())
        drop[save_idx_perc] = True
        drop_perc[save_idx_ranks] = True

        # produce and save plots
        # ----------------------
        if save_fig:
            fname = f'pack_{pack_idx:02g}_cluster_{cluster_idx:02g}_01_coincid.png'
            fig = pln.spike_distance.plot_high_similarity_cluster(
                spk_pack, simil, clusters, suspicious_idx,
                cluster_idx=cluster_idx)
            fig.savefig(op.join(save_fig_dir, fname), dpi=300)
            plt.close(fig)

            spk_sel = spk.copy().pick_cells(cell_idx)
            fname = f'pack_{pack_idx:02g}_cluster_{cluster_idx:02g}_02_score_percentiles.png'
            fig = plot_scores(spk_sel, score)
            fig.savefig(op.join(save_fig_dir, fname), dpi=300)
            plt.close(fig)

            fname = f'pack_{pack_idx:02g}_cluster_{cluster_idx:02g}_03_score_within_cluster_ranks.png'
            fig = plot_scores(spk_sel, score_ranks)
            fig.savefig(op.join(save_fig_dir, fname), dpi=300)
            plt.close(fig)

    if save_fig:
        print('Plotting ignored clusters...')
        ignored_clst_idx = np.where(counts < min_ref_channels)[0]

        if len(ignored_clst_idx) > 0:
            ignored_dir = op.join(save_fig_dir, 'ignored_clusters')
            if not op.isdir(ignored_dir):
                os.mkdir(ignored_dir)

        for cluster_idx in ignored_clst_idx:
            fname = f'pack_{pack_idx:02g}_cluster_{cluster_idx:02g}_01_coincid.png'
            fig = pln.spike_distance.plot_high_similarity_cluster(
                spk_pack, simil, clusters, suspicious_idx,
                cluster_idx=cluster_idx)
            fig.savefig(op.join(save_fig_dir, 'ignored_clusters', fname), dpi=300)
            plt.close(fig)

    print('All done.')

# %%
# import h5io

# drop_save_dir = r'G:\.shortcut-targets-by-id\1XlCWYRlHP0YDbmo3p1NGIC6lN9XZ8l1O\switchorder\derivatives\sorting\sub-W02\ses-main'
# drop_fname = r'sub-W02_ses-main_task-switchorder_run-01_sorter-osort_norm-False_drop.hdf5'
# h5io.write_hdf5(op.join(drop_save_dir, drop_fname), drop)