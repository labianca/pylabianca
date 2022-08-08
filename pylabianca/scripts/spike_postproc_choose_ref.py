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

import pylabianca as pln


# TODO:
# - [ ] compute coicidence only for 8-channel packs
#       (but detecting 8-packs on exported spikes may be sometimes difficult
#        - when there are no spikes for first channel, there will be no file
#          for it, which could lead to offsets and bad results)
# - [ ] accept not mm format

# SETTINGS
# --------

# the directory with sorting results - that is after manual curation and
# export using updateSORTINGresults_mm matlab function located in
# psy_screenning-\helpers\sorting_utils)
data_dir = (r'G:\.shortcut-targets-by-id\1XlCWYRlHP0YDbmo3p1NGIC6lN9XZ8'
            r'l1O\switchorder\derivatives\sorting\sub-W02\ses-main\sub-'
            r'W02_ses-main_task-switchorder_run-01_sorter-osort_norm-False')
save_fig_dir = (r'C:\Users\mmagnuski\Dropbox\PROJ\Labianka\sorting\ref_test'
                r's\sub-W02_test01')



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

# alignment sample index - should true for all our osort files
algn_smpl = 94


# %% MAIN SCRIPT

# read the file
print('Reading files - including all waveforms...')
spk = pln.io.read_osort(data_dir, waveform=True, format='standard')
print('done.')


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


# calculate measures
# ------------------
# isi, fr, snr, std, dns

print('Calculating FR, SNR, ISI, STD and DNS...')
fr = [len(tms) for tms in spk.timestamps]
fr_prc = turn_to_percentiles(fr)

n_spikes = len(spk)
spk_epo = spk.to_epochs()

snr = np.zeros(n_spikes, dtype='float')
isi = np.zeros(n_spikes, dtype='float')
std = np.zeros(n_spikes, dtype='float')
dns = np.zeros(n_spikes, dtype='float')

for ix in range(n_spikes):
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
    hist, xbins, ybins, time_edges = (
        pln.viz._calculate_waveform_density_image(
            spk, ix, False, 100)
    )
    hist_sort = np.sort(hist)[:, ::-1]
    vals = (hist_sort[:, :20] / hist_sort.max()).mean(axis=-1)
    dns[ix] = vals.mean()

snr_prc = turn_to_percentiles(snr)
isi_prc = turn_to_percentiles(isi)
std_prc = turn_to_percentiles(std)
dns_prc = turn_to_percentiles(dns)

print('Done.')


# compute coincidence
# -------------------
# or read from disk if already computed
import h5io

fname = 'coincidence.hdf5'
files = os.listdir(data_dir)
has_simil = fname in files

if not has_simil:
    print('Calculating similarity matrix, this can take a few minutes...')
    similarity = pln.spike_distance.compute_spike_coincidence_matrix(spk)
    print('done.')

    # save to disk
    h5io.write_hdf5(op.join(data_dir, fname), similarity, overwrite=True)
else:
    similarity = h5io.read_hdf5(op.join(data_dir, fname))

suspicious_idx, clusters, counts = (
    pln.spike_distance.find_coincidence_clusters(
        similarity,
        threshold=coincidence_threshold
    )
)

# detect likely REF clusters and select units
# -------------------------------------------
check_clst_idx = np.where(counts >= min_ref_channels)[0]

for cluster_idx in check_clst_idx:
    print(f'Processing cluster {cluster_idx}...')
    cell_idx = suspicious_idx[clusters[cluster_idx]]
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

    # produce and save plots
    # ----------------------
    fname = f'cluster_{cluster_idx:02g}_01_coincid.png'
    fig = pln.spike_distance.plot_high_similarity_cluster(
        spk, similarity, clusters, suspicious_idx, cluster_idx=cluster_idx)
    fig.savefig(op.join(save_fig_dir, fname), dpi=300)
    plt.close(fig)

    spk_sel = spk.copy().pick_cells(cell_idx)

    fname = f'cluster_{cluster_idx:02g}_02_score_percentiles.png'
    fig = plot_scores(spk_sel, score)
    fig.savefig(op.join(save_fig_dir, fname), dpi=300)
    plt.close(fig)

    fname = f'cluster_{cluster_idx:02g}_03_score_within_cluster_ranks.png'
    fig = plot_scores(spk_sel, score_ranks)
    fig.savefig(op.join(save_fig_dir, fname), dpi=300)
    plt.close(fig)

print('Plotting ignored clusters...')
ignored_clst_idx = np.where(counts < min_ref_channels)[0]

if len(ignored_clst_idx) > 0:
    os.mkdir(op.join(save_fig_dir, 'ignored_clusters'))

for cluster_idx in ignored_clst_idx:
    fname = f'cluster_{cluster_idx:02g}_01_coincid.png'
    fig = pln.spike_distance.plot_high_similarity_cluster(
        spk, similarity, clusters, suspicious_idx, cluster_idx=cluster_idx)
    fig.savefig(op.join(save_fig_dir, 'ignored_clusters', fname), dpi=300)
    plt.close(fig)

print('All done.')
