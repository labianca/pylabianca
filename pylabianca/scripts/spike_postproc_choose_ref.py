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
from tqdm import tqdm
import pylabianca as pln


algn_smpl = 94
min_ref_channels = 4
coincidence_threshold = 0.3

data_dir = (r'G:\.shortcut-targets-by-id\1XlCWYRlHP0YDbmo3p1NGIC6'
            r'lN9XZ8l1O\switchorder\derivatives\sorting\sub-U06\ses'
            r'-main\sub-U06_ses-main_task-switchorder_run-01_sorter'
             '-osort_norm-False')
save_fig_dir = r'C:\Users\mmagnuski\Dropbox\PROJ\Labianka\sorting\test_figs'

# %%
# read the file
print('Reading files - including all waveforms...')
spk = pln.io.read_osort(data_dir, waveform=True)
print('done.')

# calculate for every neuron:

# 0. FR (n spikes)
# 1. SNR at alignment
# 2. ISI < threshold
# 3. similarity matrix


def turn_to_percentiles(arr):
    percentiles = np.vectorize(
        lambda x: scipy.stats.percentileofscore(arr, x))(arr)
    return percentiles / 100.


def plot_scores(spk_sel, score):
    n_uni = len(spk_sel)
    letters = 'bcdefghijklmn'[:n_uni]
    ax = plt.figure(constrained_layout=True, figsize=(12, 6)).subplot_mosaic(
        'a' * n_uni + '\n' + letters,
         gridspec_kw={'height_ratios': [3, 1]}
    )

    # plot scores
    ax['a'].plot(score, marker='o')

    # plot best score in red
    best_idx = score.argmax()
    ax['a'].plot(best_idx, score[best_idx], marker='o', markersize=18,
                 markerfacecolor='r')

    # add waveforms at the bottom
    letters = list(letters)
    for ix in range(len(spk_sel)):
        spk_sel.plot_waveform(ix, ax=ax[letters[ix]], labels=False)
        ax[letters[ix]].set_title(f'{len(spk_sel.timestamps[ix])} spikes')
    
    ax['a'].set_ylabel('Weighted score', fontsize=14)
    return ax['a'].figure


print('Calculating FR, SNR and ISI...')
fr = [len(tms) for tms in spk.timestamps]
fr_prc = turn_to_percentiles(fr)

n_spikes = len(spk)
spk_epo = spk.to_epochs()

snr = np.zeros(n_spikes, dtype='float')
isi = np.zeros(n_spikes, dtype='float')

for ix in range(n_spikes):
    # SNR
    waveform_ampl = np.mean(spk.waveform[ix][:, algn_smpl], axis=0)
    waveform_std = np.std(spk.waveform[ix][:, algn_smpl], axis=0)
    snr[ix] = np.abs(waveform_ampl) / waveform_std

    # ISI
    isi_val = np.diff(spk_epo.time[ix])
    prop_below_3ms = (isi_val < 0.003).mean()
    isi[ix] = prop_below_3ms

snr_prc = turn_to_percentiles(snr)
isi_prc = turn_to_percentiles(isi)

print('Done.')

# compute coincidence
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

# detect likely REF clusters and process
check_clst_idx = np.where(counts >= min_ref_channels)[0]

for cluster_idx in check_clst_idx:
    print('Processing cluster {cluster_idx}...')
    cell_idx = suspicious_idx[clusters[cluster_idx]]
    channels = spk.cellinfo.loc[cell_idx, 'channel'].unique()
    if len(channels) < min_ref_channels:
        continue
    
    # get percentile scores
    score = (snr_prc[cell_idx] * 0.6
             + fr_prc[cell_idx] * 0.2
             + (1 - isi_prc[cell_idx]) * 0.2)

    # scores from within-cluster ranks
    fr_ranks = scipy.stats.rankdata(np.array(fr)[cell_idx])
    snr_ranks = scipy.stats.rankdata(np.array(snr)[cell_idx])
    isi_ranks = scipy.stats.rankdata(1 - np.array(isi)[cell_idx])
    
    score_ranks = (snr_ranks * 0.6 +
                   fr_ranks  * 0.2 +
                   isi_ranks * 0.2)
    
    # save_plots
    fname = f'cluster_{cluster_idx:02g}_01_coincid.png'
    fig = pln.spike_distance.plot_high_similarity_cluster(
        spk, similarity, clusters, suspicious_idx, cluster_idx=cluster_idx)
    fig.savefig(op.join(save_fig_dir, fname), dpi=300)
    plt.close(fig)

    # save fig scores
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
# %%

