# -*- coding: utf-8 -*-
"""
Scirpt that matches units identified in mixed alignment to corresponding
units in min/max alignment to identify merge candidates.

@author: mmagnuski
"""

# SETTINGS
# --------

# the directory with sorting results - that is after manual curation and
# export using updateSORTINGresults_mm matlab function located in
# psy_screenning-\helpers\sorting_utils)
results_dir = (r'G:\.shortcut-targets-by-id\1XlCWYRlHP0YDbmo3p1NGIC6lN9XZ8l1O\switchorder\derivatives\sorting\sub-W02\ses-main\sub-W02_ses-main_task-switchorder_run-01_sorter-osort_norm-False')

# the directory with raw sorting data - should contain min and max
# subdirectories with respective threshold subdirectories
sorting_dir = (r'C:\Users\mmagnuski\Dropbox\PROJ\Labianka\switchorder\sorting\sub-W02\osorted')
# the sorting threshold used (defines subdirectory in min and max directories)
sorting_threshold = '5'

# whether to save figures (use only for testing purposes, many figures will
#                          be created so the script may bet much slower)
save_figures = True

# where to save figures
fig_dir = (r'C:\Users\mmagnuski\Dropbox\PROJ\Labianka\sorting\merge_'
           'candidates_sub-W02')

# where to save results
save_spk_fname = 'all_channels.mat'
save_spk_dir = (r'C:\Users\mmagnuski\Dropbox\PROJ\Labianka\sorting'
                '\sub-W02_switchorder_after_cleanup')

# neg / pos peak ratio above which (or below its inverse) a waveform is
# classified as neg / pos polarity. This detected polarity is 
polarity_threshold = 1.333

# coincidence_threshold defines above what spike coincidence value
# a matching unit in max/min alignment will be accepted
# the default is 50% (0.5) - and seems to be a good choice
coincidence_threshold = 0.5


# %%
import os.path as op
from warnings import warn

import numpy as np
import matplotlib.pyplot as plt

import pylabianca as pln

# read spike times and waveforms
spk = pln.io.read_osort(results_dir, waveform=True)


def plot_waveform_polarity_choice(spk, idx, unit_type):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5), sharey=True)

    spk.plot_waveform(idx, ax=ax[0], labels=False)
    avg_waveform = spk.waveform[idx].mean(axis=0)
    ax[1].plot(avg_waveform)

    fig.suptitle(f'Classified as {unit_type[idx]}', fontsize=14)
    return fig


# TODO:
# for merge candidate plotting we need:
# mix_spk, match_spk, indices from first, one index from second
# and coincidence for each of the candidates
def plot_merge_candidates(spk, merge_info):

        merge_names = merge_info['cell_names']
        n_merge = len(merge_names)
        spk = spk.copy().pick_cells(merge_names)

        matching_alignment = merge_info['matching_alg']

        fig, ax = plt.subplots(
            ncols=n_merge + 1, figsize=(3 * (n_merge + 1), 5), sharey=True,
            gridspec_kw={'top': 0.8, 'right': 0.98, 'left': 0.06}
        )

        for idx in range(n_merge):
            spk.plot_waveform(idx, ax=ax[idx])

            this_n = spk.waveform[idx].shape[0]
            this_coincid = merge_info['coincidence'][idx]
            ax1_title = (f'cluster id: {spk.cellinfo.loc[idx, "cluster"]} '
                         f'\n(N = {this_n}) coinc: {this_coincid:.2f}')
            ax[idx].set_title(ax1_title, fontsize=12)

        # plot match
        spk_match = merge_info['matching_spk']
        spk_match.plot_waveform(0, ax=ax[-1])
        n_match = spk_match.waveform[0].shape[0]
        ax2_title = (f'Matching unit from {matching_alignment}\n'
                     f'(N = {n_match}) cluster id: {spk_match.cellinfo.loc[0, "cluster"]}')
        ax[-1].set_title(ax2_title, fontsize=12)

        title = (f'Channel {spk.cellinfo.loc[0, "channel"]}\n')
        fig.suptitle(title, fontsize=14)
        return fig


# find mixed alignment channels and test whether every channel has only one
# alignment type
msk_mixed = spk.cellinfo['alignment'] == 'mixed'
mixed_chans = spk.cellinfo.loc[msk_mixed, 'channel'].unique()

# make sure each channel contains only one alignment
chans = spk.cellinfo.channel.unique()
for chan in chans:
    info_sel = spk.cellinfo.query(f'channel == "{chan}"')

    alignments = info_sel.alignment.unique()
    n_alignments = len(alignments)

    if n_alignments > 1:
        msg = f'Channel {chan} contains {n_alignments} alignments:'
        for algn in alignments:
            n_units = (info_sel.alignment == algn).sum()
            msg += f'\n{algn}: {n_units} units; '
        warn(msg)


# main part of the script
# -----------------------
merges = list()

for chan in mixed_chans:
    simil = dict()
    spk_alg = dict()

    info_sel = spk.cellinfo.query(f'channel == "{chan}"')
    cell_idx = info_sel.index.to_numpy()
    cell_names = spk.cell_names[cell_idx]
    spk_sel = spk.copy().pick_cells(cell_idx)

    for check_type in ['pos', 'neg']:

        use_dir = 'max' if check_type == 'pos' else 'min'
        this_sorting_dir = op.join(sorting_dir, use_dir, sorting_threshold)
        spk2 = pln.io.read_osort(
            this_sorting_dir, channels=chan,
            format='standard', progress=False
        )

        # compare coincidence- spk_sel vs spk2
        simil[use_dir] = pln.spike_distance.compute_spike_coincidence_matrix(
            spk_sel, spk2=spk2, tol=0.002, progress=False)
        spk_alg[use_dir] = spk2.copy()

    all_combine = list()
    all_simil = list()
    all_alg = list()
    matches = list()

    for alg in ['min', 'max']:
        msk = simil[alg] > coincidence_threshold
        n_merged = msk.sum(axis=0)
        merge_candidates = n_merged > 1
        if merge_candidates.any():
            combine = msk[:, merge_candidates]
            all_combine.append(combine)
            all_simil.append(simil[alg][:, merge_candidates])
            all_alg.extend([alg] * sum(merge_candidates))
            matches.extend(np.where(merge_candidates)[0].tolist())

    n_comb = len(all_combine)
    if n_comb > 1:
        all_combine = np.concatenate(all_combine, axis=1)
        all_simil = np.concatenate(all_simil, axis=1)
    elif n_comb == 1:
        all_combine = all_combine[0]
        all_simil = all_simil[0]

    if n_comb > 0:
        # what to do if there are overlaps?
        has_overlaps = all_combine.sum(axis=1) > 1
        overlapping_idx = np.where(has_overlaps)[0]
        is_solved = np.zeros(all_combine.shape[1], dtype=bool)
        remove = np.zeros(all_combine.shape[1], dtype=bool)

        for ix in range(len(overlapping_idx)):
            col_mask = all_combine[overlapping_idx[ix], :]

            if not (is_solved[col_mask]).all():
                col_mask = col_mask & ~remove
                if col_mask.sum() < 2:
                    continue

                pair = all_combine[:, col_mask]
                assert pair.shape[1] == 2
                if (pair[:, 0] == pair[:, 1]).all():
                    # check if overlapping merges are identical
                    is_solved[col_mask] = True
                    rem_idx = np.where(col_mask)[0][-1]
                    remove[rem_idx] = True
                else:
                    # else - pick the one with highest average coincidence
                    pair_simil = all_simil[:, col_mask]
                    print('two different suggestions')
                    print(pair_simil)

                    pair_simil[~pair] = np.nan
                    avg_simil = np.nanmean(pair_simil, axis=0)
                    is_solved[col_mask] = True
                    rem_idx = np.where(col_mask)[0][avg_simil.argmin()]
                    remove[rem_idx] = True

        all_combine = all_combine[:, ~remove]
        all_simil = all_simil[:, ~remove]
        all_alg = np.array(all_alg)[~remove]
        matches = np.array(matches)[~remove]

        # turn combine matrices into cell merge indices
        for idx in range(all_combine.shape[1]):
            msk = all_combine[:, idx]
            merge_info = dict()
            merge_info['channel'] = chan
            merge_info['cell_names'] = cell_names[msk]
            merge_info['cell_idx'] = cell_idx[msk]
            merge_info['coincidence'] = all_simil[:, idx][msk]

            matching_alg = all_alg[idx]
            matching_idx = matches[idx]
            matching_spk = spk_alg[matching_alg].copy().pick_cells(matching_idx)
            merge_info['matching_alg'] = matching_alg
            merge_info['matching_spk'] = matching_spk

            merges.append(merge_info)

            # plot
            if save_figures:
                fname = f'merge_candidates_chan-{chan}_{idx:02g}.png'
                fig = plot_merge_candidates(spk, merge_info)
                plt.savefig(op.join(fig_dir, fname), dpi=100)
                plt.close(fig)
