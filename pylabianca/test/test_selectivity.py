import numpy as np
import pandas as pd

import pylabianca as pln
from pylabianca.utils import (create_random_spikes, _symmetric_window_samples,
                              _gauss_kernel_samples)
from pylabianca.selectivity import compute_selectivity_continuous

import pytest


def test_selectivity_continuous():
    spk_epochs = create_random_spikes(
        n_cells=20, n_trials=60, n_spikes=(10, 50))

    # add metadata
    emotions = np.array(['sad', 'happy', 'neutral'])
    emo = np.random.choice(emotions, size=60)
    spk_epochs.metadata = pd.DataFrame({'emo': emo})

    # compute firing rate in one window
    fr = spk_epochs.spike_rate(tmin=0.1, tmax=1.1, step=False)
    fr = np.sqrt(fr)

    results = compute_selectivity_continuous(
        fr, compare='emo', n_perm=5_000, n_jobs=4
    )

    # we shouldn't have too many selective cells
    # (because we don't simulate the effect)
    selective = (results['stat'] > results['thresh']).values
    assert selective.mean() <= 0.25

    # make sure the thresholds are correct
    percentiles = np.percentile(results['dist'], 95, axis=0)
    assert (results['thresh'] == percentiles).all().item()

    # simulate the effect in some cells and test again
    selective_should_be = [0, 2, 3, 17]
    for cell_idx in selective_should_be:
        pick_cond = np.random.choice([0, 1, 2])
        cond_mask = emo == emotions[pick_cond]
        cond_idx = np.where(cond_mask)[0]
        n_tri = len(cond_idx)
        effect = np.random.rand(n_tri) * 2 + 0.5
        fr[cell_idx, cond_idx] += effect

    results = compute_selectivity_continuous(
        fr, compare='emo', n_perm=10_000, n_jobs=4)

    # at least 3 out of 4 should be in the selective list
    selective_are = np.where((results['stat'] > results['thresh']).values)[0]
    assert np.in1d(selective_should_be, selective_are).mean() > 0.7

    # add cellinfo and make sure it is retained in results
    n_cells = spk_epochs.n_units()
    cell_types = ['A', 'B', 'C']
    regions = ['AMY', 'HIP']

    cellinfo = pd.DataFrame(
        {'cell_type': np.random.choice(cell_types, size=n_cells),
        'region': np.random.choice(regions, size=n_cells)})
    spk_epochs.cellinfo = cellinfo

    fr = spk_epochs.spike_rate(tmin=0.1, tmax=1.1, step=False)
    results = compute_selectivity_continuous(
        fr, compare='emo', n_perm=10, n_jobs=1)

    for key in results.keys():
        assert all([coo in results[key].coords for coo in cellinfo.columns])

    # test also in time with two conditions
    spk = create_random_spikes(n_cells=1, n_trials=50, n_spikes=(5, 15))

    # add metadata
    spk.metadata = pd.DataFrame(
        {'cond': np.concatenate([np.ones(25), np.ones(25) * 2])}
    )
    fr = spk.spike_density(fwhm=0.2, sfreq=100.)

    sel = pln.selectivity.compute_selectivity_continuous(
        fr, compare='cond', n_jobs=2)

    # we shouldn't have too many selective (abs(tval) > 2) cells
    assert np.abs(np.abs(sel['thresh']).mean() - 2).item() < 0.12
    assert sel['stat'].shape[-1] == fr.shape[-1]
    assert sel['stat'].shape[0] == fr.shape[0]


def test_cluster_based_selectivity():
    # create random spikes
    spk = create_random_spikes(n_cells=1, n_trials=50, n_spikes=(5, 15))

    # add metadata and cellinfo
    spk.metadata = pd.DataFrame(
        {'cond': np.concatenate([np.ones(25), np.ones(25) * 2])}
    )
    spk.cellinfo = pd.DataFrame(
        {'cell_type': ['A'], 'region': ['AMY'], 'quality': [0.78]}
    )
    fr_orig = spk.spike_density(fwhm=0.2)

    # add effect to one condition
    window, _ = _symmetric_window_samples(winlen=1., sfreq=500.)
    gauss = _gauss_kernel_samples(window, gauss_sd=100)
    gauss /= gauss.max()
    n_gauss = len(gauss)

    cond_sel = np.where(fr_orig.cond == 1)[0]
    effect = np.random.randint(low=4, high=10, size=len(cond_sel))
    effect = np.tile(effect[:, None], [1, n_gauss])[None, :, :] * gauss

    fr_effect = fr_orig.copy()
    zero_idx = pln.utils.find_index(fr_orig.time.values, 0)[0]
    fr_effect[:, cond_sel, zero_idx:zero_idx + n_gauss:] += effect

    # test that effect is present
    df = pln.selectivity.cluster_based_selectivity(
        fr_effect, 'cond', n_permutations=250, calculate_pev=True,
        calculate_peak_pev=True, copy_cellinfo=['region', 'quality'])

    msk = df.pval < 0.05
    assert msk.any()

    # make sure relevant cellinfo columns were copied
    assert 'region' in df.columns
    assert 'quality' in df.columns
    assert np.unique(df.region)[0] == 'AMY'
    assert np.isclose(df.quality[0], 0.78)

    # "int_toi" and "selective" columns are added by assess_selectivity
    # function, so they should not be present in the original data
    assert 'in_toi' not in df.columns
    assert 'selective' not in df.columns

    df = pln.selectivity.assess_selectivity(
        df, min_cluster_p=0.01, window_of_interest=(0.25, 0.75),
        min_time_in_window=0.15, min_depth_of_selectivity=0.15,
        min_pev=0.1, min_peak_pev=0.12, min_FR_vs_baseline=1.25,
        min_FR_preferred=4.5
    )
    assert 'in_toi' in df.columns
    assert 'selective' in df.columns
    assert df.selective.sum() == 1
    clst_idx = df.pval.argmin()
    assert np.where(df.selective)[0][0] == clst_idx

    pref = eval(df.preferred[clst_idx])
    assert pref == [0]
    assert df.n_preferred[clst_idx] == 1

    # test that effect was absent in original data:
    df_no_effect = pln.selectivity.cluster_based_selectivity(
        fr_orig, 'cond', n_permutations=250, calculate_pev=True,
        calculate_peak_pev=True)
    df_no_effect = pln.selectivity.assess_selectivity(
        df_no_effect, min_cluster_p=0.01, window_of_interest=(0.25, 0.75),
        min_time_in_window=0.15, min_depth_of_selectivity=0.15,
        min_pev=0.1, min_peak_pev=0.12, min_FR_vs_baseline=1.25,
        min_FR_preferred=4.5
    )
    assert (
        (df_no_effect.shape[0] == 0)
        or (df_no_effect.pval > 0.05).all()
        or (df_no_effect.selective.sum() == 0)
    )


def test_threshold_selectivity():
    import xarray as xr

    # create data
    n_cells, n_times = 10, 120
    times = np.linspace(-0.25, 1., num=n_times)
    sel_data = np.random.randn(n_cells, n_times)
    sel = xr.DataArray(sel_data, dims=['cell', 'time'], coords={'time': times})

    # test with a scalar threshold
    thresh1 = 1.85
    sel_bool = pln.selectivity.threshold_selectivity(sel, thresh1)
    assert (sel_bool.data == (np.abs(sel.data) > thresh1)).all()

    # create a threshold array (pos and neg tail)
    thresh_pos = 1 + np.random.rand(n_cells, n_times)
    thresh_neg = (1 + np.random.rand(n_cells, n_times)) * -1
    thresh = xr.DataArray(
        np.stack([thresh_pos, thresh_neg], axis=0),
        dims=['tail', 'cell', 'time'],
        coords={'tail': ['pos', 'neg'], 'time': times})

    # use both tails
    expected_bool = (sel.data > thresh_pos) | (sel.data < thresh_neg)
    sel_bool = pln.selectivity.threshold_selectivity(sel, thresh)
    assert (sel_bool.data == expected_bool).all()

    # only pos tail:
    expected_bool = sel.data > thresh_pos
    sel_bool = pln.selectivity.threshold_selectivity(sel, thresh.sel(tail='pos'))
    assert (sel_bool.data == expected_bool).all()

    # only neg tail:
    expected_bool = sel.data < thresh_neg
    sel_bool = pln.selectivity.threshold_selectivity(sel, thresh.sel(tail='neg'))
    assert (sel_bool.data == expected_bool).all()

    # raises error when trying to use a vector without labels (not an xarray):
    with pytest.raises(ValueError, match='Threshold must be '):
        pln.selectivity.threshold_selectivity(sel, thresh_pos)
