import numpy as np
import pandas as pd

import pylabianca as pln
from pylabianca.utils import (create_random_spikes, _symmetric_window_samples,
                              _gauss_kernel_samples)
from pylabianca.selectivity import compute_selectivity_continuous


def test_selectivity_continuous():
    spk_epochs = pln.utils.create_random_spikes(
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
    assert np.in1d(selective_are, selective_should_be).mean() > 0.7

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

    # test also in time
    # ...


def test_cluster_based_selectivity():
    # create random spikes
    spk = pln.utils.create_random_spikes(
        n_cells=1, n_trials=50, n_spikes=(5, 15))

    # add metadata
    spk.metadata = pd.DataFrame(
        {'cond': np.concatenate([np.ones(25), np.ones(25) * 2])}
    )
    fr = spk.spike_density(fwhm=0.2)

    # add effect to one condition
    window, _ = _symmetric_window_samples(winlen=1.6, sfreq=500.)
    gauss = pln.utils._gauss_kernel_samples(window, gauss_sd=125)
    use_gauss = gauss[:np.where(window == 0)[0][0]]
    use_gauss /= use_gauss.max()
    n_gauss = len(use_gauss)

    cond_sel = np.where(fr.cond == 1)[0]
    effect = np.random.randint(low=2, high=10, size=len(cond_sel))
    effect = np.tile(effect[:, None], [1, n_gauss])[None, :, :] * use_gauss

    fr2 = fr.copy()
    fr2[:, :len(cond_sel), -n_gauss:] += effect

    # test that effect is present
    df2 = pln.selectivity.cluster_based_selectivity(
        fr2, 'cond', n_permutations=250)

    msk = df2.pval < 0.05
    assert msk.any()
    clst_idx = np.where(msk)[0][0]

    effect_window = [float(x) for x in df2.window[clst_idx].split(' - ')]
    assert np.diff(effect_window)[0] > 0.2

    pref = eval(df2.preferred[clst_idx])
    assert pref == [0]
    assert df2.n_preferred[clst_idx] == 1

    assert df2['FR_vs_baseline'][clst_idx] > 1.25
    assert df2['DoS'][clst_idx] > 0.2

    # test that effect was absent in original data:
    df = pln.selectivity.cluster_based_selectivity(
        fr, 'cond', n_permutations=250)
    assert (df.shape[0] == 0) or (df.pval > 0.05).all()
