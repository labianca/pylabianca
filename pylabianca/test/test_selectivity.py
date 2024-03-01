import numpy as np
import pandas as pd

import pylabianca as pln
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

    # LATER: add cellinfo and make sure it is retained in results
    # ...

    # test also in time
    # ...