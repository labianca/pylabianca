# test ZETA test
import time
import importlib.util
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.io import loadmat

import pylabianca as pln
from pylabianca.testing import random_spikes


@pytest.mark.skipif(
    importlib.util.find_spec("zetapy") is None,
    reason="requires optional dependency: zetapy",
)
def test_against_zetapy():
    from zetapy import zetatest2

    numba_available = pln.utils.has_numba()
    fpath = pln.utils.get_zeta_example_data()
    data = loadmat(fpath, squeeze_me=True)

    times1 = data['sNeuron']['SpikeTimes'][0]
    times2 = data['sNeuron']['SpikeTimes'][1]

    stim_info = data['sStim']
    event_on = stim_info['StimOnTime'].item()
    event_off = stim_info['StimOffTime'].item()
    stim_ori = stim_info['Orientation'].item()

    n_events = event_on.shape[0]
    events = np.stack([event_on, event_off, np.ones(n_events)], axis=1)

    # run the two-sample ZETA-test
    tmax = 1.
    n_permutations = 500

    cond1 = stim_ori == 0
    cond2 = stim_ori == 90

    time_start = time.time()
    zeta_pval, zeta_info = zetatest2(
        times1, events[cond1, 0],
        times1, events[cond2, 0],
        tmax, n_permutations,
        boolPlot=False
    )
    time_taken_zetapy = time.time() - time_start

    # turn to pylabianca object
    spk = pln.Spikes([times1, times2], sfreq=1.)

    # create epochs and add metadata
    spk_epochs = spk.epoch(events, event_id=1, tmin=-0.25, tmax=1.)
    spk_epochs.metadata = pd.DataFrame({'orientation': stim_ori})

    # select epochs
    spk_epochs_sel = spk_epochs['orientation in [0, 90]']

    time_start = time.time()
    z_val, p_val, dist = pln.selectivity.zeta_test(
        spk_epochs_sel, compare='orientation', n_permutations=n_permutations,
        tmax=1., return_dist=True, backend='numpy', picks=0)
    time_taken_pln = time.time() - time_start

    # pylabianca should be faster than zetapy
    assert (time_taken_pln * 4) < time_taken_zetapy

    # cumulative traces should be similar
    np.testing.assert_allclose(
        zeta_info['vecRealDiff'], dist['trace'][0]
    )

    # similar p value (within 20 permutations difference)
    assert np.abs(p_val[0] - zeta_pval) < (20 / n_permutations)

    # similar z value
    assert np.abs(zeta_info['dblZETA'] - z_val[0]) < 0.35

    # if has_numba
    if numba_available:
        z_val_numba, p_val_numba, dist_numba = pln.selectivity.zeta_test(
            spk_epochs_sel, compare='orientation', n_permutations=500,
            tmax=1., return_dist=True, backend='numba', picks=0)

        # make sure traces and pvals are similar
        np.testing.assert_allclose(
            dist['trace'], dist_numba['trace']
        )
        assert np.abs(p_val[0] - p_val_numba[0]) < (20 / n_permutations)
        assert np.abs(z_val[0] - z_val_numba[0]) < 0.25

    # make sure permutation vectors are accurate
    z_val, p_val, dist1 = pln.selectivity.zeta_test(
        spk_epochs, compare='orientation', n_permutations=n_permutations,
        tmax=1., return_dist=True, backend='numpy', picks=0)

    perm_idx = np.random.randint(0, high=n_permutations)
    spk_epochs_cp = spk_epochs.copy()
    spk_epochs_cp.metadata.loc[:, 'orientation'] = dist1['perm_vec'][perm_idx, :]

    # this also tests that n_permutations=0 works
    _, _, dist2 = pln.selectivity.zeta_test(
        spk_epochs_cp, compare='orientation', n_permutations=0,
        tmax=1., return_dist=True, backend='numpy', picks=0)

    assert (dist1['perm_trace'][0][perm_idx] == dist2['trace'][0]).all()

    # test N string conditions
    stim_ori_str = np.array([str(ori) for ori in stim_ori], dtype='object')
    spk_epochs.metadata = pd.DataFrame({'orientation': stim_ori_str})

    backend = 'numba' if numba_available else 'numpy'
    z_val_n1, p_val_n1 = pln.selectivity.zeta_test(
        spk_epochs, compare='orientation', n_permutations=100,
        backend=backend, picks=0, significance='empirical')
    z_val_n2, p_val_n2 = pln.selectivity.zeta_test(
        spk_epochs, compare='orientation', n_permutations=100,
        backend=backend, picks=0, significance='both')

    assert isinstance(p_val_n2, dict)
    assert 'empirical' in p_val_n2 and 'gumbel' in p_val_n2
    assert (np.abs(p_val_n1 - p_val_n2['empirical']) < (20 / 100)).all()


# TODO: later might be a good idea to add metadata / cellinfo creation to
#       the testing function
def _create_random_spikes_with_metadata_and_cellinfo():
    np.random.seed(0)
    n_trials = 12
    h_tri = n_trials // 2

    metadata = pd.DataFrame(
        {'image': np.array(['A'] * h_tri + ['B'] * h_tri)}
    )
    cellinfo = pd.DataFrame({
        'region': ['hipp', 'amyg'], 'quality': ['good', 'mua']
    })

    spk = random_spikes(
        n_cells=2, n_trials=n_trials, n_spikes=(8, 13),
        metadata=metadata, cellinfo=cellinfo
    )
    return spk


def test_zeta_return_type_xarray():
    spk = _create_random_spikes_with_metadata_and_cellinfo()

    np.random.seed(0)
    z_np, p_np, dist_np = pln.selectivity.zeta_test(
        spk, compare='image', n_permutations=50, return_dist=True,
        return_type='numpy', backend='numpy'
    )

    np.random.seed(0)
    zeta_xr = pln.selectivity.zeta_test(
        spk, compare='image', n_permutations=50, return_dist=True,
        return_type='xarray', backend='numpy'
    )

    assert isinstance(zeta_xr, xr.Dataset)
    assert set(zeta_xr.data_vars) >= {'stat', 'dist', 'pval'}

    np.testing.assert_allclose(z_np, zeta_xr['z'].values)
    np.testing.assert_allclose(p_np, zeta_xr['pval'].values)
    np.testing.assert_allclose(dist_np['max'], zeta_xr['stat'].values)
    np.testing.assert_allclose(dist_np['perm_max'], zeta_xr['dist'].values)
    assert np.array_equal(dist_np['perm_vec'], zeta_xr['perm_vec'].values)

    assert 'region' in zeta_xr.coords
    assert 'quality' in zeta_xr.coords


def test_zeta_return_type_xarray_significance_both():
    spk = _create_random_spikes_with_metadata_and_cellinfo()

    with pytest.raises(ValueError, match='not supported'):
        pln.selectivity.zeta_test(
            spk, compare='image', n_permutations=20, return_dist=True,
            return_type='xarray', backend='numpy', significance='both',
            permute_independently=True
        )


def test_zeta_return_type_xarray_no_dist():
    spk = _create_random_spikes_with_metadata_and_cellinfo()

    out = pln.selectivity.zeta_test(
        spk, compare='image', n_permutations=20, return_dist=False,
        return_type='xarray', backend='numpy'
    )

    assert isinstance(out, xr.Dataset)
    assert set(out.data_vars) >= {'stat', 'pval'}
    assert 'dist' not in out
