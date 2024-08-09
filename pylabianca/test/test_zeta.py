# test ZETA test
import time
import numpy as np
import pandas as pd
from scipy.io import loadmat

import pylabianca as pln
from pylabianca._zeta import ZETA


def test_against_zetapy():
    from zetapy import zetatest2

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
    z_val, p_val, dist = ZETA(
        spk_epochs_sel, compare='orientation', n_permutations=500,
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
    assert np.abs(zeta_info['dblZETA'] - z_val[0]) < 0.25

    # if has_numba
    has_numba = pln.utils.has_numba()
    if has_numba:
        z_val_numba, p_val_numba, dist_numba = ZETA(
            spk_epochs_sel, compare='orientation', n_permutations=500,
            tmax=1., return_dist=True, backend='numba', picks=0)

        # make sure traces and pvals are similar
        np.testing.assert_allclose(
            dist['trace'], dist_numba['trace']
        )
        assert np.abs(p_val[0] - p_val_numba[0]) < (20 / n_permutations)
        assert np.abs(z_val[0] - z_val_numba[0]) < 0.25

    # test N-conditions
    stim_ori_str = np.array([str(ori) for ori in stim_ori], dtype='object')
    spk_epochs.metadata = pd.DataFrame({'orientation': stim_ori_str})

    backend = 'numba' if has_numba else 'numpy'
    z_val_n1, p_val_n1 = ZETA(
        spk_epochs, compare='orientation', n_permutations=100,
        backend='numpy', picks=0, significance='empirical')
    z_val_n2, p_val_n2 = ZETA(
        spk_epochs, compare='orientation', n_permutations=100,
        backend=backend, picks=0, significance='both')

    assert isinstance(p_val_n2, dict)
    assert 'empirical' in p_val_n2 and 'gumbel' in p_val_n2
    assert (np.abs(p_val_n1 - p_val_n2['empirical']) < (20 / 100)).all()
