import os.path as op

import numpy as np
import pytest

import pylabianca as pln
from pylabianca.utils import has_elephant, find_index
from pylabianca.testing import ft_data, spk_epochs


@pytest.mark.skipif(not has_elephant(), reason="requires elephant")
def test_firing_rate_against_elephant(spk_epochs):
    import borsar
    from scipy.stats import pearsonr
    import quantities as q
    import elephant.statistics as elestat

    # test .to_neo() and .to_spiketools()
    spike_train = spk_epochs.to_neo(0)
    spikes = pln.io.to_spiketools(spk_epochs, picks=0)

    for tri_idx in range(3):
        assert(len(spike_train[tri_idx] == len(spikes[tri_idx])))

    # compare mean firing rate
    avg_fr = spk_epochs.spike_rate(tmin=-2.75, tmax=3., step=False)
    avg_rate = elestat.mean_firing_rate(spike_train[0])

    assert avg_rate.item() == avg_fr[0, 0].item()

    # compare spike rates
    fr = spk_epochs.spike_rate(winlen=0.25)
    kernel = elestat.kernels.RectangularKernel(sigma=0.0715 * q.second)
    rate = elestat.instantaneous_rate(
        spike_train[0], 1 / 500. * q.second, kernel=kernel)

    dist = np.array(rate.times)[:, None] - fr.time.values[None, :]
    idx = np.abs(dist).argmin(axis=0)
    sel_rate = rate.magnitude[idx].ravel()

    avg_diff = np.mean(sel_rate - fr[0, 0].values)
    assert avg_diff < 0.5

    rval, _ = pearsonr(sel_rate, fr[0, 0].values)
    assert rval > 0.998

    # compare spike density
    sigma = 0.075
    fwhm = pln.spike_rate._FWHM_from_window(gauss_sd=sigma)
    fr = spk_epochs.spike_density(fwhm=fwhm)

    kernel = elestat.kernels.GaussianKernel(sigma=sigma * q.second)

    for cell_idx in range(spk_epochs.n_units()):
        spike_train = spk_epochs.to_neo(cell_idx)
        for tri_idx in range(spk_epochs.n_trials):
            rate = elestat.instantaneous_rate(
                spike_train[tri_idx], 1 / 500. * q.second, kernel=kernel)

            idx = find_index(np.array(rate.times), fr.time[[0, -1]].values)
            elephant_fr = rate.magnitude.ravel()[idx[0]:idx[1] + 1]
            rval, _ = pearsonr(fr[cell_idx, tri_idx].values, elephant_fr)
            assert rval > 0.998

    # test a case where times vector and n_steps did not align (lead to error)
    this_spk = spk_epochs.copy().crop(tmin=-1., tmax=2.)
    this_spk.spike_rate(winlen=0.35, step=0.05)


def test_time_centering():
    spk = pln.utils.create_random_spikes(n_cells=3, n_trials=10)

    fr = spk.spike_rate()
    zero_dist = np.abs(fr.time - 0).min().item()

    fr2 = spk.spike_rate(center_time=True)
    zero_dist2 = np.abs(fr2.time - 0).min().item()

    assert zero_dist > zero_dist2
    assert zero_dist2 == 0.


def test_frate_writes_to_netcdf4(spk_epochs, tmp_path):
    import xarray as xr

    fr = spk_epochs.spike_rate(winlen=0.25, step=0.05)

    fname = 'test_spike_rate.nc'
    fpath = op.join(tmp_path, fname)
    fr.to_netcdf(fpath)
    fr2 = xr.load_dataarray(fpath)

    assert fr.equals(fr2)


def test_frate_no_spikes():
    # test 01 - 1 cell with 0 spikes
    # ------------------------------
    spk = pln.SpikeEpochs([np.array([])], [np.array([])])

    # compute firing rate
    # BUT: could in future raise similar error (or warning) than spk.spike_denisty
    #      as there is not enough time points for one FR step ...
    fr = spk.spike_rate()

    assert fr.shape == (1, 0, 0)

    msg = (r"Convolution kernel length \(151 samples\) exceeds the length of "
           r"the data \(0 samples\).")
    with pytest.raises(ValueError, match=msg):
        spk.spike_density()

    # test 02 - 2 cells, one with spikes, one without
    # -----------------------------------------------
    tri = np.array([0, 0, 0, 1, 1, 2])
    tim = np.array([-0.1, 0.8, 1.1, 0.2, 0.6, -0.3])
    spk = pln.SpikeEpochs([tim, np.array([])], [tri, np.array([])])

    fr = spk.spike_rate()
    assert fr.shape == (2, 3, 117)

    fd = spk.spike_density()
    assert fd.shape == (2, 3, 551)
