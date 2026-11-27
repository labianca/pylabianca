import numpy as np
import pylabianca as pln
import pytest

from pylabianca.utils import has_numba


def simple_acorr_hist(times, bins):
    n_times = len(times)
    dist = times[:, None] - times[None, :]
    idx = np.diag_indices(n_times)
    dist[idx] = np.nan
    hist, _ = np.histogram(dist.ravel(), bins)

    return hist


def simple_xcorr_hist(times1, times2, bins):
    dist = times2[:, None] - times1[None, :]
    hist, _ = np.histogram(dist.ravel(), bins)
    return hist


def test_xcorr():
    _has_numba = has_numba()
    if _has_numba:
        from pylabianca._numba import (_xcorr_hist_auto_numba,
                                       _xcorr_hist_cross_numba)

    spk = pln.io.read_plexon_nex(pln.utils.get_fieldtrip_data())
    spk_ep = spk.to_epochs()

    # read and select
    picks = [name for name in spk.cell_names if '_wf' in name]
    spk.pick_cells(picks)

    step = 0.01
    bins = np.arange(-0.2, 0.2, step=step) + (step / 2)

    # auto-correlation
    times = spk_ep.time[1][:1000]
    hist = pln.spike_distance._xcorr_hist_auto_py(times, bins)
    hist2 = simple_acorr_hist(times, bins)

    assert (hist == hist2).all()

    if _has_numba:
        hist3 = _xcorr_hist_auto_numba(times, bins)
        assert (hist2 == hist3).all()

        times_full = spk_ep.time[1]
        hist_full1 = pln.spike_distance._xcorr_hist_auto_py(times_full, bins)
        hist_full2 = _xcorr_hist_auto_numba(times_full, bins)
        assert (hist_full1 == hist_full2).all()

    # cross-correlation
    times1 = spk_ep.time[1][:1000]
    times2 = spk_ep.time[3][:1000]
    hist1 = simple_xcorr_hist(times1, times2, bins)
    hist2 = pln.spike_distance._xcorr_hist_cross_py(times1, times2, bins)

    np.testing.assert_array_equal(hist1, hist2)

    if _has_numba:
        hist3 = _xcorr_hist_cross_numba(times1, times2, bins)
        assert (hist2 == hist3).mean() > 0.95


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_xcorr_spike_identity(backend):
    if backend == 'numba' and not has_numba():
        pytest.skip('numba is not installed')

    spikes = pln.Spikes([np.array([100, 100, 200, 300]),
                         np.array([100, 100, 205, 305])], sfreq=1000.)
    bins = np.array([-7.5, -2.5, 2.5, 7.5]) / 1000

    # Each spike at 100 ms in A pairs with both spikes at 100 ms in B.
    cross = spikes.xcorr(0, 1, bins=bins, backend=backend)
    np.testing.assert_array_equal(cross.values[0], [0, 4, 2])

    # Within A, only the two ordered pairs of distinct spikes at 100 ms remain.
    auto = spikes.xcorr(0, 0, bins=bins, backend=backend)
    np.testing.assert_array_equal(auto.values[0], [0, 2, 0])


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_xcorr_lag_sign_follows_pair_order(backend):
    if backend == 'numba' and not has_numba():
        pytest.skip('numba is not installed')

    times_a = np.array([100, 200, 300])
    spikes = pln.Spikes([times_a, times_a + 5], sfreq=1000.,
                        cell_names=['A', 'B'])
    bins = np.array([-7.5, -2.5, 2.5, 7.5]) / 1000
    hist = spikes.xcorr(['A', 'B'], ['B', 'A'], bins=bins, backend=backend)

    np.testing.assert_allclose(hist.lag.values, [-0.005, 0., 0.005])
    np.testing.assert_array_equal(hist.sel(cell='A x B'), [0, 0, 3])
    np.testing.assert_array_equal(hist.sel(cell='B x A'), [3, 0, 0])


def test_xcorr_lag_coordinates_are_bin_centers():
    time = [np.array([0., 0.]), np.array([0.1, 0.1])]
    trial = [np.array([0, 1]), np.array([0, 1])]
    spk = pln.SpikeEpochs(
        time, trial, time_limits=(-0.2, 0.2), n_trials=2)
    bins = np.array([-0.15, -0.05, 0.05, 0.15])

    xcorr = pln.spike_distance.xcorr_hist(
        spk, picks=[0], picks2=[1], bins=bins, backend='numpy')

    np.testing.assert_allclose(xcorr.lag.values, [-0.1, 0., 0.1])
    np.testing.assert_array_equal(
        xcorr.values[0, :2], [[0, 0, 1], [0, 0, 1]])
    peak_lag = xcorr.lag.values[xcorr.values[0, 0].argmax()]
    assert peak_lag == pytest.approx(0.1)
