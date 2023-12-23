import numpy as np
import pylabianca as pln
import pytest

from pylabianca.utils import download_test_data, get_data_path


download_test_data()
data_dir = get_data_path()
ft_data = pln.utils.get_fieldtrip_data()


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
    from borsar.utils import has_numba
    if has_numba:
        from pylabianca._numba import (_xcorr_hist_auto_numba,
                                    _xcorr_hist_cross_numba)

    spk = pln.io.read_plexon_nex(ft_data)
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

    if has_numba:
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

    # for some reason sometimes one spike gets in the adjacent bin
    assert (hist1 == hist2).mean() > 0.95

    if has_numba:
        hist3 = _xcorr_hist_cross_numba(times1, times2, bins)
        assert (hist2 == hist3).mean() > 0.95
