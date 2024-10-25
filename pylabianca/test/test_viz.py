import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from scipy.ndimage import gaussian_filter1d
import pylabianca as pln

import pytest


def test_plot_shaded():
    # create random xarray
    # cell x trial x time

    n_cells, n_trials, n_times = 12, 100, 500
    times = np.linspace(-0.5, 1.5, num=n_times)
    data = np.random.randn(n_cells, n_trials, n_times)
    data = gaussian_filter1d(data, sigma=15)

    # add cell names
    cell_names = [f'cell_{i:02d}' for i in range(n_cells)]

    xarr1 = xr.DataArray(
        data, dims=['cell', 'trial', 'time'],
        coords={'cell': cell_names, 'time': times})

    msg_to_match = 'DataArray contains too many dimensions'
    with pytest.raises(ValueError, match=msg_to_match):
        pln.plot_shaded(xarr1)

    ax = pln.plot_shaded(xarr1, groupby='cell')

    # has n_cells lines, each n_times long
    assert len(ax.lines) == n_cells
    for idx, line in enumerate(ax.lines):
        # line has correct x dim coords
        assert (line.get_xdata() == times).all()

    # line has correct y values
    assert (line.get_ydata() == xarr1.isel(cell=idx).mean(dim='trial')).all()

    # check that there is legend present with correct labels
    leg = ax.get_legend()
    leg.get_title().get_text() == 'cell:'

    for idx, txt in enumerate(leg.get_texts()):
        this_text = txt.get_text()
        assert this_text == cell_names[idx]

    # groupby dim should be removed from auto-finding reduce_dim
    ax1 = pln.plot_shaded(xarr1, groupby='trial', legend=False)
    ax2 = pln.plot_shaded(xarr1, groupby='trial', reduce_dim='cell',
                          legend=False)

    assert len(ax1.lines) == len(ax2.lines) == n_trials

    for line1, line2 in zip(ax1.lines, ax2.lines):
        assert (line1.get_xdata() == line2.get_xdata()).all()
        assert (line1.get_ydata() == line2.get_ydata()).all()


def test_plot_raster():
    spk = pln.utils.create_random_spikes(n_cells=1)
    ax = pln.viz.plot_raster(spk)

    for idx, tri in enumerate(np.unique(spk.trial[0])):
        tri_msk = spk.trial[0] == tri
        time_real = spk.time[0][tri_msk]
        time_plot = ax.collections[idx].get_positions()
        assert (time_real == time_plot).all()


def test_plot_isi():
    # currently a smoke test
    spk = pln.utils.create_random_spikes(
        n_cells=7, n_trials=0, n_spikes=(35, 153))
    ax = spk.plot_isi(min_spikes=20, max_isi=1000)

    assert ax.shape == (2, 4)
    for this_ax in ax.ravel():
        bars = this_ax.findobj(plt.Rectangle)
        assert len(bars) >= 10

    ax = spk.plot_isi(picks=np.arange(6), min_spikes=20, max_isi=500)
    assert ax.shape == (2, 3)
