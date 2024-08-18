import numpy as np
import xarray as xr

from scipy.ndimage import gaussian_filter1d
import pylabianca as pln


def test_plot_shaded():
    # create random xarray
    # cell x trial x time

    n_cells, n_trials, n_times = 12, 100, 500
    times = np.linspace(-0.5, 1.5, num=n_times)
    data = np.random.randn(n_cells, n_trials, n_times)
    data = gaussian_filter1d(data, sigma=15)

    # add cell names

    xarr1 = xr.DataArray(
        data, dims=['cell', 'trial', 'time'],
        coords={'time': times})

    # %%

    # error when doing
    # pln.plot_shaded(xarr1)
    #
    # it should test that the array is too big (without groupby dim)
    # or in future plot multi-lines for each element of the left fim (?)

    ax = pln.plot_shaded(xarr1, groupby='cell')

    # has n_cells lines, each n_times long
    assert len(ax.lines) == n_cells
    for idx, line in enumerate(ax.lines):
        # line has correct x dim coords
        assert (line.get_xdata() == times).all()

    # line has correct y values
    assert (line.get_ydata() == xarr1.isel(cell=idx).mean(dim='trial')).all()

    # LATER: check that there is legend present with correct labels

    # %%
    # groupby dim should be removed from auto-finding reduce_dim
    ax1 = pln.plot_shaded(xarr1, groupby='trial', legend=False)

    ax2 = pln.plot_shaded(xarr1, groupby='trial', reduce_dim='cell', legend=False)

    assert len(ax1.lines) == len(ax2.lines) == n_trials

    for line1, line2 in zip(ax1.lines, ax2.lines):
        assert (line1.get_xdata() == line2.get_xdata()).all()
        assert (line1.get_ydata() == line2.get_ydata()).all()