import numpy as np
import pandas as pd
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


def test_plot_shaded_colors():
    df = pd.DataFrame({'condition': ['A'] * 13 + ['B'] * 12})
    spk = pln.utils.create_random_spikes(
        n_cells=2, n_trials=25, n_spikes=(15, 35), metadata=df)
    fr = spk.spike_density(fwhm=0.2)

    color_rgb = plt.cm.colors.to_rgb('crimson')
    ax = pln.viz.plot_shaded(fr.isel(cell=0), colors=[color_rgb])
    assert ax.lines[0].get_color() == color_rgb

    colors_str = ['crimson', 'cornflowerblue']
    colors_rgb = [plt.cm.colors.to_rgb(color) for color in colors_str]
    ax = pln.viz.plot_shaded(fr.isel(cell=0), groupby='condition',
                             colors=colors_rgb)
    assert ax.lines[0].get_color() == colors_rgb[0]
    assert ax.lines[1].get_color() == colors_rgb[1]


# TODO: could split this into two text with pytest parametrization
def test_plot_shaded_colors_and_title():
    conditions = ['A', 'B', 'C']
    n_trials_per_cond = 10
    conditions = np.tile(conditions, (n_trials_per_cond, 1))
    conditions = conditions.T.ravel()

    n_times = 21
    time = np.linspace(-0.5, 1.5, num=n_times)

    effect_times = {'B': (0.3, 0.7), 'C': (0.5, 0.8)}
    data = np.random.rand(len(conditions), n_times)

    for cond, time_range in effect_times.items():
        trial_msk = conditions == cond
        time_msk = (time >= time_range[0]) & (time <= time_range[1])
        n_time_msk = time_msk.sum()

        data[np.ix_(trial_msk, time_msk)] += (
            np.random.rand(n_trials_per_cond, n_time_msk)
        )

    data = xr.DataArray(
        data,
        dims=('trial', 'time'),
        coords={'time': time, 'cond': ('trial', conditions)}
    )

    # test that title is not set to last groupby value
    ax = pln.plot_shaded(data, groupby='cond')
    ttl = ax.get_title()
    assert not (ttl == 'cond = C')

    # test that string colors work
    ax = pln.plot_shaded(
        data, groupby='cond',
        colors={'A': 'cornflowerblue', 'B': 'magenta', 'C': 'lawngreen'})

    correct_colors = [
        (0.39215686274509803, 0.5843137254901961, 0.9294117647058824),
        (1.0, 0.0, 1.0),
        (0.48627450980392156, 0.9882352941176471, 0.0)
    ]

    for line, expected_color in zip(ax.lines, correct_colors):
        assert line.get_color() == expected_color

    # one color string
    ax = pln.plot_shaded(data, colors='magenta')
    assert ax.lines[0].get_color() == correct_colors[1]

    # test errors
    msg = r"Missing colors for: \['C'\]"
    with pytest.raises(ValueError, match=msg):
        pln.plot_shaded(
            data, groupby='cond',
            colors={'A': 'cornflowerblue', 'B': 'magenta'}
        )

    msg = 'Expected 3 colors, got 2.'
    with pytest.raises(ValueError, match=msg):
        pln.plot_shaded(
            data, groupby='cond',
            colors=['cornflowerblue', 'magenta']
        )

    msg = ('colors must be a string, list, tuple, np.ndarray, or dict, '
           "got <class 'set'>.")
    with pytest.raises(TypeError, match=msg):
        pln.plot_shaded(
            data, groupby='cond',
            colors={'cornflowerblue', 'magenta', 'lawngreen'}
        )


def test_plot_raster():
    spk = pln.utils.create_random_spikes(n_cells=1)
    ax = pln.viz.plot_raster(spk)

    for idx, tri in enumerate(np.unique(spk.trial[0])):
        tri_msk = spk.trial[0] == tri
        time_real = spk.time[0][tri_msk]
        time_plot = ax.collections[idx].get_positions()
        assert (time_real == time_plot).all()


def test_plot_isi():
    # currently mostly a smoke test
    spk = pln.utils.create_random_spikes(
        n_cells=7, n_trials=0, n_spikes=(35, 153))
    ax = spk.plot_isi(min_spikes=20, max_isi=1000)

    assert ax.shape == (2, 4)
    has_bars = list()
    for this_ax in ax.ravel():
        bars = this_ax.findobj(plt.Rectangle)
        has_bars.append(len(bars) >= 10)
    assert np.mean(has_bars) > 0.5

    ax = spk.plot_isi(picks=np.arange(6), min_spikes=20, max_isi=500)
    assert ax.shape == (2, 3)

    ax = spk.plot_isi(min_spikes=5)


def check_if_same_limits(axes):
    '''Helper function to test if axes have the same limits.'''
    if len(axes) == 1:
        return True, True

    y_lims = list()
    x_lims = list()
    for ax in axes:
        y_lims.append(ax.get_ylim())
        x_lims.append(ax.get_xlim())

    same_x = list()
    same_y = list()

    for xlm, ylm in zip(x_lims[1:], y_lims[1:]):
        same_x.append(x_lims[0] == xlm)
        same_y.append(y_lims[0] == ylm)

    same_x = all(same_x)
    same_y = all(same_y)
    return same_x, same_y


def test_axis_helpers():
    # axis size in pixels
    fig, ax = plt.subplots(figsize=(10, 4))
    pix_w, pix_h = pln.viz.get_axis_size_pix(ax)
    assert np.abs((pix_w / pix_h) - (10 / 4)) < 0.1

    # normalizing axis limits
    # -----------------------
    fig, ax = plt.subplots(ncols=4, nrows=2)
    axs = ax.ravel()

    # set random x and y limits
    rnd_x = np.random.random(8)
    rnd_y = np.random.random(8)
    for idx, ax in enumerate(axs):
        ax.set_xlim(0, rnd_x[idx])
        ax.set_ylim(0, rnd_y[idx])

    # check that all axes have different limits
    same_x, same_y = check_if_same_limits(axs)
    assert not same_x
    assert not same_y

    # align y axes limits (default)
    pln.viz.align_axes_limits(axs)
    same_x, same_y = check_if_same_limits(axs)
    assert not same_x
    assert same_y

    # align x axes limits
    pln.viz.align_axes_limits(axs, ylim=False, xlim=True)
    same_x, same_y = check_if_same_limits(axs)
    assert same_x


def compare_box_ranges(ax, ranges):
    ranges_test = [np.array(rng) for rng in ranges]
    tol = 0.01

    n_check = len(ranges)
    rct = ax.findobj(plt.Rectangle)

    checks = [list() for _ in range(n_check)]
    for r in rct:
        bbx = r.get_bbox()
        box_range = np.array([bbx.x0, bbx.x1])
        for idx, rng in enumerate(ranges_test):
            same_range = (np.abs(rng - box_range) <= tol).all()
            checks[idx].append(same_range)
            if same_range:
                break

    checks = [np.array(x) for x in checks]
    return checks


def test_add_highlights():
    # prepare data
    spk = pln.utils.create_random_spikes(
        n_cells=2, n_trials=50, n_spikes=(20, 50))
    fr = spk.spike_density(fwhm=0.2)

    # prepare cluster masks and p-values
    time = fr.time.values
    clst1_msk = (time > 0.45) & (time < 0.7)
    clst2_msk = (time > 0.82) & (time < 1.)
    clusters = [clst1_msk, clst2_msk]
    pvals = np.array([0.012, 0.085])

    # test with default values (p threshold is 0.05)
    ax = pln.viz.plot_shaded(fr.isel(cell=0))
    pln.viz.add_highlights(fr, clusters, pvals)

    ttl = ax.get_title()
    assert ttl == 'cell = cell000'

    ranges = ((0.45, 0.7),)
    checks = compare_box_ranges(ax, ranges)
    assert len(checks) == 1
    assert checks[0].sum() == 2

    # test with custom p_threshold
    ax = pln.viz.plot_shaded(fr.isel(cell=0))
    pln.viz.add_highlights(fr, clusters, pvals, p_threshold=0.1)

    ranges = ((0.45, 0.7), (0.82, 1.))
    checks = compare_box_ranges(ax, ranges)
    assert len(checks) == 2
    assert checks[0].sum() == 2
    assert checks[1].sum() == 2
