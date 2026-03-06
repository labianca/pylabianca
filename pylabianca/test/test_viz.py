import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from scipy.ndimage import gaussian_filter1d
import pylabianca as pln

import pytest


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


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

    # make sure that length-one dimensions are dropped
    xarr_one_cell = xarr1.isel(cell=[2])
    pln.plot_shaded(xarr_one_cell)


def test_plot_shaded_colors():
    df = pd.DataFrame({'condition': ['A'] * 13 + ['B'] * 12})
    spk = pln.utils.create_random_spikes(
        n_cells=2, n_trials=25, n_spikes=(15, 35), metadata=df)
    fr = spk.spike_density(fwhm=0.2)

    color_rgb = plt.cm.colors.to_rgb('crimson')
    ax = pln.viz.plot_shaded(fr.isel(cell=0), colors=[color_rgb])

    # check line and shaded area colors
    assert ax.lines[0].get_color() == color_rgb
    assert (ax.collections[0].get_facecolor()[0, :3] == color_rgb).all()

    colors_str = ['crimson', 'cornflowerblue']
    colors_rgb = [plt.cm.colors.to_rgb(color) for color in colors_str]
    ax = pln.viz.plot_shaded(fr.isel(cell=0), groupby='condition',
                             colors=colors_rgb)

    for idx, color_rgb in enumerate(colors_rgb):
        assert ax.lines[idx].get_color() == color_rgb
        assert (ax.collections[idx].get_facecolor()[0, :3] == color_rgb).all()


# tests


@pytest.fixture
def grouped_data():
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

    return xr.DataArray(
        data,
        dims=('trial', 'time'),
        coords={'time': time, 'cond': ('trial', conditions)}
    )


def test_plot_shaded_title_not_last_groupby(grouped_data):
    ax = pln.plot_shaded(grouped_data, groupby='cond')
    ttl = ax.get_title()
    assert ttl != 'cond = C'


@pytest.mark.parametrize(
    "colors",
    [
        ['cornflowerblue', 'magenta', 'lawngreen'],
        {'A': 'cornflowerblue', 'B': 'magenta', 'C': 'lawngreen'},
        'magenta',
    ],
)
def test_plot_shaded_color_inputs(grouped_data, colors):
    kwargs = {"colors": colors}
    if not isinstance(colors, str):
        kwargs["groupby"] = "cond"
    ax = pln.plot_shaded(grouped_data, **kwargs)

    if isinstance(colors, dict):
        expected = [
            plt.cm.colors.to_rgb(colors[c]) for c in sorted(colors)
        ]
    elif isinstance(colors, str):
        expected = [plt.cm.colors.to_rgb(colors)]
    else:
        expected = [plt.cm.colors.to_rgb(c) for c in colors]

    for line, exp_col in zip(ax.lines, expected):
        assert line.get_color() == exp_col

    for shade, exp_col in zip(ax.collections, expected):
        assert (shade.get_facecolor()[0, :3] == exp_col).all()

    if isinstance(colors, list):
        msg = 'Expected 3 colors, got 2.'
        with pytest.raises(ValueError, match=msg):
            pln.plot_shaded(
                grouped_data, groupby='cond',
                colors=['cornflowerblue', 'magenta']
            )

    if isinstance(colors, dict):
        msg = r"Missing colors for: \['C'\]"
        with pytest.raises(ValueError, match=msg):
            pln.plot_shaded(
                grouped_data, groupby='cond',
                colors={'A': 'cornflowerblue', 'B': 'magenta'}
            )

        msg = (
            'colors must be a string, list, tuple, np.ndarray, or dict, '
            "got <class 'set'>."
        )
        with pytest.raises(TypeError, match=msg):
            pln.plot_shaded(
                grouped_data, groupby='cond',
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

    # test with clusters of shape (1, n)
    clusters = [c[None, :] for c in clusters]
    ax = pln.viz.plot_shaded(fr.isel(cell=0))
    pln.viz.add_highlights(fr, clusters, pvals)

    ranges = ((0.45, 0.7),)
    checks = compare_box_ranges(ax, ranges)
    assert len(checks) == 1
    assert checks[0].sum() == 2


def test_plot_shaded_col_row_facets():
    """Test col and row faceting in plot_shaded."""
    # Create test data with multiple dimensions
    n_trials, n_times, n_cond, n_subj = 50, 100, 2, 3
    times = np.linspace(-0.5, 1.5, num=n_times)

    # Create data with different effects for different conditions and subjects
    np.random.seed(42)
    data = np.random.randn(n_trials, n_times, n_cond, n_subj)

    # Add different effects for each condition
    data[:, 40:60, 0, :] += 2.0  # condition A has positive bump
    data[:, 40:60, 1, :] -= 1.5  # condition B has negative bump

    data = gaussian_filter1d(data, sigma=5, axis=1)

    xarr = xr.DataArray(
        data,
        dims=['trial', 'time', 'condition', 'subject'],
        coords={
            'time': times,
            'condition': ['A', 'B'],
            'subject': ['S1', 'S2', 'S3']
        },
        name='fr'
    )

    def _test_axes(axes, data, times, subject=None, groupby=None):
        axes = axes.ravel()
        if groupby is not None:
            grp_vals = data.coords[groupby].values
            grp_uni = np.unique(grp_vals)
            n_grp = len(grp_uni)
        else:
            n_grp = 1

        for idx, cond in enumerate(['A', 'B']):
            ax = axes[idx]
            ax_data = data.sel(condition=cond)

            # Check that the line data matches
            assert len(ax.lines) == n_grp
            for line_idx, line in enumerate(ax.lines):
                if groupby is None:
                    expected_data = ax_data
                else:
                    query_str = f'{groupby} == "{grp_uni[line_idx]}"'
                    expected_data = ax_data.query(trial=query_str)

                expected_data = expected_data.mean(dim='trial')
                assert (line.get_xdata() == times).all()
                assert np.allclose(line.get_ydata(), expected_data.values, atol=1e-10)

            # Check title
            title = ax.get_title()
            assert f'condition = {cond}' in title
            if subject is not None:
                assert f'subject = {subject}' in title

            # Check that legend exists
            if groupby is not None:
                legend = ax.get_legend()
                assert legend is not None
                legend_labels = [txt.get_text() for txt in legend.get_texts()]
                for val in grp_vals:
                    assert val in legend_labels

    # Test 1: Column faceting only
    # -----------------------------
    use_data = xarr.sel(subject='S1')
    axes = pln.plot_shaded(use_data, col='condition')

    # test shape and data
    assert axes.shape == (1, n_cond)
    _test_axes(axes[0, :], use_data, times)

    # Test 2: Row faceting only
    # -------------------------
    use_data = xarr.sel(subject='S2')
    axes = pln.plot_shaded(use_data, row='condition')

    # test shape and data
    assert axes.shape == (n_cond, 1)
    _test_axes(axes[:, 0], use_data, times)

    # Test 3: Both row and column faceting
    # ------------------------------------
    axes = pln.plot_shaded(xarr, row='condition', col='subject')

    # test shape and data
    assert axes.shape == (n_cond, n_subj)
    for j, subj in enumerate(['S1', 'S2', 'S3']):
        ax_row = axes[:, j]
        use_data = xarr.sel(subject=subj)
        _test_axes(ax_row, use_data, times, subject=subj)

    # Test 4: Faceting with groupby
    # -----------------------------
    # Add a hemisphere coordinate to trials for grouping
    hemisphere = xr.DataArray(['left', 'right'] * 25, dims=['trial'])
    xarr_grouped = xarr.assign_coords(hemisphere=hemisphere)

    use_data = xarr_grouped.sel(subject='S1')
    axes = pln.plot_shaded(use_data, col='condition', groupby='hemisphere')

    # test shape and data
    assert axes.shape == (1, n_cond)
    _test_axes(axes[0, :], use_data, times, groupby='hemisphere')

    # Test 5: Faceting should also work for "nested" coords
    # -----------------------------------------------------
    next_axes = pln.plot_shaded(use_data, col='hemisphere',
                                groupby='condition')
    assert (next_axes[0, 0].lines[0].get_ydata()
            == axes[0, 0].lines[0].get_ydata()).all()
    assert (next_axes[0, 0].lines[1].get_ydata()
            == axes[0, 1].lines[0].get_ydata()).all()

    # Test 6: Single facet (should return single axis)
    # ------------------------------------------------
    arr_single = xarr.sel(condition=['A'])
    result = pln.plot_shaded(arr_single.sel(subject='S1'), col='condition')

    # Should return a single axis, not an array
    assert isinstance(result, plt.Axes)

    # Test 7: Axes should share x and y limits
    # ----------------------------------------
    axes = pln.plot_shaded(xarr, row='condition', col='subject')

    # Get all x and y limits
    x_lims = [ax.get_xlim() for ax in axes.ravel()]
    y_lims = [ax.get_ylim() for ax in axes.ravel()]

    # All should be the same (shared axes)
    assert all(xlim == x_lims[0] for xlim in x_lims)
    assert all(ylim == y_lims[0] for ylim in y_lims)

    # Test 8: Faceting works with colors parameter
    # --------------------------------------------
    use_data = xarr_grouped.sel(subject='S3')
    colors_dict = {'A': 'crimson', 'B': 'cornflowerblue'}
    axes = pln.plot_shaded(
        use_data, col='condition', groupby='hemisphere',
        colors={'left': 'crimson', 'right': 'cornflowerblue'}
    )

    # Check colors are applied correctly in each facet
    expected_colors = [
        plt.cm.colors.to_rgb('crimson'),
        plt.cm.colors.to_rgb('cornflowerblue')
    ]
    for idx, _ in enumerate(['A', 'B']):
        ax = axes[0, idx]
        assert len(ax.lines) == 2

        # Check line colors
        for line, exp_color in zip(ax.lines, expected_colors):
            assert line.get_color() == exp_color

    # Test 9: ValueErrors are raised
    # ------------------------------
    msg_to_match = 'Coordinate "krecik" not found.'
    with pytest.raises(ValueError, match=msg_to_match):
        pln.plot_shaded(use_data, col='krecik', groupby='condition')

    krecik = np.random.rand(n_trials, n_subj)
    use_data = xarr_grouped.assign_coords(
        krecik=(('trial', 'subject'), krecik))
    msg_to_match = 'Coordinate "krecik" does not have exactly 1 dimension'
    with pytest.raises(ValueError, match=msg_to_match):
        pln.plot_shaded(use_data, col='krecik', groupby='condition')

