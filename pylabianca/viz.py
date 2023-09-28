import numpy as np
import matplotlib.pyplot as plt


# TODO - ! x_dim='auto' (infers which is the likely x dimension) !
# TODO - ! also support mask !
# TODO - allow for colors (use ``mpl.colors.to_rgb('C1')`` etc.)
# TODO - get y axis from xarray data name (?)
# TODO - the info about "one other dimension" (that is reduced) seems to be no
#        longer accurate
def plot_spike_rate(frate, reduce_dim='trial', groupby=None, ax=None,
                    x_dim='time', legend=True, legend_pos=None, colors=None,
                    labels=True):
    '''Plot spike rate with standard error of the mean.

    Parameters
    ----------
    frate : xarray.DataArray
        Xarray with at least two dimensions: one is plotted along the x axis
        (this is controlled with ``x_dim`` argument); the other is reduced
        by averaging (see ``reduce_dim`` argument below). The averaged
        dimension also gives rise to the standard error of the mean, which is
        plotted as a shaded area.
    reduce_dim : str
        The dimension to reduce (average). The standard error is also computed
        along this dimension. The default is ``'trial'``.
    groupby : str | None
        The dimension (or sub-dimension) to use as grouping variable plotting
        the spike rate into separate lines. The default is ``None``, which
        does not perform grouping.
    ax : matplotlib.Axes | None
        Axis to plot into. The default is ``None`` which creates an new axis.
    x_dim : str
        Dimension to use for the x axis. The default is ``'time'``.
    legend : bool
        Whether to plot the legend.
    legend_pos : str | None
        Legend position (standard matplotlib names like "upper left"). Defaults
        to ``None`` which uses ``'best'`` position.
    colors : list of arrays | dictionary of arrays | None
        List of RGB arrays to use as colors for condition groups. Can also be
        a dictionary linking condition names / values and RBG arrays. Default
        is ``None`` which uses the default matplotlib color cycle.
    labels : bool
        Whether to add labels to the axes.

    Returns
    -------
    ax : matplotlib.Axes
        Axis with the plot.
    '''
    if ax is None:
        _, ax = plt.subplots()

    if ('cell' in frate.coords and not reduce_dim == 'cell'
        and len(frate.cell.shape) > 0):
        if len(frate.coords['cell'] == 1):
            frate = frate.isel(cell=0)
        else:
            msg = ('DataArray contains more than one cell to plot - this is '
                    'not supported.')
            raise RuntimeError(msg)

    # compute mean, std and n
    if groupby is not None:
        frate = frate.groupby(groupby)

    # calculate standard error of the mean
    avg = frate.mean(dim=reduce_dim)
    std = frate.std(dim=reduce_dim)
    n = frate.count(dim=reduce_dim)
    std_err = std / np.sqrt(n)
    ci_low = avg - std_err
    ci_high = avg + std_err

    # handle colors
    if colors is not None:
        if groupby is not None:
            group_names = avg.coords[groupby].values
            n_groups = len(group_names)
        else:
            n_groups = 1
            group_names = ['base']

        assert len(colors) == n_groups
        if isinstance(colors, (list, tuple, np.ndarray)):
            assert all(isinstance(x, (list, np.ndarray)) for x in colors)
            colors = {group: color for group, color in zip(group_names, colors)}
        else:
            assert all(name in colors.keys() for name in group_names)
            assert all(isinstance(x, (list, tuple, np.ndarray))
                       for x in colors.values())

    # plot each line with error interval
    if groupby is not None:
        sel = {groupby: 0}
        for val in avg.coords[groupby]:
            val = val.item()
            sel[groupby] = val

            add_arg = {'color': colors[val]} if colors is not None else dict()
            lines = avg.sel(**sel).plot(label=val, ax=ax, **add_arg)
            ax.fill_between(avg.coords[x_dim], ci_low.sel(**sel),
                            ci_high.sel(**sel), alpha=0.3, linewidth=0,
                            **add_arg)
    else:
        add_arg = {'color': colors['base']} if colors is not None else dict()
        lines = avg.plot(ax=ax, **add_arg)
        ax.fill_between(avg.coords[x_dim], ci_low, ci_high, linewidth=0,
                        alpha=0.3, **add_arg)


    if groupby is not None and legend:
        pos = 'best' if legend_pos is None else legend_pos
        ax.legend(title=f'{groupby}:', loc=pos)

    if labels:
        if x_dim == 'time':
            ax.set_xlabel('Time (s)', fontsize=14)

        ax.set_ylabel('Spike rate (Hz)', fontsize=14)
        add_txt = '' if groupby is None else f' grouped by {groupby}'
        ttl = 'Firing rate' + add_txt
        ax.set_title(ttl, fontsize=16)

    return lines[0].axes


# TODO: move to sarna sometime
# TODO: make sure the pbar is cleared ... (tqdm._instances.clear() may help)
def check_modify_progressbar(pbar, total=None):
    '''Reset ``pbar`` and change its total if it is a tqdm progressbar.
    Otherwise create new progressbar.

    Parameters
    ----------
    pbar : bool | str | tqdm progressbar
        Progressbar to modify or instructions on whether (if bool) or of what
        kind (if str) a progressbar should be created.
    total : int | None
        Total number of steps in the progressbar.

    Returns
    -------
    pbar : tqdm progressbar | empty sarna progressbar
        Modified or newly created progressbar. Can also be an empty progressbar
        from sarna - a progressbar that has some of the tqdm's API but does not
        do anything.
    '''
    if isinstance(pbar, bool):
        if pbar:
            from tqdm.auto import tqdm
            pbar = tqdm(total=total)
        else:
            from sarna.utils import EmptyProgressbar
            pbar = EmptyProgressbar(total=total)
    elif isinstance(pbar, str):
        if pbar == 'notebook':
            from tqdm.notebook import tqdm
        elif pbar == 'text':
            from tqdm import tqdm
        pbar = tqdm(total=total)
    else:
        from tqdm.notebook import tqdm_notebook
        if isinstance(pbar, tqdm_notebook):
            pbar.reset(total=total)
    return pbar


# TODO:
# - [x] ! fix x axis units !
# - [ ] kind='line' ?
# - [ ] datashader backend?
# - [ ] allow to plot multiple average waveforms as lines
def plot_waveform(spk, pick=0, upsample=False, ax=None, labels=True,
                  y_bins=100, times=None):
    '''Plot waveform heatmap for one cell.

    Parameters
    ----------
    spk : pylabianca.spikes.Spikes | pylabianca.spikes.SpikeEpochs
        Spike object to use.
    pick : int | str
        Cell index to plot waveform for.
    upsample : bool | float
        Whether to upsample the waveform (defaults to ``False``). If
        ``True`` the waveform is upsampled by a factor of three. Can also
        be a value to specify the upsampling factor.
    ax : matplotlib.Axes | None
        Axis to plot to. By default opens a new figure.
    labels : bool
        Whether to add labels to the axes.
    y_bins : int
        How many bins to use for the y axis.

    Returns
    -------
    ax : matplotlib.Axes
        Axis with the waveform plot.
    '''

    hist, _, ybins, time_edges = _calculate_waveform_density_image(
        spk, pick, upsample, y_bins, times=times
    )
    max_alpha = np.percentile(hist[hist > 0], 45)
    max_lim = np.percentile(hist[hist > 0], 99)

    alpha2 = hist.T * (hist.T <= max_alpha) / max_alpha
    alpha_sum = (hist.T > max_alpha).astype('float') + alpha2
    alpha_sum[alpha_sum > 1] = 1

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(hist.T, alpha=alpha_sum, vmax=max_lim, origin='lower',
              extent=(time_edges[0], time_edges[-1], ybins[0], ybins[-1]),
              aspect='auto')
    if labels:
        time_unit = 'samples' if times is None else 'ms'
        ax.set_xlabel(f'Time ({time_unit})', fontsize=14)
        ax.set_ylabel('Amplitude ($\mu$V)', fontsize=14)
    return ax


def _calculate_waveform_density_image(spk, pick, upsample, y_bins,
                                      density=True, y_range=None, times=None):
    '''Helps in calculating 2d density histogram of the waveforms.'''
    from pylabianca.utils import _deal_with_picks

    pick = _deal_with_picks(spk, pick)[0]
    n_spikes, n_samples = spk.waveform[pick].shape
    waveform = spk.waveform[pick]
    if upsample:
        if isinstance(upsample, bool):
            upsample = 3
        from scipy import interpolate

        x = np.arange(n_samples)
        interp = interpolate.interp1d(x, waveform, kind='cubic',
                                      assume_sorted=True)

        new_x = np.linspace(0, n_samples - 1, num=n_samples * upsample)
        waveform = interp(new_x)
        n_samples = len(new_x)
    else:
        upsample = 1

    x_coords = np.tile(np.arange(n_samples), (n_spikes, 1))

    # sample_time = 1000 / sfreq  (combinato)
    # sample_time = 1 if times is None else np.diff(times).mean()
    # sample_edge = -94  # -19 for combinato
    # time_edges = [sample_edge * sample_time,
    #               (n_samples / upsample + (sample_edge - 1)) * sample_time]

    if times is not None:
        time_edges = [times[0], times[-1]]
    else:
        time_edges = [0, n_samples / upsample]

    xs = x_coords.ravel()
    ys = waveform.ravel()
    nan_mask = np.isnan(ys)
    if nan_mask.any():
        range = [[np.nanmin(xs), np.nanmax(xs)],
                 [np.nanmin(ys), np.nanmax(ys)]]
        xs = xs[~nan_mask]
        ys = ys[~nan_mask]
    else:
        range = None

    if y_range is not None:
        if nan_mask.any():
            range[1] = y_range
        else:
            range = [[np.min(xs), np.max(xs)], y_range]

    hist, xbins, ybins = np.histogram2d(xs, ys, bins=[n_samples, y_bins],
                                        range=range, density=density)

    return hist, xbins, ybins, time_edges


# TODO: add order=False for groupby?
def plot_raster(spk, pick=0, groupby=None, ax=None, labels=True):
    '''Show spike rasterplot.

    Parameters
    ----------
    spk : pylabianca.spikes.Spikes | pylabianca.spikes.SpikeEpochs
        Spike object to use.
    pick : int | str
        Cell index or name to plot raster for.
    groupby : str | None
        If not None, group the raster by given variable (requires present
        ``.metadata`` field of the ``spk``).
    ax : matplotlib.Axes | None
        Matplotlib axis to plot to. If ``None`` a new figure is opened.
    labels : bool
        Whether to add labels to the axes.

    Returns
    -------
    ax : matplotlib.Axes
        Axis with the raster plot.
    '''

    if ax is None:
        _, ax = plt.subplots()

    spk_cell = spk.copy().pick_cells(picks=pick)

    tri_spikes = list()
    colors = list()

    if groupby is not None:
        values = np.unique(spk_cell.metadata.loc[:, groupby])
    else:
        values = [None]

    for idx, value in enumerate(values):
        img_color = f'C{idx}'
        if groupby is not None:
            trials = (spk_cell.metadata.query(f'{groupby} == {value}')
                      .index.values)
        else:
            if spk_cell.metadata is not None:
                trials = spk_cell.metadata.index.values
            else:
                trials = np.arange(spk_cell.n_trials)

        for trial in trials:
            msk = spk_cell.trial[0] == trial
            tri_spikes.append(spk_cell.time[0][msk])
            colors.append(img_color)

    ax.eventplot(tri_spikes, colors=colors)

    # set y limits
    n_trials = len(tri_spikes)
    ax.set_ylim(-1, n_trials)

    # set x limits
    xlim = spk.time_limits + np.array([-0.05, 0.05])
    ax.set_xlim(xlim)

    if labels:
        ax.set_xlabel('Time (s)', fontsize=14)
        ax.set_ylabel('Trial', fontsize=14)

    return ax


def plot_spikes(spk, frate, groupby=None, df_clst=None, pick=0,
                min_pval=0.001, ax=None):
    '''Plot average spike rate and spike raster.

    spk : pylabianca.spikes.SpikeEpochs
        Spike object to use.
    frate : xarray.DataArray
        Firing rate xarray in the format returned by ``spk.spike_rate()`` or
        ``spk.spike_density()``.
    groupby : str | None
        How to group the plots. If None, no grouping is done.
    df_clst : pandas.DataFrame | None
        DataFrame with cluster time ranges and p values. If None, no cluster
        information is shown.
    pick : int | str
        Name or index of the cell to plot.
    min_pval : float
        Minimum p-value of cluster to mark on the plot. Only used if
        ``df_clst`` is not None.
    ax: matplotlib.Axes | None
        Two axes to plot on: first is used for average firing ratem the second
        is used for raster plot. If None, a new figure is created.

    Returns
    -------
    fig : matplotlib.Figure
        Figure with the plots.
    '''
    # select cell from frate
    if isinstance(pick, str):
        cell_name = pick
        this_frate = frate.sel(cell=cell_name)
    else:
        this_frate = frate.isel(cell=pick)
        cell_name = this_frate.coords['cell'].item()

    # plot
    if ax is None:
        gridspec_kw = {'bottom': 0.15, 'left': 0.15}
        fig, ax = plt.subplots(nrows=2, gridspec_kw=gridspec_kw)
    else:
        assert(len(ax) == 2)
        fig = ax[0].figure
    plot_spike_rate(this_frate, groupby=groupby, ax=ax[0])

    # add highlight
    if df_clst is not None:
        this_clst = df_clst.query(f'neuron == "{cell_name}"')
        pvals = this_clst.pval.values
        masks = [_create_mask_from_window_str(twin, this_frate)
                 for twin in this_clst.window.values]
        add_highlights(this_frate, masks, pvals, ax=ax[0],
                       min_pval=min_pval)

    plot_raster(spk.copy().pick_cells(cell_name), pick=0,
                groupby=groupby, ax=ax[1])
    ylim = ax[1].get_xlim()
    ax[0].set_xlim(ylim)

    ax[0].set_xlabel('')
    ax[0].set_ylabel('Spike rate (Hz)', fontsize=12)
    ax[1].set_xlabel('Time (s)', fontsize=12)
    ax[1].set_ylabel('Trials', fontsize=12)

    return fig


def _create_mask_from_window_str(window, frate):
    twin = [float(x) for x in window.split(' - ')]
    mask = (frate.time.values >= twin[0]) & (frate.time.values <= twin[1])
    return mask


# consider moving to sarna?
# TODO: could infer x coords from plot (if lines are already present)
def add_highlights(arr, clusters, pvals, p_threshold=0.05, ax=None,
                   min_pval=0.001, bottom_extend=True, pval_text=True,
                   text_props=None):
    '''Highlight significant clusters along the last array dimension.

    Parameters
    ----------
    arr : xarray.DataArray | numpy.ndarray
        xarray Data array or numpy array. Used only for axis coordinates.
        If xarray, the last dimension is taken as x axis coordinates.
        If numpy array, the values are assumed to be the x axis coordinates.
    clusters : list of np.array
        List of boolean arrays, where each array contains cluster membership
        information (which points along the last array dimension contribute
        to the given cluster).
    pvals : list-like
        List or array of cluster p values.
    p_threshold : float
        Alpha significance threshold. Clusters with p value below this
        threshold will be shown.
    ax : matplotlib.Axes | None
        Axis to plot to. Optional, defaults to ``None`` - which creates the
        axis automatically.
    min_pval : float
        Minimum meaningful p value. P values below this value will be displayed
        as this value. This is to avoid p = 0, which can result in the cluster
        based permutation test, when no cluster from the null distribution had
        stronger summary statistic than the observed cluster. ``min_pval`` is
        best defined as ``1 / n_permutations``.
    bottom_extend : bool
        Whether to extend the lower limits of y axis when adding bottom
        significance bars. Defaults to ``True``.
    pval_text : bool
        Whether to add p value text boxes to respective cluster ranges in the
        plot. Defaults to ``True``.
    text_props : dict | None
        Dictionary with text properties for p value text boxes. If None,
        defaults to
        ``{'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.75, edgecolor='gray'}``.

    Returns
    -------
    ax : matplotlib.Axes
        Axis with the plot.
    '''
    import sarna
    try:
        import xarray as xr
        has_xarray = True
    except ModuleNotFoundError:
        has_xarray = False

    if text_props is None and pval_text:
        text_props = dict(boxstyle='round', facecolor='white', alpha=0.75,
                          edgecolor='gray')

    if ax is None:
        ax = plt.gca()

    if pvals is None:
        return

    pvals_significant = pvals < p_threshold

    if isinstance(arr, np.ndarray):
        x_coords = arr
    elif has_xarray and isinstance(arr, xr.DataArray):
        last_dim = arr.dims[-1]
        x_coords = arr.coords[last_dim].values
    else:
        raise RuntimeError('The `arr` has to be either a numpy array or'
                           'xarray.DataArray.')

    extend_textbox_x = 4

    clusters_x_sorting = np.argsort([np.where(x)[0][0] for x in clusters])

    if pvals_significant.any():
        import borsar

        ylm = ax.get_ylim()
        y_rng = np.diff(ylm)[0]

        # TODO: better estimate text y pos from textbox?
        y_step = 0.075 * y_rng
        text_y = ylm[1] - y_step
        ax.set_ylim([ylm[0], ylm[1] + 2 * y_step])

        sig_idx = np.where(pvals_significant)[0]

        sig_clusters = [clusters[ix] for ix in sig_idx]

        try:
            sarna.viz.highlight(
                x_coords, sig_clusters, axis=ax,
                bottom_bar=True, bottom_extend=bottom_extend
            )
        except TypeError:
            sarna.viz.highlight(
                x_coords, sig_clusters, ax=ax,
                bottom_bar=True, bottom_extend=bottom_extend
            )

        texts = list()
        for ix in clusters_x_sorting:
            if not pvals_significant[ix]:
                continue

            this_pval = pvals[ix]
            text_x = np.mean(x_coords[clusters[ix]])

            if this_pval < min_pval:
                p_txt = 'p < {:.3f}'.format(min_pval)
            else:
                p_txt = borsar.stats.format_pvalue(this_pval)

            this_text = ax.text(
                text_x, text_y, p_txt,
                bbox=text_props, horizontalalignment='center'
            )
            try:
                textbox = this_text.get_window_extent()
            except RuntimeError:
                ax.figure.canvas.draw()
                textbox = this_text.get_window_extent()

            textbox.set_points(
                np.array([
                    [textbox.x0 - extend_textbox_x, textbox.y0],
                    [textbox.x1 + extend_textbox_x, textbox.y1]
                ])
            )

            if len(texts) > 0:
                while textbox.count_overlaps(texts):
                    x, y = this_text.get_position()
                    this_text.set_position((x, y - y_step))
                    textbox = this_text.get_window_extent()
            texts.append(textbox)

    return ax


# - [ ] combine with layout functions from sarna
def align_axes_limits(axes=None, ylim=True, xlim=False):
    '''Align the limits of all ``axes``.'''
    if axes is None:
        axes = plt.gcf().get_axes()

    do_lim = dict(x=xlim, y=ylim)
    limits = dict(x=[np.inf, -np.inf], y=[np.inf, -np.inf])
    iter = (axes if isinstance(axes, (list, np.ndarray))
            else axes.values() if isinstance(axes, dict) else None)

    for ax in iter:
        get_lim = dict(x=ax.get_xlim, y=ax.get_ylim)
        for lim in ('x', 'y'):
            if do_lim[lim]:
                this_lim = get_lim[lim]()
                if limits[lim][0] > this_lim[0]:
                    limits[lim][0] = this_lim[0]
                if limits[lim][1] < this_lim[1]:
                    limits[lim][1] = this_lim[1]

    iter = (axes if isinstance(axes, (list, np.ndarray))
            else axes.values() if isinstance(axes, dict) else None)
    for ax in iter:
        set_lim = dict(x=ax.set_xlim, y=ax.set_ylim)
        for lim in ('x', 'y'):
            if do_lim[lim]:
                set_lim[lim](limits[lim])


# TODO - move this to separate submodule .waveform ?
def calculate_perceptual_waveform_density(spk, cell_idx):
    # get waveform 2d histogram image
    hist, xbins, ybins, time_edges = (
        _calculate_waveform_density_image(
            spk, cell_idx, False, 100)
    )

    # correct y range
    # TODO: would be cheaper to calculate the range in the function above
    sm = hist.sum(axis=0)
    msk = sm < 1e-6
    n_smp = len(msk)

    start_ix, end_ix = 0, n_smp

    if msk[0]:
        # trim from front
        for ix in range(n_smp):
            if not msk[ix]:
                break
        start_ix = ix

    if msk[-1]:
        # trim from back
        for ix in range(-1, -n_smp, -1):
            if not msk[ix]:
                break
        end_ix = ix

    # create the 2d hist image with corrected y range
    y_range = [ybins[start_ix], ybins[end_ix]]
    hist, xbins, ybins, time_edges = (
        _calculate_waveform_density_image(
            spk, cell_idx, False, 100, y_range=y_range)
    )

    # calculate dns
    # TODO: a better way would be to calculate mean from a spatial window
    #       around max, not the top N values
    hist_sort = np.sort(hist)[:, ::-1]
    vals = (hist_sort[:, :15] / hist_sort.max()).mean(axis=-1)
    dns = vals.mean()

    return dns
